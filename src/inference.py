import os
import re
import glob
import pickle
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from threading import Thread
from skimage.morphology import area_opening, area_closing
import warnings
import rasterio as r
from rasterio.enums import Resampling
import time
import tkinter as tk

from src.models import UNetSemanticSegmentation, UNetSemanticSegmentationS2
from src.helpers import DataScaler, get_slope, get_points_and_distance_map, tile_cleaner

def inference(model_path:str, input_info:dict, result_path:str, clean_result:bool=False, ui=None, device=torch.device('cpu'), 
    bayesian_dropout:bool=False, show_plot:bool=False):
    '''
    input_info should have the following keys:
        input_files:list
            sentinel-1: [image1: VH-VV], sentinel-2: [image1: blue-green-red-nir-swir16-swir22], planetscope: [image1: B-G-R-N, image2: aux]
        data_type:str
            "sentinel-1", "sentinel-2", or "planetscope"
        sar_is_dB:bool
            wheter or not SAR data is in decibels
    '''
    time_start = time.time()
    # check if it's running the ui or terminal version
    if not ui is None:
        ui['progress_bar']['value'] = 0

    # if it is a model that rely on neural networks...
    if not model_path in ('models/Otsu_Threshold', 'models/WF2_unetv2_bgriswirs'):
        # innitially load the model weights
        try:
            model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except pickle.UnpicklingError:
            # some checkpoints (e.g. UNet-S2.pt) store extra numpy-typed training
            # metadata (loss curves) that the safe unpickler rejects; model files
            # under models/ are trusted local input to this tool.
            model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        # load model for inference
        match model_path:
            case 'models/UNet-S1.pt':
                model = UNetSemanticSegmentation(in_channels=2, out_channels=3, base=32).to(device)
            case 'models/DistanceMap.pt':
                model = UNetSemanticSegmentation(in_channels=4, out_channels=2, base=32).to(device)
            case 'models/UNet-PlanetScope.pt':
                model = UNetSemanticSegmentation(in_channels=4, out_channels=2, base=32).to(device)
            case 'models/UNet-S2.pt':
                model = UNetSemanticSegmentationS2(in_channels=6, out_channels=3, base=64).to(device)
                # this checkpoint was saved without the tiling metadata the other
                # models carry; these mirror the values used to train/predict with it.
                model_data.setdefault('chip_size', 256)
                model_data.setdefault('chip_border', 94)
                model_data.setdefault('use_terrain', False)
                model_data.setdefault('use_distance_map', False)
                model_data.setdefault('data_preprocessing', 'none')
            case _:
                raise ValueError(f'{model_path} is not a valid model name.')

        model.load_state_dict(model_data['model_state_dict'])
        model.eval()

        # activating dropouts if Bayesian dropout option was given
        if bayesian_dropout:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        # load input spectral data
        dataset, nodata_mask = load_input_data(
            input_files = input_info['input_files'],
            data_type = input_info['data_type'],
            sar_data_is_in_dB = input_info['sar_is_dB'] if 'sar_is_dB' in input_info.keys() else True,
            processing_type = model_data['data_preprocessing']
        )
        dataset = np.stack([np.pad(dataset[i], pad_width=model_data['chip_border'], mode='reflect') for i in range(len(dataset))])

        # getting terrain slope
        if model_data['use_terrain']:
            if not ui is None:
                ui['button_run']["text"] = "Downloading DEM..."
            slope = get_slope(input_info['input_files'][0])
            # dataset was padded by chip_border above so tiles can overlap at the
            # edges; slope must match that padded shape or edge tiles' slices
            # come back smaller than chip_size.
            slope = np.pad(slope, pad_width=model_data['chip_border'], mode='reflect')

            # load data scaler
            data_scaler = DataScaler()

        # define placeholder for final result
        inference = np.zeros([3, dataset.shape[1], dataset.shape[2]])
        if bayesian_dropout:
            uncertainty = np.zeros([dataset.shape[1], dataset.shape[2]], dtype=np.float32)
        
        # define tiles for processing and batches as well
        if not ui is None:
            ui['button_run']["text"] = "Getting tiles..."
        
        starting_coordinates_batches = build_tile_batches(dataset.shape[1], dataset.shape[2], model_data['chip_size'], model_data['chip_border'], nodata_mask)

        # Do the inference!
        if not ui is None:
            ui['button_run']["text"] = "Doing the inference..."
        
        batch_count = 0
        
        for batch in tqdm(starting_coordinates_batches, desc='Inference:'):
            batch_input_data = []
            batch_slope = []
            batch_dm = []
            for i,j in batch:
                # loads sar data
                input_data = dataset[:, i:i+model_data['chip_size'], j:j+model_data['chip_size']]
                batch_input_data.append(input_data)

                # adding terrain if necessary
                if model_data['use_terrain']:
                    t = np.stack([np.zeros([model_data['chip_size'],model_data['chip_size']]), slope[i:i+model_data['chip_size'], j:j+model_data['chip_size']]])
                    match model_data['data_preprocessing']:
                        case 'normalize':
                            t = data_scaler.normalize_data('terrain', t)
                        case 'scale':
                            t = data_scaler.scale_data('terrain', t)
                        case _:
                            raise ValueError(f'Model metadata data_preprocessing is {model_data["data_preprocessing"]} (not supported).')
                    batch_slope.append(t[1])
                
                # doing distance map if necessary
                if model_data['use_distance_map']:
                    coords_f, coords_n, dmap = get_points_and_distance_map(input_data, t, max_points_per_class_map=100)
                    batch_dm.append(dmap)
            
            batch_input_data = np.array(batch_input_data)
            batch_input_data[np.isnan(batch_input_data)] = 0

            batch_data = torch.Tensor(np.asarray(batch_input_data, dtype=np.float32))

            if model_data['use_terrain']:
                batch_slope = torch.Tensor(np.array(batch_slope)[:,None,:,:])
                batch_data = torch.cat([batch_data, batch_slope], dim=1)
            
            if model_data['use_distance_map']:
                batch_dm = torch.Tensor(np.array(batch_dm))
                batch_data = torch.cat([batch_data, batch_dm], dim=1)
            
            batch_data = batch_data.to(device)

            # doing the inference
            with torch.no_grad():
                for repetition in range(1 if not bayesian_dropout else 25):
                    tiles_pred = model(batch_data).detach().cpu().numpy()
                    index = 0
                    for i,j in batch:
                        inference[
                            :, 
                            i+model_data['chip_border']:i+model_data['chip_size']-model_data['chip_border'], 
                            j+model_data['chip_border']:j+model_data['chip_size']-model_data['chip_border']] += (
                                tiles_pred[index][:,model_data['chip_border']:-model_data['chip_border'],model_data['chip_border']:-model_data['chip_border']] if model_path in ('models/UNet-S1.pt', 'models/UNet-S2.pt')
                                else np.concat([
                                    np.zeros_like(tiles_pred[index][0])[None,model_data['chip_border']:-model_data['chip_border'], model_data['chip_border']:-model_data['chip_border']],
                                    tiles_pred[index][:,model_data['chip_border']:-model_data['chip_border'], model_data['chip_border']:-model_data['chip_border']]
                                ], axis=0))
                        if bayesian_dropout:
                            uncertainty[i+model_data['chip_border']:i+model_data['chip_size']-model_data['chip_border'], j+model_data['chip_border']:j+model_data['chip_size']-model_data['chip_border']] += -1*np.sum(np.log(tiles_pred[index][:,model_data['chip_border']:-model_data['chip_border'],model_data['chip_border']:-model_data['chip_border']])*tiles_pred[index][:,model_data['chip_border']:-model_data['chip_border'],model_data['chip_border']:-model_data['chip_border']], 0)
                        index+=1
            
            batch_count+=1
            if not ui is None:
                ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)
        
        # final prediction
        inference = np.argmax(inference[:, model_data['chip_border']:-model_data['chip_border'], model_data['chip_border']:-model_data['chip_border']], axis=0).astype(np.byte)
        if bayesian_dropout:
            uncertainty = uncertainty[model_data['chip_border']:-model_data['chip_border'], model_data['chip_border']:-model_data['chip_border']]

    # if it is not a model that rely on neural networks
    else:
        if model_path=='models/Otsu_Threshold':
            if not ui is None:
                ui['button_run']["text"] = "Applying Otsu Threshold..."
            
            if input_info['data_type']!='sentinel-1':
                raise IOError('Unsuitable data type for Otsu Thresholding.')

            bayesian_dropout = False
            
            inference, nodata_mask = apply_otsu_threshold(input_info['input_files'][0], input_info.get('sar_is_dB', True))

        elif model_path=='models/WF2_unetv2_bgriswirs':
            if input_info['data_type']!='sentinel-2':
                raise IOError('Unsuitable data type for WF2_unetv2_bgriswirs.')

            bayesian_dropout = False

            inference, nodata_mask = run_wf2_inference(input_info['input_files'][0], device, ui=ui)

    # clean small areas if necessary
    if clean_result:
        print('Post-processing end result...')
        if not ui is None:
            ui['button_run']["text"] = "Post-processing..."
        inference = tile_cleaner(inference, 500, 16, foreground_val=2, background_val=1)
    
    # setting no data correctly
    # inference += 1
    inference[nodata_mask] = 0
    if bayesian_dropout:
        uncertainty[~np.isfinite(uncertainty)] = 0
    
    # saving final result
    with r.Env():
        profile = r.open(resolve_profile_reference_path(input_info)).profile
        profile.update(
            driver='GTiff',
            dtype=r.uint8 if not bayesian_dropout else r.float32,
            count=1 if not bayesian_dropout else 2,
            nodata=0 if not bayesian_dropout else None,
            compress='lzw')

        with r.open(result_path, 'w', **profile) as dst:
            dst.write(inference.astype(r.uint8 if not bayesian_dropout else r.float32), 1)
            if bayesian_dropout:
                dst.write(uncertainty.astype(r.float32), 2)
            dst.descriptions = ['Flood Mask'] if not bayesian_dropout else ['Flood Mask', 'Dropout Uncertainty']
    
    time_end = time.time()
    time_elapsed = time_end-time_start
    print(f'Total time: {time_elapsed/60:.2f} minutes.')

    if not ui is None:
        alert_finished(result_path, time_elapsed/60)
        if show_plot:
            # plotting must happen on the tkinter (main) thread, not this worker thread
            ui['window'].after(0, lambda: show_results_plot(ui['window'], input_info, result_path))
        ui['button_run']['state'] = 'normal'
        ui['button_run']["text"] = "Start Processing"

# TODO: Check if inputs are valid
# TODO: Disable everything in the UI
def show_results_plot(window, input_info, result_path):
    try:
        from src.plotter import plot_results
        plot_results(input_info['input_files'][0], result_path, input_info['data_type'], parent=window)
    except Exception as exc:
        show_error(f'Could not plot the results: {exc}')

def start_processing(model_name, input_info, output_path, post_processing=False, window=None, pb=None, 
    device=None, bt_run=None, bayesian_dropout:bool=False, show_plot:bool=False):

    # if input_image_path and output_path:
    os.makedirs('images', exist_ok=True)
    if not window is None:
        bt_run["state"] = "disabled"
        bt_run["text"] = "Loading data..."

    print('Device:', device)

    Thread(
        target=inference,
        args=(
            f'models/{model_name}',
            input_info, 
            output_path, 
            post_processing, 
            None if window is None else {'window': window, 'progress_bar': pb, 'button_run': bt_run},
            device,
            bayesian_dropout,
            show_plot
        )
    ).start()

def build_tile_batches(height, width, chip_size, chip_border, nodata_mask, batch_size=8):
    '''
    Splits a (height, width) raster into overlapping chip_size x chip_size tiles
    (so that chip_border pixels of context are available on every side once the
    border is cropped back off after inference), skipping tiles that are
    entirely nodata, and groups their top-left (i,j) coordinates into batches.
    '''
    starting_coordinates_batches = []
    batch = []
    for i in range(0, height, int(chip_size-2*chip_border)):
        if i+chip_size>height:
            i = height-chip_size

        for j in range(0, width, int(chip_size-2*chip_border)):
            if j+chip_size>width:
                j = width-chip_size

            if np.sum(~nodata_mask[i:i+chip_size, j:j+chip_size])!=0:
                batch.append((i,j))
                if len(batch)>=batch_size:
                    starting_coordinates_batches.append(batch)
                    batch = []

    if batch:
        starting_coordinates_batches.append(batch)

    return starting_coordinates_batches

def load_input_data(input_files:list, data_type:str, sar_data_is_in_dB:bool=False, processing_type:str='normalize', path_input_image_aux:str=None):
    if data_type == 'sentinel-1':
        data, nodata_mask = load_and_normalize_sentinel_1_data(input_files[0], sar_data_is_in_dB, processing_type)
        
    if data_type == 'sentinel-2':
        data, nodata_mask = load_and_normalize_sentinel_2_data(input_files[0])

    if data_type == 'planetscope':
        data, nodata_mask = load_and_normalize_planetscope_data(input_files[0], input_files[1])

    return data, nodata_mask

SENTINEL_2_BAND_IDS = {
    'blue': 'B02',
    'green': 'B03',
    'red': 'B04',
    'nir': 'B08',
    'swir16': 'B11',
    'swir22': 'B12'
}

# Scene Classification Layer codes treated as unusable for flood mapping: cloud
# shadows (3), cloud medium probability (8), cloud high probability (9).
SENTINEL_2_CLOUD_SCL_CLASSES = (3, 8, 9)

def load_and_normalize_sentinel_2_data(input_path):
    if os.path.isdir(input_path):
        return load_and_normalize_sentinel_2_safe_folder(input_path)

    dataset_s2 = r.open(input_path)
    blue = dataset_s2.read(1)
    green = dataset_s2.read(2)
    red = dataset_s2.read(3)
    nir = dataset_s2.read(4)
    swir16 = dataset_s2.read(5)
    swir22 = dataset_s2.read(6)

    nodata_mask = blue==0

    # an optional 7th band (SCL) enables cloud/cloud-shadow masking; older
    # 6-band files (or manually prepared ones) simply won't have it
    if dataset_s2.count >= 7:
        scl = dataset_s2.read(7)
        nodata_mask = nodata_mask | np.isin(scl, SENTINEL_2_CLOUD_SCL_CLASSES)

    data = np.stack([blue, green, red, nir, swir16, swir22]).astype(np.float32)

    return data, nodata_mask

def resolve_profile_reference_path(input_info):
    '''Picks the file used as the georeferencing template for the output raster.'''
    path = input_info['input_files'][0]
    if input_info['data_type']=='sentinel-2' and os.path.isdir(path):
        return find_sentinel_2_band_path(path, SENTINEL_2_BAND_IDS['blue'])
    return path

def find_sentinel_2_band_path(safe_folder, band_id):
    '''Finds a band's .jp2 file inside a Sentinel-2 .SAFE product, picking the finest resolution available if more than one copy exists (e.g. B02 under both R10m and R20m).'''
    matches = glob.glob(os.path.join(safe_folder, '**', f'*_{band_id}_*.jp2'), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(safe_folder, '**', f'*_{band_id}.jp2'), recursive=True)
    if not matches:
        raise IOError(f'Could not find band {band_id} inside "{safe_folder}".')

    def resolution_of(path):
        match = re.search(r'(\d+)m', os.path.basename(path))
        return int(match.group(1)) if match else 0

    matches.sort(key=resolution_of)
    return matches[0]

def get_sentinel_2_safe_folder_boa_offset(safe_folder):
    '''
    Since processing baseline 04.00 (January 2022), Sentinel-2 L2A products add a
    constant offset (usually -1000) to every band's digital numbers, to allow
    representing negative reflectances; products before that baseline don't.
    Reads the actual value from the product's own metadata (all bands share the
    same value in practice), returning 0 if the tag is absent (older baseline).
    '''
    mtd_path = os.path.join(safe_folder, 'MTD_MSIL2A.xml')
    if not os.path.isfile(mtd_path):
        return 0
    for elem in ET.parse(mtd_path).getroot().iter('BOA_ADD_OFFSET'):
        return int(elem.text)
    return 0

def load_and_normalize_sentinel_2_safe_folder(safe_folder):
    band_order = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']
    band_paths = [find_sentinel_2_band_path(safe_folder, SENTINEL_2_BAND_IDS[name]) for name in band_order]

    with r.open(band_paths[0]) as ref:
        ref_shape = (ref.height, ref.width)

    bands = []
    for path in band_paths:
        with r.open(path) as dataset:
            bands.append(dataset.read(1, out_shape=ref_shape, resampling=Resampling.bilinear))

    data = np.stack(bands).astype(np.float32)
    nodata_mask = data[0]==0

    # remove the baseline-04.00+ offset so digital numbers are on the same scale
    # regardless of which processing baseline produced the product
    boa_offset = get_sentinel_2_safe_folder_boa_offset(safe_folder)
    if boa_offset:
        data[:, ~nodata_mask] = np.clip(data[:, ~nodata_mask] + boa_offset, a_min=0, a_max=None)

    # SCL is categorical, so it must be resampled with nearest neighbor, not bilinear
    scl_path = find_sentinel_2_band_path(safe_folder, 'SCL')
    with r.open(scl_path) as dataset:
        scl = dataset.read(1, out_shape=ref_shape, resampling=Resampling.nearest)
    nodata_mask = nodata_mask | np.isin(scl, SENTINEL_2_CLOUD_SCL_CLASSES)

    return data, nodata_mask

def load_and_normalize_planetscope_data(input_path, input_path_aux):
    input_dataset = r.open(input_path)
    data = np.stack([
        input_dataset.read(int(np.argmax(np.asarray(input_dataset.descriptions)=='blue')+1)),
        input_dataset.read(int(np.argmax(np.asarray(input_dataset.descriptions)=='green')+1)),
        input_dataset.read(int(np.argmax(np.asarray(input_dataset.descriptions)=='red')+1)),
        input_dataset.read(int(np.argmax(np.asarray(input_dataset.descriptions)=='nir')+1))
    ], axis=0).astype(np.float32)

    nodata_mask = r.open(input_path_aux).read(1)==0

    for i in range(4):
        data[i] = (data[i]-np.mean(data[i, ~nodata_mask]))/np.std(data[i, ~nodata_mask])
    
    return data, nodata_mask

def load_and_normalize_sentinel_1_data(path_input_image, data_is_in_dB:bool=False, processing_type:str='normalize'):
    dataset_sar = r.open(path_input_image)
    vv = dataset_sar.read(2)
    vh = dataset_sar.read(1)

    nodata_mask = vv==0

    data = np.stack([vv, vh])

    if not data_is_in_dB:
        data = 10*np.log10(data)
    
    datascaler = DataScaler()
    if processing_type=='normalize':
        data = datascaler.normalize_data('s1_during_flood', data)
    if processing_type=='scale':
        data = datascaler.scale_data('s1_during_flood', data)
    
    return data, nodata_mask

def apply_otsu_threshold(input_path, data_is_in_dB):
    from skimage.filters import threshold_otsu
    vh = r.open(input_path).read(1)
    nodata_mask = vh==0
    if not data_is_in_dB:
        vh = 10*np.log10(vh)
    otsu_value = threshold_otsu(vh[~nodata_mask])

    # same labels as the neural-net models: 1 = dry, 2 = flood (0 is reserved for nodata)
    otsued_data = (vh<=otsu_value).astype(np.uint8) + 1
    
    return otsued_data, nodata_mask

# ml4floods' WF2_unetv2_bgriswirs: https://huggingface.co/isp-uv-es/ml4floods
WF2_REPO_ID = 'isp-uv-es/ml4floods'
WF2_FILENAME = 'models/WF2_unetv2_bgriswirs/model.pt'
WF2_CHIP_SIZE = 256
WF2_CHIP_BORDER = 94
WF2_BATCH_SIZE = 8

# normalization statistics (mean, std) for blue, green, red, nir, swir16, swir22,
# and water/cloud decision thresholds, as used by ml4floods for this model
WF2_S2_MEAN = [3758.07467509, 3238.08247208, 3418.90147615, 3981.96268494, 2391.66101119, 1790.32497137]
WF2_S2_STD = [2794.09579088, 2549.4940614, 2811.78109878, 2500.47885249, 1500.02533014, 1241.9817628]
WF2_TH_WATER = 0.5
WF2_TH_CLOUD = 0.5
WF2_TH_BRIGHTNESS = 3500

def load_wf2_model(device):
    from huggingface_hub import hf_hub_download
    from src.models import UNetML4FloodsWF2

    weights_path = hf_hub_download(repo_id=WF2_REPO_ID, filename=WF2_FILENAME)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    state_dict = {k.removeprefix('network.'): v for k, v in state_dict.items()}

    model = UNetML4FloodsWF2(n_channels=6, n_class=2).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_wf2_chips(model, batch_chips, device):
    '''
    batch_chips: (batch, 6, H, W) raw blue/green/red/nir/swir16/swir22 digital numbers.
    Returns a (batch, H, W) uint8 array with labels {0: nodata, 1: land, 2: water, 3: cloud}.
    '''
    mean = torch.tensor(WF2_S2_MEAN, device=device).view(1,6,1,1)
    std = torch.tensor(WF2_S2_STD, device=device).view(1,6,1,1)

    x = torch.nan_to_num(batch_chips, nan=0.0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model((x-mean)/(std+1e-6)))
    cloud = probs[:,0] > WF2_TH_CLOUD
    water = probs[:,1] > WF2_TH_WATER
    brightness = torch.sqrt((x[:,[2,1,0]]**2).sum(dim=1))  # red, green, blue

    mask = torch.ones(water.shape, dtype=torch.uint8, device=device)
    mask[water] = 2
    mask[cloud & (brightness>WF2_TH_BRIGHTNESS)] = 3
    mask[(x==0).all(dim=1)] = 0
    return mask.cpu().numpy()

def run_wf2_inference(input_path, device, ui=None):
    '''
    Runs ml4floods' WF2_unetv2_bgriswirs model on a Sentinel-2 input (single 6+
    band file or a .SAFE folder), tiling the same way as this tool's other
    neural-net models. Returns (inference, nodata_mask) with inference labels
    {0: nodata, 1: land, 2: water, 3: cloud}.
    '''
    if not ui is None:
        ui['button_run']["text"] = "Downloading model weights..."
    model = load_wf2_model(device)

    dataset, nodata_mask = load_and_normalize_sentinel_2_data(input_path)
    dataset = np.stack([np.pad(dataset[i], pad_width=WF2_CHIP_BORDER, mode='reflect') for i in range(len(dataset))])

    if not ui is None:
        ui['button_run']["text"] = "Getting tiles..."
    starting_coordinates_batches = build_tile_batches(dataset.shape[1], dataset.shape[2], WF2_CHIP_SIZE, WF2_CHIP_BORDER, nodata_mask, batch_size=WF2_BATCH_SIZE)

    if not ui is None:
        ui['button_run']["text"] = "Doing the inference..."
    inference = np.zeros([dataset.shape[1], dataset.shape[2]], dtype=np.uint8)

    batch_count = 0
    for batch in tqdm(starting_coordinates_batches, desc='Inference:'):
        batch_chips = np.stack([dataset[:, i:i+WF2_CHIP_SIZE, j:j+WF2_CHIP_SIZE] for i,j in batch])
        tiles_pred = predict_wf2_chips(model, torch.Tensor(batch_chips), device)

        for index, (i,j) in enumerate(batch):
            inference[i+WF2_CHIP_BORDER:i+WF2_CHIP_SIZE-WF2_CHIP_BORDER, j+WF2_CHIP_BORDER:j+WF2_CHIP_SIZE-WF2_CHIP_BORDER] = (
                tiles_pred[index][WF2_CHIP_BORDER:-WF2_CHIP_BORDER, WF2_CHIP_BORDER:-WF2_CHIP_BORDER])

        batch_count += 1
        if not ui is None:
            ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)

    inference = inference[WF2_CHIP_BORDER:-WF2_CHIP_BORDER, WF2_CHIP_BORDER:-WF2_CHIP_BORDER]
    return inference, nodata_mask

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

def alert_finished(output_path, elapsed_time_minutes):
    tk.messagebox.showinfo(title='Processing Finished', message=f'Process finished!\nLocation: {output_path}\nTime: {elapsed_time_minutes:.2f} minutes.')