import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from threading import Thread
from skimage.morphology import area_opening, area_closing
import warnings
import rasterio as r
import time
import tkinter as tk

from src.models import UNetSemanticSegmentation
from src.helpers import DataScaler, get_slope, get_points_and_distance_map, tile_cleaner

def inference(model_path, path_input_image, result_path, clean_result=False, ui=None, device=torch.device('cpu'), 
    sar_is_dB:bool=False, bayesian_dropout:bool=False, path_input_image_aux:str=None):
    time_start = time.time()
    # check if it's running the ui or terminal version
    if not ui is None:
        ui['progress_bar']['value'] = 0

    # if it is a model that rely on neural networks...
    if not model_path in ['models/Otsu_Threshold']:
        # innitially load the model weights
        model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        # load model for inference
        match model_path:
            case 'models/UNet-S1.pt':
                model = UNetSemanticSegmentation(in_channels=2, out_channels=3, base=32).to(device)
            case 'models/DistanceMap.pt':
                model = UNetSemanticSegmentation(in_channels=4, out_channels=2, base=32).to(device)
            case 'models/UNet-PlanetScope.pt':
                model = UNetSemanticSegmentation(in_channels=4, out_channels=2, base=32).to(device)
            case _:
                raise ValueError(f'{model_path} is not a valid model name.')
        
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()

        # activating dropouts if Bayesian dropout option was given
        if bayesian_dropout:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        # load data scaler
        data_scaler = DataScaler()

        # load input data
        dataset, nodata_mask = load_input_data(path_input_image, model_data['data_type'], sar_is_dB, model_data['data_preprocessing'], path_input_image_aux)

        # getting terrain
        if model_data['use_terrain']:
            if not ui is None:
                ui['button_run']["text"] = "Downloading DEM..."
            slope = get_slope(path_input_image)

        # define placeholder for final result
        inference = np.zeros([3, dataset.shape[1], dataset.shape[2]])
        if bayesian_dropout:
            uncertainty = np.zeros([dataset.shape[1], dataset.shape[2]], dtype=np.float32)
        
        # define tiles for processing and batches as well
        if not ui is None:
            ui['button_run']["text"] = "Getting tiles..."
        
        starting_coordinates_batches = []
        batch_size = 8
        batch = []
        for i in range(0, dataset.shape[1], int(model_data['chip_size']/2)):
            if i+model_data['chip_size']>dataset.shape[1]:
                i = dataset.shape[1]-model_data['chip_size']
            
            for j in range(0, dataset.shape[2], int(model_data['chip_size']/2)):
                if j+model_data['chip_size']>dataset.shape[2]:
                    j = dataset.shape[2]-model_data['chip_size']

                if np.sum(~nodata_mask[i:i+model_data['chip_size'], j:j+model_data['chip_size']])!=0:
                    batch.append((i,j))
                    if len(batch)>=batch_size:
                        starting_coordinates_batches.append(batch)
                        batch = []
        
        if batch:
            starting_coordinates_batches.append(batch)

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

                # doing terrain if necessary
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
                for repetition in range(1 if not bayesian_dropout else 10):
                    tiles_pred = model(batch_data).detach().cpu().numpy()
                    index = 0
                    for i,j in batch:
                        inference[:, i:i+model_data['chip_size'], j:j+model_data['chip_size']] += tiles_pred[index] if model_path=='models/UNet-S1.pt' else np.concat([np.zeros_like(tiles_pred[index][0][None,:,:]), tiles_pred[index]], axis=0)
                        if bayesian_dropout:
                            uncertainty[i:i+model_data['chip_size'], j:j+model_data['chip_size']] += -1*np.sum(np.log(tiles_pred[index])*tiles_pred[index], 0)
                        index+=1
            
            batch_count+=1
            if not ui is None:
                ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)
        
        # final prediction
        inference = (inference[-1]>=1).astype(np.byte) if not bayesian_dropout else np.argmax(inference, axis=0).astype(np.byte)
    
    # if it is not a model that rely on neural networks
    else:
        if model_path=='models/Otsu_Threshold':
            if not ui is None:
                ui['button_run']["text"] = "Applying Otsu Threshold..."
            
            bayesian_dropout = False
            
            inference, nodata_mask = apply_otsu_threshold(path_input_image, sar_data_is_in_dB)

    # clean small areas if necessary
    if clean_result:
        print('Post-processing end result...')
        if not ui is None:
            ui['button_run']["text"] = "Post-processing..."
        inference = tile_cleaner(inference, 500, 16, 2)
    
    # setting no data correctly
    inference += 1
    inference[nodata_mask] = 0
    if bayesian_dropout:
        uncertainty[~np.isfinite(uncertainty)] = 0
    
    # saving final result
    with r.Env():
        profile = r.open(path_input_image).profile
        profile.update(
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
        ui['button_run']['state'] = 'normal'
        ui['button_run']["text"] = "Start Processing"

# TODO: Check if inputs are valid
# TODO: Disable everything in the UI
def start_processing(model_name, input_image_path, output_path, post_processing=False, window=None, pb=None, 
    device=None, bt_run=None, sar_is_dB:bool=False, bayesian_dropout:bool=False, input_image_path_aux:str=None):
    if not input_image_path:
        show_error('Please insert an input path!')
        return
    if not output_path:
        show_error('Please insert an output path!')
        return

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
            input_image_path, 
            output_path, 
            post_processing, 
            None if window is None else {'window': window, 'progress_bar': pb, 'button_run': bt_run},
            device,
            sar_is_dB,
            bayesian_dropout,
            input_image_path_aux
        )
    ).start()

def load_input_data(path_input_image, data_type, sar_data_is_in_dB:bool=False, processing_type:str='normalize', path_input_image_aux:str=None):
    if data_type == 's1_during_flood':
        data, nodata_mask = load_and_normalize_sentinel_1_data(path_input_image, sar_data_is_in_dB, processing_type)
        
    if data_type == 'planetscope':
        data, nodata_mask = load_and_normalize_planetscope_data(path_input_image, path_input_image_aux)

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

    otsued_data = (vh<=otsu_value).astype(np.uint8)
    
    return otsued_data, nodata_mask

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

def alert_finished(output_path, elapsed_time_minutes):
    tk.messagebox.showinfo(title='Processing Finished', message=f'Process finished!\nLocation: {output_path}\nTime: {elapsed_time_minutes:.2f} minutes.')