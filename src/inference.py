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

def inference(model_path, path_input_image, result_path, clean_result=False, ui=None, device=torch.device('cpu'), sar_is_dB:bool=False, bayesian_dropout:bool=False):
    time_start = time.time()
    # check if it's running the ui or terminal version
    if not ui is None:
        ui['progress_bar']['value'] = 0
    
    # innitially load the model weights
    data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    # load model for inference
    match model_path:
        case 'models/UNet.pt':
            model = UNetSemanticSegmentation(in_channels=2, out_channels=3, base=32).to(device)
        case 'models/DistanceMap.pt':
            model = UNetSemanticSegmentation(in_channels=4, out_channels=2, base=32).to(device)
        case _:
            raise ValueError(f'{model_path} is not a valid model name.')
    
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    # activating dropouts if Bayesian dropout option was given
    if bayesian_dropout:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    # load data scaler
    data_scaler = DataScaler()

    # load input data
    dataset_sar = r.open(path_input_image)
    vv = dataset_sar.read(2)
    vh = dataset_sar.read(1)

    # getting terrain
    if data['use_terrain']:
        if not ui is None:
            ui['button_run']["text"] = "Downloading DEM..."
        slope = get_slope(path_input_image)

    # define placeholder for final result
    inference = np.zeros([3, dataset_sar.height, dataset_sar.width])
    if bayesian_dropout:
        uncertainty = np.zeros([dataset_sar.height, dataset_sar.width], dtype=np.float32)
    
    # define tiles for processing and batches as well
    if not ui is None:
        ui['button_run']["text"] = "Getting tiles..."
    starting_coordinates_batches = []
    batch_size = 8
    batch = []
    for i in range(0, dataset_sar.height, data['chip_size']):
        if i+data['chip_size']>dataset_sar.height:
            i = dataset_sar.height-data['chip_size']
        
        for j in range(0, dataset_sar.width, data['chip_size']):
            if j+data['chip_size']>dataset_sar.width:
                j = dataset_sar.width-data['chip_size']

            if np.sum(vv[i:i+data['chip_size'], j:j+data['chip_size']])!=0:
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
        batch_s1 = []
        batch_slope = []
        batch_dm = []
        for i,j in batch:
            # loads sar data
            s1 = np.stack([
                vv[i:i+data['chip_size'], j:j+data['chip_size']], 
                vh[i:i+data['chip_size'], j:j+data['chip_size']]
            ])

            if not sar_is_dB: # convert linear data to dB
                s1 = 10*np.log10(s1)
            
            match data['data_preprocessing']:
                case 'normalize':
                    s1 = data_scaler.normalize_data('s1_during_flood', s1)
                case 'scale':
                    s1 = data_scaler.scale_data('s1_during_flood', s1)
                case _:
                    raise ValueError(f'Model metadata data_preprocessing is {data["data_preprocessing"]} (not supported).')
            
            batch_s1.append(s1)

            # doing terrain if necessary
            if data['use_terrain']:
                t = np.stack([np.zeros([data['chip_size'],data['chip_size']]), slope[i:i+data['chip_size'], j:j+data['chip_size']]])
                match data['data_preprocessing']:
                    case 'normalize':
                        t = data_scaler.normalize_data('terrain', t)
                    case 'scale':
                        t = data_scaler.scale_data('terrain', t)
                    case _:
                        raise ValueError(f'Model metadata data_preprocessing is {data["data_preprocessing"]} (not supported).')
                batch_slope.append(t[1])
            
            # doing distance map if necessary
            if data['use_distance_map']:
                coords_f, coords_n, dmap = get_points_and_distance_map(s1, t, max_points_per_class_map=100)
                batch_dm.append(dmap)
        
        batch_s1 = np.array(batch_s1)
        batch_s1[np.isnan(batch_s1)] = 0

        batch_data = torch.Tensor(np.asarray(batch_s1, dtype=np.float32))

        if data['use_terrain']:
            batch_slope = torch.Tensor(np.array(batch_slope)[:,None,:,:])
            batch_data = torch.cat([batch_data, batch_slope], dim=1)
        
        if data['use_distance_map']:
            batch_dm = torch.Tensor(np.array(batch_dm))
            batch_data = torch.cat([batch_data, batch_dm], dim=1)
        
        batch_data = batch_data.to(device)

        # doing the inference
        with torch.no_grad():
            for repetition in range(1 if not bayesian_dropout else 10):
                tiles_pred = model(batch_data).detach().cpu().numpy()
                index = 0
                for i,j in batch:
                    inference[:, i:i+data['chip_size'], j:j+data['chip_size']] += tiles_pred[index] if model_path=='models/UNet.pt' else np.concat([np.zeros_like(tiles_pred[index][0][None,:,:]), tiles_pred[index]], axis=0)
                    if bayesian_dropout:
                        uncertainty[i:i+data['chip_size'], j:j+data['chip_size']] += -1*np.sum(np.log(tiles_pred[index])*tiles_pred[index], 0)
                    index+=1
        
        batch_count+=1
        if not ui is None:
            ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)
    
    # final correction
    inference[:,vv==0] = 0
    if bayesian_dropout:
        uncertainty[~np.isfinite(uncertainty)] = 0

    # clean if necessary
    if clean_result:
        print('Post-processing end result...')
        if not ui is None:
            ui['button_run']["text"] = "Post-processing..."
        inference = tile_cleaner(inference, 500, 16, 2)
    
    # saving final result
    with r.Env():
        profile = dataset_sar.profile
        profile.update(
            dtype=r.uint8 if not bayesian_dropout else r.float32,
            count=1 if not bayesian_dropout else 2,
            nodata=0 if not bayesian_dropout else None,
            compress='lzw')

        with r.open(result_path, 'w', **profile) as dst:
            dst.write(np.argmax(inference, axis=0).astype(r.uint8 if not bayesian_dropout else r.float32), 1)
            if bayesian_dropout:
                dst.write(uncertainty.astype(r.float32), 2)
            dst.descriptions = ['Flood Mask'] if not bayesian_dropout else ['Flood Mask', 'Dropout Uncertainty']
    
    time_end = time.time()
    time_elapsed = time_end-time_start
    print(f'Total time: {time_elapsed/60:.2f} minutes.')

    if not ui is None:
        alert_finished(result_path, time_elapsed)
        ui['button_run']['state'] = 'normal'
        ui['button_run']["text"] = "Start Processing"

# TODO: Check if inputs are valid
# TODO: Disable everything in the UI
def start_processing(model_name, input_image_path, output_path, post_processing=False, window=None, pb=None, device=None, bt_run=None, sar_is_dB:bool=False, bayesian_dropout:bool=False):
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
            bayesian_dropout
        )
    ).start()

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

def alert_finished(output_path, elapsed_time_minutes):
    tk.messagebox.showinfo(title='Processing Finished', message=f'Process finished!\nLocation: {output_path}\nTime: {elapsed_time_minutes:.2f} minutes.')