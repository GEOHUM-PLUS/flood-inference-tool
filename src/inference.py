import os
import pickle
import argparse
import rioxarray
from rioxarray.merge import merge_arrays
import torch
import torch.nn as nn
import rasterio as r
from rasterio.crs import CRS
from rasterio import warp
import numpy as np
from tqdm import tqdm
from threading import Thread
import pystac_client
import planetary_computer
import geopandas
from shapely import Polygon
import subprocess
from skimage.morphology import area_opening, area_closing
import warnings

from src.helpers import get_points_and_distance_map
from src.models import SimpleUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def inference(model_path, path_input_image, result_path, clean_result=False, ui=None):
    # check if it's running the ui or terminal version
    if not ui is None:
        ui['progress_bar']['value'] = 0
    
    # innitially load the model weights
    data = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=True)

    # load model for inference
    match data['model_name']:
        case 'UNet':
            model = SimpleUNet().to(DEVICE)
        case 'SimpleUNetEmb':
            model = SimpleUNetEmb(4,2,data['chip_size']).to(DEVICE)
        case _:
            raise ValueError(f'{data["model_name"]} is not a valid model name.')
    model.load_state_dict(data['model_state_dict'])
    model.eval()

    # load data scaler
    data_scaler = DataScaler()

    # load input data
    dataset_sar = r.open(path_input_image)
    vv = dataset_sar.read(2)
    vh = dataset_sar.read(1)
    if not ui is None:
        ui['button_run']["text"] = "Downloading DEM..."

    # get slope
    slope = get_slope(path_input_image)

    # define placeholder for final result
    inference = np.zeros([dataset_sar.height, dataset_sar.width])+255
    
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

    # get the embeddings, if necessary
    if data['use_emb']:
        if not ui is None:
            ui['button_run']["text"] = "Getting SatCLIP embeddings..."
        embeddings = get_embeddings(
            starting_coordinates=starting_coordinates_batches, 
            ref_dataset=dataset_sar, 
            chip_size=data['chip_size']
        )

    # Do the inference!
    print('Doing inference... (finally!)')
    if not ui is None:
        ui['button_run']["text"] = "Doing the inference..."
    batch_count = 0
    
    print('TODO: Implement option to use linear or dB units')
    for batch in tqdm(starting_coordinates_batches, ncols=70):
        batch_s1 = []
        batch_t = []
        batch_d = []
        for i,j in batch:
            if False: #data in linear units
                s1 = data_scaler.scale_data('s1_during_flood', np.stack([10*np.log10(vv[i:i+data['chip_size'], j:j+data['chip_size']]), 10*np.log10(vh[i:i+data['chip_size'], j:j+data['chip_size']])]))
            else: # data in dB
                s1 = data_scaler.scale_data('s1_during_flood', np.stack([vv[i:i+data['chip_size'], j:j+data['chip_size']], vh[i:i+data['chip_size'], j:j+data['chip_size']]]))
            t = data_scaler.scale_data('terrain', np.stack([np.zeros([data['chip_size'],data['chip_size']]), slope[i:i+data['chip_size'], j:j+data['chip_size']]]))
            coords_f, coords_n, dmap = get_points_and_distance_map(s1, t, max_points_per_class_map=100)
            batch_d.append(dmap)
            batch_s1.append(s1)
            batch_t.append(t[1])
        
        batch_s1 = np.array(batch_s1)
        batch_s1[np.isnan(batch_s1)] = 0
        batch_t = np.array(batch_t)[:,None,:,:]
        batch_d = np.array(batch_d)
        
        batch_data = torch.Tensor(np.concatenate([batch_s1, batch_t, batch_d], axis=1)).to(DEVICE)

        with torch.no_grad():
            if data['use_emb']:
                tiles_pred = torch.argmax(model(batch_data, torch.Tensor(embeddings[batch_count]).to(torch.float32).to(DEVICE)).detach().cpu(), axis=1)
            else:
                tiles_pred = torch.argmax(model(batch_data).detach().cpu(), axis=1)
            index = 0
            for i,j in batch:
                inference[i:i+data['chip_size'], j:j+data['chip_size']] = tiles_pred[index].numpy()
                index+=1
        
        batch_count+=1
        if not ui is None:
            ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)
    
    # final correction
    inference[vv==0] = 2

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
            dtype=r.uint8,
            count=1,
            nodata=2,
            compress='lzw')

        with r.open(result_path, 'w', **profile) as dst:
            dst.write(inference.astype(r.uint8), 1)
    
    if not ui is None:
        ui['button_run']['state'] = 'normal'
        ui['button_run']["text"] = "Start Processing"

# TODO: Check if inputs are valid
# TODO: Disable everything in the UI
def start_processing(model_name, input_image_path, output_path, post_processing=False, window=None, pb=None, device=None, bt_run=None):
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
        global DEVICE
        DEVICE = device

    print('Device:', DEVICE)

    Thread(
        target=inference,
        args=(
            f'models/{model_name}',
            input_image_path, 
            output_path, 
            post_processing, 
            None if window is None else {'window': window, 'progress_bar': pb, 'button_run': bt_run}
        )
    ).start()