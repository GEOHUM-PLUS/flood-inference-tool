import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'satclip/satclip'))

from huggingface_hub import hf_hub_download
from load import get_satclip

import rioxarray
from rioxarray.merge import merge_arrays
import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import pickle
import argparse
import os
import rasterio as r
from rasterio.crs import CRS
from rasterio import warp
import numpy as np
from distancemap import distance_map
from tqdm import tqdm
from threading import Thread
import pystac_client
import planetary_computer
import geopandas
from shapely import Polygon
import subprocess
from skimage.morphology import area_opening, area_closing
import warnings
from PIL import Image, ImageTk

from helpers import get_points_and_distance_map
from models import SimpleUNet, SimpleUNetEmb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DataScaler:
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'percentile_limits.pickle'), 'rb') as f:
            self.STRETCH_LIMITS = pickle.load(f)
            self.percentile_bttm = 5
            self.percentile_top = 95
    
    def scale_data(self, data_type, data):
        # follows scales only if needed
        if data_type in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'terrain']:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_bttm))])/(self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_top))]-self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_bttm))])
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i]['0'])/(self.STRETCH_LIMITS[data_type][i]['100']-self.STRETCH_LIMITS[data_type][i]['0'])
        
        # clipping to 0 1
        data = np.clip(data, a_min=0, a_max=1)

        return data

def get_slope(path_reference):
    print('Getting slope...')
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    ref = r.open(path_reference)

    bounds = Polygon([
        [ref.bounds.left,  ref.bounds.top],
        [ref.bounds.right, ref.bounds.top],
        [ref.bounds.right, ref.bounds.bottom],
        [ref.bounds.left,  ref.bounds.bottom],
        [ref.bounds.left,  ref.bounds.top],
    ])

    gdf = geopandas.GeoDataFrame(geometry=[bounds], crs=ref.crs)
    gdf_p = gdf.to_crs(epsg=4326)

    bottom = float(gdf_p.bounds['miny'].values[0])
    top = float(gdf_p.bounds['maxy'].values[0])
    left = float(gdf_p.bounds['minx'].values[0])
    right = float(gdf_p.bounds['maxx'].values[0])

    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [left,  top],
                [right, top],
                [right, bottom],
                [left,  bottom],
                [left,  top],
            ]
        ],
    }

    search = catalog.search(
        collections=["cop-dem-glo-30"], intersects=aoi
    )
    items = search.item_collection()

    arrays = []
    for item in items:
        arrays.append(rioxarray.open_rasterio(item.assets['data'].href))
    
    merged = merge_arrays(arrays)

    merged.rio.to_raster('images/COP-DEM-GLO-30.tif', driver='GTiff', compress='LZW')

    subprocess.call(f'gdaldem slope images/COP-DEM-GLO-30.tif images/slope.tif -alg ZevenbergenThorne -s 111120', shell=True)

    slope = rioxarray.open_rasterio('images/slope.tif')
    ref = rioxarray.open_rasterio(path_reference)

    matched = slope.rio.reproject_match(ref)

    return matched.to_numpy()[0]

# TODO: Paralellize this
def tile_cleaner(data, tile_size=1000, min_feature_size_px=16, nodata_val=255):
    result = np.zeros(data.shape, dtype=np.uint8)
    data2 = data.copy()
    data2[data2==nodata_val] = 0

    for i in tqdm(range(0, data.shape[0], tile_size-min_feature_size_px), ncols=70):
        if i+tile_size>result.shape[0]:
            i = result.shape[0]-tile_size
        for j in range(0, data.shape[1], tile_size-min_feature_size_px):
            if j+tile_size > result.shape[1]:
                j = result.shape[1]-tile_size
            
            if np.sum(data2[i:i+tile_size, j:j+tile_size]==0)!=tile_size*tile_size:
                result[i:i+tile_size, j:j+tile_size] = (
                    result[i:i+tile_size, j:j+tile_size]+
                    area_closing(
                        area_opening(
                            data2[i:i+tile_size, j:j+tile_size], min_feature_size_px
                        ), min_feature_size_px
                    )
                )

            if j == result.shape[1]-tile_size:
                break
        
        if i == result.shape[0]-tile_size:
            break
    result = np.asarray(result>0, dtype=np.uint8)
    result[data==nodata_val] = nodata_val
    return result

def get_embeddings(starting_coordinates, ref_dataset, chip_size):
    print('Getting embeddings...')
    # converting the coordinates from image to lat long
    batch_coordinates_latlong = []
    for batch in starting_coordinates:
        batch = np.array(batch)+int(chip_size/2)
        coords_raster = np.asarray(r.transform.xy(ref_dataset.transform, rows=batch[:,1], cols=batch[:,0])).T
        coords_latlong = np.asarray(warp.transform(src_crs=ref_dataset.crs, dst_crs=CRS.from_epsg(4326), xs=coords_raster[:,0], ys=coords_raster[:,1])).T
        batch_coordinates_latlong.append(coords_latlong)

    # loading SatCLIP
    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
        device=DEVICE,
    )  # Only loads location encoder by default

    # creating embeddings
    embs = []
    for batch in tqdm(batch_coordinates_latlong, ncols=70):
        embs.append(model(torch.Tensor(batch).to(torch.float64).to(DEVICE)).detach().cpu().numpy())
    
    return embs

def inference(model_path, path_input_image, result_path, clean_result=False, ui=None):

    # check if it's running the ui or terminal version
    if not ui is None:
        ui['progress_bar']['value'] = 0
    
    # innitially load the model weights
    data = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=True)

    # load model for inference
    match data['model_name']:
        case 'SimpleUNet':
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
    for batch in tqdm(starting_coordinates_batches, ncols=70):
        batch_s1 = []
        batch_t = []
        batch_d = []
        for i,j in batch:
            s1 = data_scaler.scale_data('s1_during_flood', np.stack([10*np.log10(vv[i:i+data['chip_size'], j:j+data['chip_size']]), 10*np.log10(vh[i:i+data['chip_size'], j:j+data['chip_size']])]))
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

def get_file_path(entry):
    file = filedialog.askopenfilename(filetypes=[('TIF', '*.tif')])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)
        return

def create_file_path(entry):
    file = filedialog.asksaveasfilename(filetypes=[('TIF', '*.tif')])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)
        return

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

def build_ui():
    window = tk.Tk()
    window.title('GEOHUM Flood Mapper')

    img = Image.open('figures/gEOhum_Logo_NEWCD-Web.png')
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack(pady=(10, 0))

    title = tk.Label(text='Flood Mapper', master=window, font="Arial 25 bold")
    title.pack()

    # input
    frame_input_1 = tk.Frame(window)
    frame_input_2 = tk.Frame(window)
    input_label = tk.Label(text='Input file:', master=frame_input_1).pack(side=tk.LEFT)
    input_path = tk.StringVar(window)
    w_input_path = tk.Entry(master=frame_input_2, width=50, textvariable=input_path)
    w_input_path.pack(side=tk.LEFT)

    w_input_path_button = tk.Button(master=frame_input_2, text='...', command=lambda:get_file_path(w_input_path)).pack(side=tk.LEFT)

    frame_input_1.pack(fill=tk.X, pady=(10, 0))
    frame_input_2.pack(fill=tk.X)

    # output
    frame_1 = tk.Frame(window)
    frame_2 = tk.Frame(window)
    output_label = tk.Label(text='Output file:', master=frame_1).pack(side=tk.LEFT)
    output_path = tk.StringVar(window)
    w_output_path = tk.Entry(master=frame_2, width=50, textvariable=output_path)
    w_output_path.pack(side=tk.LEFT)
    w_output_path_button = tk.Button(master=frame_2, text='...', command=lambda:create_file_path(w_output_path)).pack(side=tk.LEFT)

    frame_1.pack(fill=tk.X)
    frame_2.pack(fill=tk.X)

    # device options
    frame = tk.Frame(window)
    device_label = tk.Label(text='Device: ', master=frame).pack(side=tk.LEFT)
    var_device = tk.StringVar(window, value='cpu')
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    device_menu = tk.OptionMenu(frame, var_device, *devices).pack(side=tk.LEFT)
    frame.pack(fill=tk.X)
    
    # checkbox clean
    frame = tk.Frame(window)
    use_postprocess = tk.BooleanVar(window, value=False)
    checkbox_postprocess = tk.Checkbutton(master=frame, text='Apply post-processing', variable=use_postprocess)
    checkbox_postprocess.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

    # run button
    button_run = tk.Button(text='Start Processing', command=lambda:start_processing(input_path.get(), output_path.get(), use_postprocess.get(), window, progressbar, var_device.get(), button_run))
    button_run.pack()

    # progressbar
    progressbar = ttk.Progressbar(length=500, maximum=100)
    progressbar.pack()

    window.mainloop()

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ui', '--ui-mode', action='store_true', help='Activate UI mode. Ignores all other options given.')
    parser.add_argument('-i', '--input_path', type=str, help='The path to the input image containing VH and VV bands (in this order).')
    parser.add_argument('-o', '--output-path', type=str, help='The path to the final result.')
    parser.add_argument('-pp', '--post-processing', action='store_true', help='Wheter or not to apply postprocessing and reduce noise in the results.')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='The device used to run the inference. Example values are "cpu", "cuda", and "mps".')
    parser.add_argument('-m', '--model', type=str, help='Model to use for the prediction.')

    args = parser.parse_args()

    if not args.ui_mode:
        if not args.input_path:
            raise ValueError('Input path must be given with -i or --input-path.')
        if not args.output_path:
            raise ValueError('Output path must be given with -o or --output-path.')

    if args.ui_mode:
        build_ui()
    else:
        try:
            torch.zeros(1).to(args.device)
            DEVICE = args.device
        except:
            warnings.warn(f'Device "{args.device}" not available, defaulting to "cpu".')
            DEVICE = 'cpu'
        start_processing(args.model, args.input_path, args.output_path, post_processing=args.post_processing)