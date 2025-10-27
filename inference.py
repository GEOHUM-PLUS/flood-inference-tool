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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = "cpu"

def get_points_and_distance_map(s1, t, max_points_per_class_map=100, max_points_per_class_loss=500, p=0.3):
    s1_f = (s1[0]<p) & (s1[1]<p) & (t[1]<0.05)
    s1_n = (s1[0]>(1-p)) & (s1[1]>(1-p))

    c_f = np.asarray(np.where(s1_f))
    c_n = np.asarray(np.where(s1_n))

    inds_f = np.random.choice(np.arange(len(c_f[0])), min(max_points_per_class_map, len(c_f[0])))
    inds_n = np.random.choice(np.arange(len(c_n[0])), min(max_points_per_class_map, len(c_n[0])))

    coords_f_map = [c_f[0][inds_f], c_f[1][inds_f]]
    coords_n_map = [c_n[0][inds_n], c_n[1][inds_n]]

    if len(c_f[0])>len(c_n[0])*(1/3):
        dmap = distance_map((s1_f.shape[0], s1_f.shape[1]), np.transpose(coords_f_map))
        dmap = (dmap/np.max(dmap))[None,:,:]
        dmap = 1-dmap
    else:
        dmap = distance_map((s1_n.shape[0], s1_n.shape[1]), np.transpose(coords_n_map))
        dmap = (dmap/np.max(dmap))[None,:,:]
    
    inds_f = np.random.choice(np.arange(len(c_f[0])), min(max_points_per_class_loss, len(c_f[0])))
    inds_n = np.random.choice(np.arange(len(c_n[0])), min(max_points_per_class_loss, len(c_n[0])))

    coords_f_loss = [c_f[0][inds_f], c_f[1][inds_f]]
    coords_n_loss = [c_n[0][inds_n], c_n[1][inds_n]]

    return coords_f_loss, coords_n_loss, dmap

class DataScaler:
    def __init__(self):
        with open('percentile_limits.pickle', 'rb') as f:
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

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, dropout=0.2, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv_1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv_2 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            self.get_conv_block(4, base, dropout),
            self.get_conv_block(base, base, dropout),
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout),
        )
        self.conv4 = nn.Sequential(
            self.get_conv_block(base*4, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(base*2, base, dropout),
            self.get_conv_block(base, base, dropout),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        end1 = self.conv1(x)
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        up1 = self.transp_conv_1(end3)

        end4 = self.conv4(torch.cat([up1, end2], axis=1))
        up2 = self.transp_conv_2(end4)

        end5 = self.conv5(torch.cat([up2, end1], axis=1))

        return self.out(end5)
    
    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

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

def tile_cleaner(data, tile_size=1000, min_feature_size_px=16, nodata_val=255):
    result = np.zeros(data.shape, dtype=np.uint8)
    data2 = data.copy()
    data2[data2==nodata_val] = 0

    for i in tqdm(range(0, data.shape[0], tile_size-min_feature_size_px)):
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

def inference(model, path_input_image, result_path, clean_result=False, ui=None):

    if not ui is None:
        ui['progress_bar']['value'] = 0

    model.eval()

    data_scaler = DataScaler()
    dataset_sar = r.open(path_input_image)

    vv = dataset_sar.read(2)
    vh = dataset_sar.read(1)
    if not ui is None:
        ui['button_run']["text"] = "Downloading DEM..."

    slope = get_slope(path_input_image)

    if not ui is None:
        ui['button_run']["text"] = "Running inference..."

    inference = np.zeros([dataset_sar.height, dataset_sar.width])+255

    starting_coordinates_batches = []
    batch_size = 8
    batch = []
    for i in range(0, dataset_sar.height, 256):
        if i+256>dataset_sar.height:
            i = dataset_sar.height-256
        
        for j in range(0, dataset_sar.width, 256):
            if j+256>dataset_sar.width:
                j = dataset_sar.width-256

            if np.sum(vv[i:i+256, j:j+256])!=0:
                batch.append((i,j))
                if len(batch)>=batch_size:
                    starting_coordinates_batches.append(batch)
                    batch = []
    
    if batch:
        starting_coordinates_batches.append(batch)

    batch_count = 0
    for batch in tqdm(starting_coordinates_batches):
        batch_s1 = []
        batch_t = []
        batch_d = []
        for i,j in batch:
            s1 = data_scaler.scale_data('s1_during_flood', np.stack([10*np.log10(vv[i:i+256, j:j+256]), 10*np.log10(vh[i:i+256, j:j+256])]))
            t = data_scaler.scale_data('terrain', np.stack([np.zeros([256,256]), slope[i:i+256, j:j+256]]))
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
            tiles_pred = torch.argmax(model(batch_data).detach().cpu(), axis=1)
            index = 0
            for i,j in batch:
                inference[i:i+256, j:j+256] = tiles_pred[index].numpy()
                index+=1
        
        batch_count+=1
        if not ui is None:
            ui['progress_bar']['value'] = 100*batch_count/len(starting_coordinates_batches)
            # ui['progress_bar'].step(len(starting_coordinates_batches)/100)
            # ui['window'].update()
            # ui['window'].update_idletasks()
    
    # final correction
    inference[vv==0] = 2

    # clean if necessary
    if clean_result:
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

def start_processing(input_image_path, output_path, post_processing=False, window=None, pb=None, device=None, bt_run=None):
    # TODO: check if inputs are valid
    # TODO: Disable everything in the UI

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
    model = SimpleUNet().to(DEVICE)
    data = torch.load('models/model-0218.pt', map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(data['model_state_dict'])
    Thread(
        target=inference, 
        args=(
            model, 
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
    # img = Image.open('figures/Question_mark.png').resize((15,15))
    # img = ImageTk.PhotoImage(img)
    # qm = tk.Label(frame_input_1, image=img)
    # qm.image = img
    # qm.pack(side=tk.LEFT)
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
        start_processing(args.input_path, args.output_path, post_processing=args.post_processing)