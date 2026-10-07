import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import rasterio as r
import numpy as np
import os

def load_input_data_for_plotting(path_input_data, data_type):
    if os.path.isdir(path_input_data):
        # Sentinel-2 .SAFE folder
        from src.inference import load_and_normalize_sentinel_2_safe_folder
        return load_and_normalize_sentinel_2_safe_folder(path_input_data)[0]
    with r.open(path_input_data) as dataset:
        return dataset.read()

def prepare_input_data_for_plotting(input_data, data_type):
    data_type = data_type.lower()
    input_data = input_data.astype(np.float32)

    if data_type=='sentinel-1':
        input_data = np.concatenate([input_data[:2], (input_data[0]/input_data[1])[None,:,:]], axis=0)
    else:
        # Sentinel-2 (blue, green, red, ...) and PlanetScope (B, G, R, N): show true color
        input_data = input_data[[2,1,0]]

    for i in range(input_data.shape[0]):
        statistics = np.percentile(input_data[i].ravel(), q=[1,99])
        input_data[i] = (input_data[i]-statistics[0])/(statistics[1]-statistics[0])

    input_data = np.clip(input_data, 0, 1)
    input_data[~np.isfinite(input_data)] = 0
    input_data = np.moveaxis(input_data, 0, -1)

    return input_data

def plot_results(path_input_data, path_result, data_type, parent=None):
    '''
    Shows the input data next to the flood map. If `parent` (a tkinter widget) is
    given, the plot opens in a new tkinter window owned by it; otherwise it uses
    matplotlib's own window (blocking).
    '''
    input_data = load_input_data_for_plotting(path_input_data, data_type)
    with r.open(path_result) as dataset:
        result_data = dataset.read()

    if parent is None:
        f,ax = plt.subplots(1,2)
    else:
        # not using pyplot here: its GUI backend can conflict with the running tkinter app
        f = Figure()
        ax = f.subplots(1,2)

    ax[0].imshow(prepare_input_data_for_plotting(input_data, data_type))
    ax[1].imshow(result_data[0], cmap='Blues', vmin=0.5, vmax=2)

    ax[0].axis('off')
    ax[1].axis('off')

    ax[0].title.set_text('Input Data')
    ax[1].title.set_text('Flood Map')

    f.tight_layout()

    if parent is None:
        plt.show()
    else:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        top = tk.Toplevel(parent)
        top.title('Processing Results')
        canvas = FigureCanvasTkAgg(f, master=top)
        NavigationToolbar2Tk(canvas, top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

if __name__=='__main__':
    plot_results(
        '/Users/bruno.matosak/Downloads/test_s2.tif',
        '/Users/bruno.matosak/Downloads/test_s2_flood_mask.tif',
        'Sentinel-2'
    )
