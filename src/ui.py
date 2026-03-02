import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import glob
import os
import datetime

from src.inference import start_processing

def build_ui():
    window = tk.Tk()
    window.title('GEOHUM Flood Inference Tool')
    ico = Image.open('figures/icon.ico')
    photo = ImageTk.PhotoImage(ico)
    window.wm_iconphoto(False, photo)

    img = Image.open('figures/gEOhum_Logo_NEWCD-Web.png')
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack(pady=(10, 0))

    title = tk.Label(text='Flood Inference Tool', master=window, font="Arial 25 bold")
    title.pack()

    notebook = ttk.Notebook(window)

    frame = ttk.Frame(notebook)
    notebook.add(frame,text='Sentinel-1')
    add_Sentinel_1_tab(frame)

    frame = ttk.Frame(notebook)
    notebook.add(frame,text='PlanetScope')
    add_PlanetScope_tab(frame)
    notebook.pack(pady=(10,0))

    # window.attributes('-topmost', True)
    
    window.mainloop()

def add_Sentinel_1_tab(window):
    # input
    frame_input_1 = tk.Frame(window)
    frame_input_2 = tk.Frame(window)
    input_label = tk.Label(text='Input file:', master=frame_input_1).pack(side=tk.LEFT)
    input_path = tk.StringVar(window)
    w_input_path = tk.Entry(master=frame_input_2, width=50, textvariable=input_path)
    w_input_path.pack(side=tk.LEFT)

    w_input_path_button = tk.Button(master=frame_input_2, text='...', command=lambda:get_file_path(w_input_path)).pack(side=tk.LEFT)

    frame_input_1.pack(fill=tk.X)
    frame_input_2.pack(fill=tk.X)

    # checkbox is dB
    frame = tk.Frame(window)
    sar_is_dB = tk.BooleanVar(window, value=False)
    checkbox_dB = tk.Checkbutton(master=frame, text='SAR data is in dB', variable=sar_is_dB)
    checkbox_dB.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

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

    # model options
    frame = tk.Frame(window)
    model_label = tk.Label(text='Model: ', master=frame).pack(side=tk.LEFT)
    var_model = tk.StringVar(window, value='---')
    models = ['UNet-S1.pt', 'DistanceMap.pt', 'Otsu_Threshold']
    model_menu = tk.OptionMenu(frame, var_model, *models).pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

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
    use_bayesian_dropout = tk.BooleanVar(window, value=False)
    checkbox_bayesian_dropout = tk.Checkbutton(master=frame, text='(EXPERIMENTAL) Use Bayesian Dropout to estimate uncertainty', variable=use_bayesian_dropout)
    checkbox_bayesian_dropout.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)
    
    # checkbox clean
    frame = tk.Frame(window)
    use_postprocess = tk.BooleanVar(window, value=False)
    checkbox_postprocess = tk.Checkbutton(master=frame, text='Remove noise from flood map', variable=use_postprocess)
    checkbox_postprocess.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

    # run button
    button_run = tk.Button(window,
        text = 'Start Processing',
        command = lambda:start_processing(
            model_name=var_model.get(),
            input_image_path=input_path.get(),
            output_path=output_path.get(),
            post_processing=use_postprocess.get(),
            window=window,
            pb=progressbar,
            device=var_device.get(),
            bt_run=button_run,
            sar_is_dB=sar_is_dB.get(),
            bayesian_dropout=use_bayesian_dropout.get()
        )
    )
    button_run.pack()

    # progressbar
    progressbar = ttk.Progressbar(window, length=500, maximum=100)
    progressbar.pack()

def add_PlanetScope_tab(window):
    # input
    frame_input_1 = tk.Frame(window)
    frame_input_2 = tk.Frame(window)
    input_label = tk.Label(text='Input file:', master=frame_input_1).pack(side=tk.LEFT)
    input_path = tk.StringVar(window)
    w_input_path = tk.Entry(master=frame_input_2, width=50, textvariable=input_path)
    w_input_path.pack(side=tk.LEFT)

    w_input_path_button = tk.Button(master=frame_input_2, text='...', command=lambda:get_file_path(w_input_path)).pack(side=tk.LEFT)

    frame_input_1.pack(fill=tk.X)
    frame_input_2.pack(fill=tk.X)

    # input
    frame_input_1 = tk.Frame(window)
    frame_input_2 = tk.Frame(window)
    input_label = tk.Label(text='Input auxiliary file:', master=frame_input_1).pack(side=tk.LEFT)
    input_path_auxiliary = tk.StringVar(window)
    w_input_path_aux = tk.Entry(master=frame_input_2, width=50, textvariable=input_path_auxiliary)
    w_input_path_aux.pack(side=tk.LEFT)

    w_input_path_button = tk.Button(master=frame_input_2, text='...', command=lambda:get_file_path(w_input_path_aux)).pack(side=tk.LEFT)

    frame_input_1.pack(fill=tk.X)
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

    # model options
    frame = tk.Frame(window)
    model_label = tk.Label(text='Model: ', master=frame).pack(side=tk.LEFT)
    var_model = tk.StringVar(window, value='---')
    models = ['UNet-PlanetScope.pt']
    model_menu = tk.OptionMenu(frame, var_model, *models).pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

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
    use_bayesian_dropout = tk.BooleanVar(window, value=False)
    checkbox_bayesian_dropout = tk.Checkbutton(master=frame, text='(EXPERIMENTAL) Use Bayesian Dropout to estimate uncertainty', variable=use_bayesian_dropout)
    checkbox_bayesian_dropout.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)
    
    # checkbox clean
    frame = tk.Frame(window)
    use_postprocess = tk.BooleanVar(window, value=False)
    checkbox_postprocess = tk.Checkbutton(master=frame, text='Remove noise from flood map', variable=use_postprocess)
    checkbox_postprocess.pack(side=tk.LEFT)
    frame.pack(fill=tk.X)

    # run button
    button_run = tk.Button(window,
        text = 'Start Processing',
        command = lambda:start_processing(
            model_name=var_model.get(),
            input_image_path=input_path.get(),
            output_path=output_path.get(),
            post_processing=use_postprocess.get(),
            window=window,
            pb=progressbar,
            device=var_device.get(),
            bt_run=button_run,
            bayesian_dropout=use_bayesian_dropout.get(),
            input_image_path_aux=input_path_auxiliary.get()
        )
    )
    button_run.pack()

    # progressbar
    progressbar = ttk.Progressbar(window, length=500, maximum=100)
    progressbar.pack()

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

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

def alert_finished(output_path, elapsed_time_minutes):
    tk.messagebox.showinfo(title='Processing Finished', message=f'Process finished!\nLocation: {output_path}\nTime: {elapsed_time_minutes:.2f} minutes.')