import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch

from src.inference import start_processing

MODEL_OPTIONS = {
    'sentinel-1': ['UNet-S1.pt', 'DistanceMap.pt', 'Otsu_Threshold'],
    'sentinel-2': ['UNet-S2.pt'],
    'planetscope': ['UNet-PlanetScope.pt'],
}

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
    notebook.add(frame, text='Sentinel-1')
    build_data_tab(
        frame,
        data_type='sentinel-1',
        input_labels=['Input file:'],
        models=MODEL_OPTIONS['sentinel-1'],
        show_dB_checkbox=True
    )

    frame = ttk.Frame(notebook)
    notebook.add(frame, text='Sentinel-2')
    build_data_tab(
        frame,
        data_type='sentinel-2',
        input_labels=['Input file:'],
        models=MODEL_OPTIONS['sentinel-2'],
        show_dB_checkbox=False
    )

    frame = ttk.Frame(notebook)
    notebook.add(frame, text='PlanetScope')
    build_data_tab(
        frame,
        data_type='planetscope',
        input_labels=['Input file:', 'Input auxiliary file:'],
        models=MODEL_OPTIONS['planetscope'],
        show_dB_checkbox=False
    )

    notebook.pack(pady=(10, 0))

    # window.attributes('-topmost', True)

    window.mainloop()

def build_data_tab(window, data_type, input_labels, models, show_dB_checkbox):
    input_vars = [add_file_row(window, label) for label in input_labels]

    sar_is_dB = add_checkbox_row(window, 'SAR data is in dB') if show_dB_checkbox else None

    output_path = add_file_row(window, 'Output file:', save=True)

    var_model = add_dropdown_row(window, 'Model: ', models, '---')
    var_device = add_dropdown_row(window, 'Device: ', get_available_devices(), 'cpu')

    use_bayesian_dropout = add_checkbox_row(window, '(EXPERIMENTAL) Use Bayesian Dropout to estimate uncertainty')
    use_postprocess = add_checkbox_row(window, 'Remove noise from flood map')

    def run():
        input_info = {
            'input_files': [var.get() for var in input_vars],
            'data_type': data_type,
        }
        if sar_is_dB is not None:
            input_info['sar_is_dB'] = sar_is_dB.get()

        start_processing(
            model_name=var_model.get(),
            input_info=input_info,
            output_path=output_path.get(),
            post_processing=use_postprocess.get(),
            window=window,
            pb=progressbar,
            device=var_device.get(),
            bt_run=button_run,
            bayesian_dropout=use_bayesian_dropout.get()
        )

    button_run = tk.Button(window, text='Start Processing', command=run)
    button_run.pack()

    progressbar = ttk.Progressbar(window, length=500, maximum=100)
    progressbar.pack()

def add_file_row(window, label_text, save=False):
    frame_label = tk.Frame(window)
    frame_entry = tk.Frame(window)
    tk.Label(text=label_text, master=frame_label).pack(side=tk.LEFT)

    path_var = tk.StringVar(window)
    entry = tk.Entry(master=frame_entry, width=50, textvariable=path_var)
    entry.pack(side=tk.LEFT)

    browse = create_file_path if save else get_file_path
    tk.Button(master=frame_entry, text='...', command=lambda: browse(entry)).pack(side=tk.LEFT)

    frame_label.pack(fill=tk.X)
    frame_entry.pack(fill=tk.X)

    return path_var

def add_checkbox_row(window, text, default=False):
    frame = tk.Frame(window)
    var = tk.BooleanVar(window, value=default)
    tk.Checkbutton(master=frame, text=text, variable=var).pack(side=tk.LEFT)
    frame.pack(fill=tk.X)
    return var

def add_dropdown_row(window, label_text, options, default):
    frame = tk.Frame(window)
    tk.Label(text=label_text, master=frame).pack(side=tk.LEFT)
    var = tk.StringVar(window, value=default)
    tk.OptionMenu(frame, var, *options).pack(side=tk.LEFT)
    frame.pack(fill=tk.X)
    return var

def get_available_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    return devices

def get_file_path(entry):
    file = filedialog.askopenfilename(filetypes=[('TIF', '*.tif')])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

def create_file_path(entry):
    file = filedialog.asksaveasfilename(filetypes=[('TIF', '*.tif')])
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

def show_error(message):
    tk.messagebox.showerror(title='Error', message=message)

def alert_finished(output_path, elapsed_time_minutes):
    tk.messagebox.showinfo(title='Processing Finished', message=f'Process finished!\nLocation: {output_path}\nTime: {elapsed_time_minutes:.2f} minutes.')
