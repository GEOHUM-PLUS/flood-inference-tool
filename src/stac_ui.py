import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import datetime

from tkintermapview import TkinterMapView

from src.stac import search_sentinel_2, download_sentinel_2_window

DEFAULT_MAP_POSITION = (47.803, 13.035)
DEFAULT_MAP_ZOOM = 5
DEFAULT_SEARCH_WINDOW_DAYS = 30

def open_stac_search_window(parent, on_download):
    '''Opens a window to search Sentinel-2 scenes on Planetary Computer's STAC catalog by
    drawing a bounding box on a map, then download only that bbox's window of the required
    bands for a chosen scene. on_download(output_path) is called once a download finishes.'''
    StacSearchWindow(parent, on_download)

class StacSearchWindow:
    def __init__(self, parent, on_download):
        self.on_download = on_download
        self.bbox = None
        self.items = []
        self.drawing = False
        self.draw_start = None
        self.draw_rectangle_id = None
        self.bbox_polygon = None

        self.window = tk.Toplevel(parent)
        self.window.title('Search Sentinel-2 (STAC)')

        self.map_widget = TkinterMapView(self.window, width=760, height=480, corner_radius=0)
        self.map_widget.set_position(*DEFAULT_MAP_POSITION)
        self.map_widget.set_zoom(DEFAULT_MAP_ZOOM)
        self.map_widget.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.window)
        controls.pack(fill=tk.X, padx=5, pady=5)

        self.draw_button = tk.Button(controls, text='Draw bounding box', command=self.toggle_draw_mode)
        self.draw_button.grid(row=0, column=0, padx=(0, 10))

        self.bbox_label = tk.Label(controls, text='Bounding box: none selected')
        self.bbox_label.grid(row=0, column=1, columnspan=5, sticky='w')

        tk.Label(controls, text='From:').grid(row=1, column=0, sticky='e')
        date_end = datetime.date.today()
        date_start = date_end - datetime.timedelta(days=DEFAULT_SEARCH_WINDOW_DAYS)
        self.date_start_var = tk.StringVar(self.window, value=date_start.isoformat())
        tk.Entry(controls, width=12, textvariable=self.date_start_var).grid(row=1, column=1)

        tk.Label(controls, text='To:').grid(row=1, column=2, sticky='e')
        self.date_end_var = tk.StringVar(self.window, value=date_end.isoformat())
        tk.Entry(controls, width=12, textvariable=self.date_end_var).grid(row=1, column=3)

        tk.Label(controls, text='Max cloud cover (%):').grid(row=1, column=4, sticky='e')
        self.cloud_cover_var = tk.StringVar(self.window, value='90')
        tk.Entry(controls, width=6, textvariable=self.cloud_cover_var).grid(row=1, column=5)

        self.search_button = tk.Button(controls, text='Search', command=self.search)
        self.search_button.grid(row=1, column=6, padx=(10, 0))

        results_frame = tk.Frame(self.window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_listbox = tk.Listbox(results_frame, height=8, yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_listbox.yview)

        self.download_button = tk.Button(self.window, text='Download Selected', command=self.download_selected)
        self.download_button.pack(pady=(0, 5))

    def toggle_draw_mode(self):
        if not self.drawing:
            self.drawing = True
            self.draw_button.config(text='Click and drag on the map...')
            self.map_widget.canvas.bind('<Button-1>', self.on_draw_start)
            self.map_widget.canvas.bind('<B1-Motion>', self.on_draw_drag)
            self.map_widget.canvas.bind('<ButtonRelease-1>', self.on_draw_end)
        else:
            self.stop_drawing()

    def stop_drawing(self):
        self.drawing = False
        self.draw_start = None
        self.draw_button.config(text='Draw bounding box')
        self.map_widget.canvas.bind('<Button-1>', self.map_widget.mouse_click)
        self.map_widget.canvas.bind('<B1-Motion>', self.map_widget.mouse_move)
        self.map_widget.canvas.bind('<ButtonRelease-1>', self.map_widget.mouse_release)

    def on_draw_start(self, event):
        self.draw_start = (event.x, event.y)
        if self.draw_rectangle_id is not None:
            self.map_widget.canvas.delete(self.draw_rectangle_id)
            self.draw_rectangle_id = None

    def on_draw_drag(self, event):
        if self.draw_start is None:
            return
        if self.draw_rectangle_id is not None:
            self.map_widget.canvas.delete(self.draw_rectangle_id)
        x0, y0 = self.draw_start
        self.draw_rectangle_id = self.map_widget.canvas.create_rectangle(x0, y0, event.x, event.y, outline='red', width=2)

    def on_draw_end(self, event):
        if self.draw_start is None:
            self.stop_drawing()
            return
        x0, y0 = self.draw_start
        x1, y1 = event.x, event.y

        if self.draw_rectangle_id is not None:
            self.map_widget.canvas.delete(self.draw_rectangle_id)
            self.draw_rectangle_id = None

        lat0, lon0 = self.map_widget.convert_canvas_coords_to_decimal_coords(x0, y0)
        lat1, lon1 = self.map_widget.convert_canvas_coords_to_decimal_coords(x1, y1)

        min_lon, max_lon = sorted([lon0, lon1])
        min_lat, max_lat = sorted([lat0, lat1])
        self.bbox = (min_lon, min_lat, max_lon, max_lat)

        if self.bbox_polygon is not None:
            self.bbox_polygon.delete()
        self.bbox_polygon = self.map_widget.set_polygon(
            [(max_lat, min_lon), (max_lat, max_lon), (min_lat, max_lon), (min_lat, min_lon)],
            outline_color='red', fill_color=None, border_width=3
        )

        self.bbox_label.config(text=f'Bounding box: {min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}')
        self.stop_drawing()

    def search(self):
        if self.bbox is None:
            messagebox.showerror('Error', 'Please draw a bounding box on the map first.')
            return
        try:
            max_cloud_cover = float(self.cloud_cover_var.get())
        except ValueError:
            messagebox.showerror('Error', 'Max cloud cover must be a number.')
            return

        date_start = self.date_start_var.get()
        date_end = self.date_end_var.get()
        bbox = self.bbox

        self.search_button.config(state='disabled', text='Searching...')
        self.results_listbox.delete(0, tk.END)

        def run_search():
            try:
                items = search_sentinel_2(bbox, date_start, date_end, max_cloud_cover)
            except Exception as exc:
                self.window.after(0, lambda exc=exc: self.search_failed(exc))
                return
            self.window.after(0, lambda: self.search_done(items))

        Thread(target=run_search, daemon=True).start()

    def search_failed(self, exc):
        self.search_button.config(state='normal', text='Search')
        messagebox.showerror('Search failed', str(exc))

    def search_done(self, items):
        self.items = items
        self.search_button.config(state='normal', text='Search')
        for item in items:
            cloud_cover = item.properties.get('eo:cloud_cover', float('nan'))
            date_text = item.datetime.strftime('%Y-%m-%d %H:%M') if item.datetime else '?'
            self.results_listbox.insert(tk.END, f'{date_text}  |  cloud cover: {cloud_cover:5.1f}%  |  {item.id}')
        if not items:
            messagebox.showinfo('No results', 'No Sentinel-2 scenes found for this area, date range, and cloud cover.')

    def download_selected(self):
        selection = self.results_listbox.curselection()
        if not selection:
            messagebox.showerror('Error', 'Please select a scene from the results list first.')
            return
        item = self.items[selection[0]]

        output_path = filedialog.asksaveasfilename(defaultextension='.tif', filetypes=[('TIF', '*.tif')])
        if not output_path:
            return

        bbox = self.bbox
        self.download_button.config(state='disabled', text='Downloading...')

        def run_download():
            try:
                download_sentinel_2_window(item, bbox, output_path)
            except Exception as exc:
                self.window.after(0, lambda exc=exc: self.download_failed(exc))
                return
            self.window.after(0, lambda: self.download_done(output_path))

        Thread(target=run_download, daemon=True).start()

    def download_failed(self, exc):
        self.download_button.config(state='normal', text='Download Selected')
        messagebox.showerror('Download failed', str(exc))

    def download_done(self, output_path):
        self.download_button.config(state='normal', text='Download Selected')
        self.on_download(output_path)
        messagebox.showinfo('Download complete', f'Saved to:\n{output_path}')
        self.window.destroy()
