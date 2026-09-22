import numpy as np
import rasterio as r
from rasterio import windows
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import pystac_client
import planetary_computer

from src.inference import SENTINEL_2_BAND_IDS

PLANETARY_COMPUTER_STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
SENTINEL_2_COLLECTION = 'sentinel-2-l2a'
SENTINEL_2_BAND_ORDER = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']

# Radiometrically Terrain Corrected: analysis-ready, calibrated linear-scale
# backscatter, unlike the raw sentinel-1-grd collection which needs the
# thermal-noise-removal/calibration/terrain-correction steps this tool's
# README has users run manually in SNAP.
SENTINEL_1_COLLECTION = 'sentinel-1-rtc'
SENTINEL_1_BAND_ORDER = ['vh', 'vv']

def search_sentinel_2(bbox, date_start, date_end, max_cloud_cover=90, limit=50):
    '''
    bbox: (min_lon, min_lat, max_lon, max_lat), in EPSG:4326.
    date_start, date_end: 'YYYY-MM-DD' strings.
    Returns signed pystac Items (most recent first), so their asset hrefs can be read directly.
    '''
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL_2_COLLECTION],
        bbox=bbox,
        datetime=f'{date_start}/{date_end}',
        query={'eo:cloud_cover': {'lt': max_cloud_cover}},
        limit=limit
    )
    items = list(search.item_collection())
    # the server's query-extension support is inconsistent, so re-check client-side
    items = [item for item in items if item.properties.get('eo:cloud_cover', 0) < max_cloud_cover]
    items.sort(key=lambda item: item.datetime, reverse=True)
    return items

def download_sentinel_2_window(item, bbox, output_path, progress_callback=None):
    '''
    Downloads only the bbox-cropped region (EPSG:4326: min_lon, min_lat, max_lon, max_lat)
    of the blue/green/red/nir/swir16/swir22 bands directly from the item's remote
    Cloud-Optimized GeoTIFF assets (only the needed pixels are fetched, not the full scene),
    and writes them as a single local GeoTIFF compatible with the tool's Sentinel-2 input.

    progress_callback(bands_done, bands_total), if given, is called after each band
    (including SCL) finishes downloading.
    '''
    asset_hrefs = [item.assets[SENTINEL_2_BAND_IDS[name]].href for name in SENTINEL_2_BAND_ORDER]
    band_names = SENTINEL_2_BAND_ORDER + ['SCL']
    bands_total = len(band_names)
    bands_done = 0

    with r.open(asset_hrefs[0]) as ref:
        ref_bounds = transform_bounds('EPSG:4326', ref.crs, *bbox)
        ref_window = windows.from_bounds(*ref_bounds, transform=ref.transform)
        ref_window = windows.Window(
            round(ref_window.col_off), round(ref_window.row_off),
            round(ref_window.width), round(ref_window.height)
        )
        ref_transform = ref.window_transform(ref_window)
        ref_shape = (ref_window.height, ref_window.width)
        ref_crs = ref.crs
        blue = ref.read(1, window=ref_window, boundless=True, fill_value=0)
    bands_done += 1
    if progress_callback:
        progress_callback(bands_done, bands_total)

    bands = [blue]
    for href in asset_hrefs[1:]:
        with r.open(href) as dataset:
            window_bounds = transform_bounds('EPSG:4326', dataset.crs, *bbox)
            window = windows.from_bounds(*window_bounds, transform=dataset.transform)
            bands.append(dataset.read(1, window=window, out_shape=ref_shape, resampling=Resampling.bilinear, boundless=True, fill_value=0))
        bands_done += 1
        if progress_callback:
            progress_callback(bands_done, bands_total)

    # SCL (scene classification, for cloud/cloud-shadow masking) is categorical,
    # so it must be resampled with nearest neighbor, not bilinear
    with r.open(item.assets['SCL'].href) as dataset:
        window_bounds = transform_bounds('EPSG:4326', dataset.crs, *bbox)
        window = windows.from_bounds(*window_bounds, transform=dataset.transform)
        bands.append(dataset.read(1, window=window, out_shape=ref_shape, resampling=Resampling.nearest, boundless=True, fill_value=0))
    bands_done += 1
    if progress_callback:
        progress_callback(bands_done, bands_total)

    data = np.stack(bands).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'height': ref_shape[0],
        'width': ref_shape[1],
        'count': len(band_names),
        'dtype': 'float32',
        'crs': ref_crs,
        'transform': ref_transform,
        'compress': 'lzw',
    }
    with r.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = band_names

    return output_path

def search_sentinel_1(bbox, date_start, date_end, limit=50):
    '''
    bbox: (min_lon, min_lat, max_lon, max_lat), in EPSG:4326.
    date_start, date_end: 'YYYY-MM-DD' strings.
    Returns signed pystac Items (most recent first) that have both VH and VV assets.
    '''
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL_1_COLLECTION],
        bbox=bbox,
        datetime=f'{date_start}/{date_end}',
        limit=limit
    )
    items = list(search.item_collection())
    items = [item for item in items if 'vh' in item.assets and 'vv' in item.assets]
    items.sort(key=lambda item: item.datetime, reverse=True)
    return items

def download_sentinel_1_window(item, bbox, output_path, progress_callback=None):
    '''
    Downloads only the bbox-cropped region (EPSG:4326: min_lon, min_lat, max_lon, max_lat)
    of the VH and VV bands directly from the item's remote Cloud-Optimized GeoTIFF assets
    (only the needed pixels are fetched, not the full scene), and writes them as a single
    local GeoTIFF (VH first, then VV, as linear-scale backscatter) compatible with the
    tool's Sentinel-1 input.

    progress_callback(bands_done, bands_total), if given, is called after each band
    finishes downloading.
    '''
    asset_hrefs = [item.assets[name].href for name in SENTINEL_1_BAND_ORDER]
    bands_total = len(SENTINEL_1_BAND_ORDER)
    bands_done = 0

    with r.open(asset_hrefs[0]) as ref:
        ref_bounds = transform_bounds('EPSG:4326', ref.crs, *bbox)
        ref_window = windows.from_bounds(*ref_bounds, transform=ref.transform)
        ref_window = windows.Window(
            round(ref_window.col_off), round(ref_window.row_off),
            round(ref_window.width), round(ref_window.height)
        )
        ref_transform = ref.window_transform(ref_window)
        ref_shape = (ref_window.height, ref_window.width)
        ref_crs = ref.crs
        ref_native_transform = ref.transform
        ref_native_shape = ref.shape
        vh = ref.read(1, window=ref_window, boundless=True, fill_value=0)
        # the source's own nodata value (e.g. -32768) can appear inside the scene's
        # actual bounds too, not just in the boundless padding; remap it to the
        # tool's own nodata convention (0), same as SNAP-exported inputs
        if ref.nodata is not None:
            vh[vh == ref.nodata] = 0
    bands_done += 1
    if progress_callback:
        progress_callback(bands_done, bands_total)

    bands = [vh]
    for href in asset_hrefs[1:]:
        with r.open(href) as dataset:
            # VH/VV in this collection are always jointly geocoded onto the same grid,
            # so reuse the exact same integer-snapped window rather than resampling:
            # bilinear resampling against a fractional window doesn't mask nodata
            # (-32768 here) before interpolating, which corrupts values near nodata
            # edges into wild negative outliers.
            if dataset.transform != ref_native_transform or dataset.shape != ref_native_shape:
                raise IOError(f'Unexpected: {href} is not co-registered with the VH band.')
            band = dataset.read(1, window=ref_window, boundless=True, fill_value=0)
            if dataset.nodata is not None:
                band[band == dataset.nodata] = 0
            bands.append(band)
        bands_done += 1
        if progress_callback:
            progress_callback(bands_done, bands_total)

    data = np.stack(bands).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'height': ref_shape[0],
        'width': ref_shape[1],
        'count': len(SENTINEL_1_BAND_ORDER),
        'dtype': 'float32',
        'crs': ref_crs,
        'transform': ref_transform,
        'nodata': 0,
        'compress': 'lzw',
    }
    with r.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = SENTINEL_1_BAND_ORDER

    return output_path
