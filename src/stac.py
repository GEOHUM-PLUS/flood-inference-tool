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

def download_sentinel_2_window(item, bbox, output_path):
    '''
    Downloads only the bbox-cropped region (EPSG:4326: min_lon, min_lat, max_lon, max_lat)
    of the blue/green/red/nir/swir16/swir22 bands directly from the item's remote
    Cloud-Optimized GeoTIFF assets (only the needed pixels are fetched, not the full scene),
    and writes them as a single local GeoTIFF compatible with the tool's Sentinel-2 input.
    '''
    asset_hrefs = [item.assets[SENTINEL_2_BAND_IDS[name]].href for name in SENTINEL_2_BAND_ORDER]

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

    bands = [blue]
    for href in asset_hrefs[1:]:
        with r.open(href) as dataset:
            window_bounds = transform_bounds('EPSG:4326', dataset.crs, *bbox)
            window = windows.from_bounds(*window_bounds, transform=dataset.transform)
            bands.append(dataset.read(1, window=window, out_shape=ref_shape, resampling=Resampling.bilinear, boundless=True, fill_value=0))

    # SCL (scene classification, for cloud/cloud-shadow masking) is categorical,
    # so it must be resampled with nearest neighbor, not bilinear
    with r.open(item.assets['SCL'].href) as dataset:
        window_bounds = transform_bounds('EPSG:4326', dataset.crs, *bbox)
        window = windows.from_bounds(*window_bounds, transform=dataset.transform)
        bands.append(dataset.read(1, window=window, out_shape=ref_shape, resampling=Resampling.nearest, boundless=True, fill_value=0))

    band_names = SENTINEL_2_BAND_ORDER + ['SCL']
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
