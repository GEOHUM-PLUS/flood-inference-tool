import numpy as np
import rasterio as r
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
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

# Items of the same acquisition date can come from different, adjacent tiles/orbits
# that aren't necessarily on the same CRS/pixel grid (e.g. neighboring Sentinel-2
# tiles straddling a UTM zone boundary). Mosaicking onto one common, bbox-sized
# grid in a fixed CRS (matching the approach in the reference notebook) sidesteps
# that, at the cost of one resampling pass even for a single-item "group".
MOSAIC_CRS = 'EPSG:3857'
MOSAIC_RESOLUTION = 10  # meters

def group_items_by_date(items):
    '''
    Groups STAC items by acquisition date (UTC), so that several scenes covering
    the same AOI on the same date (e.g. adjacent tiles/orbits) can be downloaded
    and mosaicked together as one. Returns a list of (date, items) tuples, most
    recent date first; each group's items keep their original relative order.
    '''
    groups = {}
    for item in items:
        groups.setdefault(item.datetime.date(), []).append(item)
    return [(date, groups[date]) for date in sorted(groups, reverse=True)]

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

def get_sentinel_2_item_boa_offset(item):
    '''
    Mirrors get_sentinel_2_safe_folder_boa_offset in src/inference.py: Sentinel-2 L2A
    products since processing baseline 04.00 apply a constant -1000 offset to every
    band's digital numbers, to allow representing negative reflectances. Planetary
    Computer's STAC items don't expose the actual per-band metadata value (unlike a
    .SAFE product's own MTD_MSIL2A.xml), so the standard ESA constant is used instead.
    '''
    baseline = float(item.properties.get('s2:processing_baseline', '0'))
    return -1000 if baseline>=4.0 else 0

def build_mosaic_grid(bbox, resolution=MOSAIC_RESOLUTION):
    '''bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. Returns (transform, width, height, crs) of a common grid covering it, in MOSAIC_CRS.'''
    min_x, min_y, max_x, max_y = transform_bounds('EPSG:4326', MOSAIC_CRS, *bbox)
    width = max(1, round((max_x-min_x)/resolution))
    height = max(1, round((max_y-min_y)/resolution))
    transform = from_origin(min_x, max_y, resolution, resolution)
    return transform, width, height, MOSAIC_CRS

def read_band_mosaic(hrefs, transform, width, height, resampling, offsets=None):
    '''
    Reads one band from each of several remote COG assets (only the pixels inside
    the (transform, width, height) grid are fetched, not the whole scenes), onto
    that common grid, and composites them with first-valid-pixel-wins (items
    earlier in `hrefs` take priority). offsets, if given, is a per-item constant
    added to that item's values (after masking nodata) before compositing, e.g.
    to correct Sentinel-2's processing-baseline digital number offset.
    '''
    composite = None
    for index, href in enumerate(hrefs):
        with r.open(href) as dataset:
            with WarpedVRT(dataset, crs=MOSAIC_CRS, transform=transform, width=width, height=height,
                            resampling=resampling, src_nodata=dataset.nodata, nodata=0) as vrt:
                band = vrt.read(1).astype(np.float32)
        if offsets is not None and offsets[index]:
            valid = band!=0
            band[valid] = np.clip(band[valid]+offsets[index], 0, None)
        if composite is None:
            composite = band
        else:
            composite = np.where(composite!=0, composite, band)
    return composite

def download_sentinel_2_window(items, bbox, output_path, progress_callback=None):
    '''
    Downloads only the bbox-cropped region (EPSG:4326: min_lon, min_lat, max_lon, max_lat)
    of the blue/green/red/nir/swir16/swir22 bands directly from each item's remote
    Cloud-Optimized GeoTIFF assets (only the needed pixels are fetched, not the full
    scenes), mosaicking them together if more than one item is given (e.g. several
    scenes from the same date), and writes the result as a single local GeoTIFF
    compatible with the tool's Sentinel-2 input.

    progress_callback(bands_done, bands_total), if given, is called after each
    item/band (including SCL) combination finishes downloading.
    '''
    transform, width, height, crs = build_mosaic_grid(bbox)
    band_names = SENTINEL_2_BAND_ORDER + ['SCL']
    bands_total = len(band_names)*len(items)
    bands_done = 0

    offsets = [get_sentinel_2_item_boa_offset(item) for item in items]

    bands = []
    for name in SENTINEL_2_BAND_ORDER:
        hrefs = [item.assets[SENTINEL_2_BAND_IDS[name]].href for item in items]
        bands.append(read_band_mosaic(hrefs, transform, width, height, Resampling.bilinear, offsets=offsets))
        bands_done += len(items)
        if progress_callback:
            progress_callback(bands_done, bands_total)

    # SCL (scene classification, for cloud/cloud-shadow masking) is categorical,
    # so it must be resampled with nearest neighbor, not bilinear, and never offset
    scl_hrefs = [item.assets['SCL'].href for item in items]
    bands.append(read_band_mosaic(scl_hrefs, transform, width, height, Resampling.nearest))
    bands_done += len(items)
    if progress_callback:
        progress_callback(bands_done, bands_total)

    data = np.stack(bands).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': len(band_names),
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
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

def download_sentinel_1_window(items, bbox, output_path, progress_callback=None):
    '''
    Downloads only the bbox-cropped region (EPSG:4326: min_lon, min_lat, max_lon, max_lat)
    of the VH and VV bands directly from each item's remote Cloud-Optimized GeoTIFF
    assets (only the needed pixels are fetched, not the full scenes), mosaicking them
    together if more than one item is given (e.g. adjacent orbits/frames from the same
    date), and writes the result as a single local GeoTIFF (VH first, then VV, as
    linear-scale backscatter) compatible with the tool's Sentinel-1 input.

    progress_callback(bands_done, bands_total), if given, is called after each
    item/band combination finishes downloading.
    '''
    transform, width, height, crs = build_mosaic_grid(bbox)
    bands_total = len(SENTINEL_1_BAND_ORDER)*len(items)
    bands_done = 0

    bands = []
    for name in SENTINEL_1_BAND_ORDER:
        hrefs = [item.assets[name].href for item in items]
        bands.append(read_band_mosaic(hrefs, transform, width, height, Resampling.bilinear))
        bands_done += len(items)
        if progress_callback:
            progress_callback(bands_done, bands_total)

    data = np.stack(bands).astype(np.float32)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': len(SENTINEL_1_BAND_ORDER),
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
        'nodata': 0,
        'compress': 'lzw',
    }
    with r.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = SENTINEL_1_BAND_ORDER

    return output_path
