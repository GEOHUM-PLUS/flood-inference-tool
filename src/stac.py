import os
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import rasterio as r
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.control import GroundControlPoint
import pystac_client
import planetary_computer

from src.inference import SENTINEL_2_BAND_IDS

PLANETARY_COMPUTER_STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
EARTH_SEARCH_STAC_URL = 'https://earth-search.aws.element84.com/v1'
SENTINEL_2_COLLECTION = 'sentinel-2-l2a'
SENTINEL_2_L1C_COLLECTION = 'sentinel-2-l1c'
SENTINEL_2_BAND_ORDER = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']

# Radiometrically Terrain Corrected: analysis-ready, calibrated linear-scale
# backscatter. The alternative, raw sentinel-1-grd, needs the thermal-noise-
# removal/calibration/terrain-correction steps this tool's README has users
# run manually in SNAP; this tool calibrates it (see download_sentinel_1_grd_window)
# but doesn't terrain-correct it, so it isn't geometrically equivalent to RTC
# (see MANUAL.md) -- it's offered because, for some use cases, users have found
# its results preferable despite that.
SENTINEL_1_COLLECTION = 'sentinel-1-rtc'
SENTINEL_1_GRD_COLLECTION = 'sentinel-1-grd'
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
    dB-scale backscatter) compatible with the tool's Sentinel-1 input, with the
    "SAR data is in dB" option checked.

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

    # the source assets are linear-scale; compositing above relies on 0 meaning
    # "no data" (a real dB value can legitimately be 0, so that must happen before
    # converting), and the final file keeps that same nodata=0 convention after
    # conversion rather than leaving nodata pixels as -inf
    data = np.stack(bands).astype(np.float32)
    nodata_mask = data==0
    with np.errstate(divide='ignore'):
        data = 10*np.log10(data)
    data[nodata_mask] = 0

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

def search_sentinel_1_grd(bbox, date_start, date_end, limit=50):
    '''Same as search_sentinel_1, but against the raw (uncalibrated, ungeocoded) sentinel-1-grd collection.'''
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[SENTINEL_1_GRD_COLLECTION],
        bbox=bbox,
        datetime=f'{date_start}/{date_end}',
        limit=limit
    )
    items = list(search.item_collection())
    items = [item for item in items if 'vh' in item.assets and 'vv' in item.assets]
    items.sort(key=lambda item: item.datetime, reverse=True)
    return items

def _parse_s1_grd_lut(href, value_tag):
    '''
    Parses one of a GRD item's calibration/noise annotation XML files (ESA's
    standard format: a coarse line/pixel grid of values across the scene) into
    arrays suitable for scipy.interpolate.RegularGridInterpolator.
    '''
    with urllib.request.urlopen(href) as resp:
        content = resp.read()
    root = ET.fromstring(content)
    list_tag = 'calibrationVectorList' if value_tag=='sigmaNought' else 'noiseRangeVectorList'
    vec_tag = 'calibrationVector' if value_tag=='sigmaNought' else 'noiseRangeVector'
    lines, grids, pixels_ref = [], [], None
    for vec in root.find(list_tag).findall(vec_tag):
        pixels = [int(x) for x in vec.find('pixel').text.split()]
        if pixels_ref is None:
            pixels_ref = pixels
        lines.append(int(vec.find('line').text))
        grids.append([float(x) for x in vec.find(value_tag).text.split()])
    return np.array(lines, dtype=np.float64), np.array(pixels_ref, dtype=np.float64), np.array(grids, dtype=np.float64)

def _interp_s1_grd_lut(lines, pixels, grid, row_coords, col_coords):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lines, pixels), grid, bounds_error=False, fill_value=None)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing='ij')
    return interp(np.stack([rr.ravel(), cc.ravel()], axis=-1)).reshape(rr.shape)

def _native_window_for_bbox(ds, bbox, margin_px=100):
    '''
    Raw GRD assets have no CRS/affine transform -- they're geocoded with a
    coarse grid of Ground Control Points (GCPs) in EPSG:4326 instead (native
    ground-range geometry, not yet orthorectified). rasterio.transform.from_gcps
    fits a single approximate affine from them, good enough (plus a safety
    margin) to locate the native-pixel window covering our bbox; the precise
    GCP-based warp itself is done later by WarpedVRT once the array carries
    real GCPs (see calibrate_sentinel_1_grd_band).
    '''
    from rasterio.transform import from_gcps
    approx_transform = ~from_gcps(ds.gcps[0])
    min_lon, min_lat, max_lon, max_lat = bbox
    cols, rows = [], []
    for lon, lat in [(min_lon,min_lat), (min_lon,max_lat), (max_lon,min_lat), (max_lon,max_lat)]:
        col, row = approx_transform*(lon, lat)
        cols.append(col); rows.append(row)
    col0, col1 = max(0, int(min(cols))-margin_px), min(ds.width, int(max(cols))+margin_px)
    row0, row1 = max(0, int(min(rows))-margin_px), min(ds.height, int(max(rows))+margin_px)
    return windows.Window(col0, row0, col1-col0, row1-row0)

def calibrate_sentinel_1_grd_band(item, pol, transform, width, height):
    '''
    Radiometrically calibrates (digital numbers -> sigma0 backscatter, with
    thermal noise subtracted) one polarization of a raw sentinel-1-grd item,
    using its own calibration/noise annotation files, then warps it onto the
    given common (transform, width, height) grid in MOSAIC_CRS. This replicates
    SNAP's "S-1 Thermal Noise Removal" + "Calibrate" steps, but *not*
    "Range-Doppler Terrain Correction" -- there's no DEM-based orthorectification
    here, only the GCP-based geocoding every GRD product already carries. See
    MANUAL.md for what that means for accuracy relative to RTC.
    '''
    dn_href = item.assets[pol].href
    cal_href = item.assets[f'schema-calibration-{pol}'].href
    noise_href = item.assets[f'schema-noise-{pol}'].href

    with r.open(dn_href) as ds:
        win = _native_window_for_bbox(ds, transform_bounds(MOSAIC_CRS, 'EPSG:4326',
            *windows.bounds(windows.Window(0, 0, width, height), transform)))
        dn = ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        window_gcps = [GroundControlPoint(row=g.row-win.row_off, col=g.col-win.col_off, x=g.x, y=g.y, z=g.z, id=g.id)
                       for g in ds.gcps[0]]
        gcp_crs = ds.gcps[1]

    row_coords = win.row_off + np.arange(win.height)
    col_coords = win.col_off + np.arange(win.width)
    cal_lines, cal_pixels, cal_grid = _parse_s1_grd_lut(cal_href, 'sigmaNought')
    sigma_lut = _interp_s1_grd_lut(cal_lines, cal_pixels, cal_grid, row_coords, col_coords)
    noise_lines, noise_pixels, noise_grid = _parse_s1_grd_lut(noise_href, 'noiseRangeLut')
    noise_lut = _interp_s1_grd_lut(noise_lines, noise_pixels, noise_grid, row_coords, col_coords)

    valid = dn!=0
    sigma0 = np.zeros_like(dn)
    # thermal-noise subtraction can fully cancel the signal over very smooth/dark
    # surfaces (e.g. open water -- exactly where a flood map needs signal), giving
    # an exact-zero sigma0 for an otherwise genuinely valid pixel. Flooring those
    # to a small positive epsilon (a physically tiny backscatter value, ~-80 dB)
    # instead of leaving them at 0 keeps them distinguishable from true nodata
    # and avoids log10(0)=-inf corrupting whole tiles through the model's
    # receptive field later on.
    calibrated = np.clip(dn[valid]**2-noise_lut[valid], 0, None)/(sigma_lut[valid]**2)
    sigma0[valid] = np.where(calibrated==0, 1e-8, calibrated)
    sigma0 = sigma0.astype(np.float32)

    # Step 1: GCP-based warp (the "ellipsoid geocoding" every GRD product gets)
    # onto a plain, affine-georeferenced grid, letting GDAL auto-compute the
    # destination extent/resolution -- passing an explicit transform/width/height
    # here makes rasterio's WarpedVRT silently return an all-nodata result for a
    # GCP-only source, so that's done as a separate, ordinary step 2 instead.
    tmp_path = f'/vsimem/_s1_grd_{pol}_{id(sigma0)}.tif'
    with r.open(tmp_path, 'w', driver='GTiff', height=sigma0.shape[0], width=sigma0.shape[1],
                count=1, dtype='float32', nodata=0, gcps=window_gcps, crs=gcp_crs) as dst:
        dst.write(sigma0, 1)
    try:
        with r.open(tmp_path) as src:
            with WarpedVRT(src, src_crs=gcp_crs, crs=MOSAIC_CRS, resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt1:
                step1_transform, step1_w, step1_h = vrt1.transform, vrt1.width, vrt1.height
                step1_data = vrt1.read(1)
    finally:
        rio_shutil.delete(tmp_path)

    tmp_path2 = f'/vsimem/_s1_grd_{pol}_{id(sigma0)}_geocoded.tif'
    with r.open(tmp_path2, 'w', driver='GTiff', height=step1_h, width=step1_w, count=1,
                dtype='float32', crs=MOSAIC_CRS, transform=step1_transform, nodata=0) as dst:
        dst.write(step1_data, 1)
    try:
        # Step 2: ordinary affine-to-affine resample onto the requested grid
        with r.open(tmp_path2) as src2:
            with WarpedVRT(src2, crs=MOSAIC_CRS, transform=transform, width=width, height=height,
                            resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt2:
                return vrt2.read(1)
    finally:
        rio_shutil.delete(tmp_path2)

def download_sentinel_1_grd_window(items, bbox, output_path, progress_callback=None):
    '''
    Downloads only the bbox-cropped region of the VH and VV bands from each raw
    sentinel-1-grd item, radiometrically calibrates them (see
    calibrate_sentinel_1_grd_band), mosaics them together if more than one item
    is given, and writes the result as a single local GeoTIFF (VH first, then
    VV, as dB-scale backscatter) compatible with the tool's Sentinel-1 input --
    the same format download_sentinel_1_window produces, but NOT terrain-corrected.

    progress_callback(bands_done, bands_total), if given, is called after each
    item/band combination finishes downloading.
    '''
    transform, width, height, crs = build_mosaic_grid(bbox)
    bands_total = len(SENTINEL_1_BAND_ORDER)*len(items)
    bands_done = 0

    bands = []
    for name in SENTINEL_1_BAND_ORDER:
        composite = None
        for item in items:
            band = calibrate_sentinel_1_grd_band(item, name, transform, width, height)
            composite = band if composite is None else np.where(composite!=0, composite, band)
            bands_done += 1
            if progress_callback:
                progress_callback(bands_done, bands_total)
        bands.append(composite)

    data = np.stack(bands).astype(np.float32)
    nodata_mask = data==0
    with np.errstate(divide='ignore'):
        data = 10*np.log10(data)
    data[nodata_mask] = 0

    profile = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': len(SENTINEL_1_BAND_ORDER),
        'dtype': 'float32', 'crs': crs, 'transform': transform, 'nodata': 0, 'compress': 'lzw',
    }
    with r.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = SENTINEL_1_BAND_ORDER

    return output_path


def search_sentinel_2_l1c(bbox, date_start, date_end, max_cloud_cover=90, limit=50):
    '''Same as search_sentinel_2, but against earth-search's sentinel-2-l1c collection (Planetary Computer only hosts L2A).'''
    catalog = pystac_client.Client.open(EARTH_SEARCH_STAC_URL)
    search = catalog.search(
        collections=[SENTINEL_2_L1C_COLLECTION],
        bbox=bbox,
        datetime=f'{date_start}/{date_end}',
        query={'eo:cloud_cover': {'lt': max_cloud_cover}},
        limit=limit
    )
    items = list(search.item_collection())
    items = [item for item in items if item.properties.get('eo:cloud_cover', 0) < max_cloud_cover]
    items.sort(key=lambda item: item.datetime, reverse=True)
    return items

def read_band_mosaic_unsigned(hrefs, transform, width, height, resampling, offsets=None):
    '''
    Same as read_band_mosaic, except src_nodata is always explicitly 0 rather
    than taken from the dataset's own (possibly unset) nodata tag, and requests
    are made anonymously: earth-search's raw sentinel-s2-l1c JP2 tiles don't
    embed a nodata value (unlike Planetary Computer's L2A COGs) -- leaving it as
    None would let WarpedVRT blend real zero-padding into nearby valid pixels --
    and the bucket is public but needs unsigned S3 requests.
    '''
    os.environ.setdefault('AWS_NO_SIGN_REQUEST', 'YES')
    composite = None
    for index, href in enumerate(hrefs):
        with r.open(href) as dataset:
            with WarpedVRT(dataset, crs=MOSAIC_CRS, transform=transform, width=width, height=height,
                            resampling=resampling, src_nodata=0, nodata=0) as vrt:
                band = vrt.read(1).astype(np.float32)
        if offsets is not None and offsets[index]:
            valid = band!=0
            band[valid] = np.clip(band[valid]+offsets[index], 0, None)
        if composite is None:
            composite = band
        else:
            composite = np.where(composite!=0, composite, band)
    return composite

def download_sentinel_2_l1c_window(items, bbox, output_path, progress_callback=None):
    '''
    Same as download_sentinel_2_window, but for Sentinel-2 L1C (top-of-atmosphere,
    earth-search) instead of L2A: no SCL band (L1C doesn't have one, so the
    result has no cloud mask -- the tool already handles a 6-band Sentinel-2
    file without one). Still corrects the same baseline-04.00+ digital-number
    offset L2A has (get_sentinel_2_item_boa_offset only looks at
    s2:processing_baseline, which both collections expose the same way).

    progress_callback(bands_done, bands_total), if given, is called after each
    item/band combination finishes downloading.
    '''
    transform, width, height, crs = build_mosaic_grid(bbox)
    bands_total = len(SENTINEL_2_BAND_ORDER)*len(items)
    bands_done = 0

    offsets = [get_sentinel_2_item_boa_offset(item) for item in items]

    bands = []
    for name in SENTINEL_2_BAND_ORDER:
        hrefs = [item.assets[name].href for item in items]
        bands.append(read_band_mosaic_unsigned(hrefs, transform, width, height, Resampling.bilinear, offsets=offsets))
        bands_done += len(items)
        if progress_callback:
            progress_callback(bands_done, bands_total)

    data = np.stack(bands).astype(np.float32)

    profile = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': len(SENTINEL_2_BAND_ORDER),
        'dtype': 'float32', 'crs': crs, 'transform': transform, 'compress': 'lzw',
    }
    with r.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = SENTINEL_2_BAND_ORDER

    return output_path
