import numpy as np
from distancemap import distance_map
import os
import pickle
import rioxarray
from rioxarray.merge import merge_arrays
import rasterio as r
from rasterio.crs import CRS
from rasterio import warp
import pystac_client
import planetary_computer
import geopandas
from shapely import Polygon
from skimage.morphology import area_opening, area_closing
from tqdm.auto import tqdm

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

def get_points_loss(s1, t, flood_mask, max_points_per_class_loss=500, p=0.3):
    s1_f = flood_mask==1 & (t[1]<0.1)
    s1_n = (s1[0]>(1-p)) & (s1[1]>(1-p))

    c_f = np.asarray(np.where(s1_f))
    c_n = np.asarray(np.where(s1_n))
    
    inds_f = np.random.choice(np.arange(len(c_f[0])), min(max_points_per_class_loss, len(c_f[0])))
    inds_n = np.random.choice(np.arange(len(c_n[0])), min(max_points_per_class_loss, len(c_n[0])))

    coords_f_loss = [c_f[0][inds_f], c_f[1][inds_f]]
    coords_n_loss = [c_n[0][inds_n], c_n[1][inds_n]]

    return coords_f_loss, coords_n_loss

class DataScaler:
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'statistics_s1.pickle'), 'rb') as f:
            self.STATISTICS_S1 = pickle.load(f)
            self.percentile_bttm = 5
            self.percentile_top = 95
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'statistics_planetscope.pickle'), 'rb') as f:
            self.STATISTICS_PLANETSCOPE = pickle.load(f)
    
    def scale_data(self, data_type, data):
        # follows scales only if needed
        if data_type in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'terrain']:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STATISTICS_S1[data_type][i][str(int(self.percentile_bttm))])/(self.STATISTICS_S1[data_type][i][str(int(self.percentile_top))]-self.STATISTICS_S1[data_type][i][str(int(self.percentile_bttm))])
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STATISTICS_S1[data_type][i]['0'])/(self.STATISTICS_S1[data_type][i]['100']-self.STATISTICS_S1[data_type][i]['0'])
        
        # clipping to 0 1
        data = np.clip(data, a_min=0, a_max=1)

        return data
    
    def normalize_data(self, data_type, data):
        # follows scales only if needed
        if data_type in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'terrain', 'global_surfece_water']:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STATISTICS_S1[data_type][i]['mean'])/self.STATISTICS_S1[data_type][i]['std']
        elif data_type == 'planetscope':
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STATISTICS_PLANETSCOPE['PS']['mean'][i])/self.STATISTICS_PLANETSCOPE['PS']['std'][i]
        elif data_type == 'LULC':
            # data = np.moveaxis(get_one_hot((data[0]/10).astype(np.byte), 11), -1,0)
            data = torch.nn.functional.one_hot(torch.Tensor((data[0]/10)-1).to(torch.long), num_classes=10).moveaxis(-1,0).numpy()
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STATISTICS_S1[data_type][i]['0'])/(self.STATISTICS_S1[data_type][i]['100']-self.STATISTICS_S1[data_type][i]['0'])

        return data
    
    def unnormalize_data(self, data_type, data):
        # follows scales only if needed
        if data_type in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'terrain', 'global_surfece_water']:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]*self.STATISTICS_S1[data_type][i]['std'])+self.STATISTICS_S1[data_type][i]['mean']
        if data_type == 'planetscope':
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]*self.STATISTICS['PS']['std'][i])+self.STATISTICS['PS']['mean'][i]
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]*(self.STATISTICS_S1[data_type][i]['100']-self.STATISTICS_S1[data_type][i]['0']))+self.STATISTICS_S1[data_type][i]['0']

        return data

def compute_slope_degrees(dem, transform, scale=111120.0, nodata_value=-9999.0):
    '''
    Slope in degrees, replicating gdaldem's "-alg ZevenbergenThorne -s <scale>"
    (without "-compute_edges") in pure numpy (no gdaldem CLI dependency).
    Like gdaldem's default, the outermost 1-pixel border (which has no full
    3x3 neighborhood) is set to `nodata_value` rather than computed.

    `scale` is the ratio of vertical (elevation) to horizontal units, same as
    gdaldem's `-s` flag; 111120 (meters per degree) is used because the DEM here
    is in geographic (EPSG:4326) coordinates while elevation is in meters.
    '''
    ew_res = abs(transform.a) * scale
    ns_res = abs(transform.e) * scale

    dem = dem.astype(np.float64)

    west = dem[1:-1, :-2]
    east = dem[1:-1, 2:]
    north = dem[:-2, 1:-1]
    south = dem[2:, 1:-1]

    dz_dx = (east - west) / (2 * ew_res)
    dz_dy = (south - north) / (2 * ns_res)

    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    result = np.full(dem.shape, nodata_value, dtype=np.float32)
    result[1:-1, 1:-1] = slope_deg.astype(np.float32)
    return result

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

    slope_data = compute_slope_degrees(merged.to_numpy()[0], merged.rio.transform())
    slope = merged.copy(data=slope_data[np.newaxis, :, :])
    slope = slope.rio.write_nodata(-9999.0)

    ref = rioxarray.open_rasterio(path_reference)
    matched = slope.rio.reproject_match(ref)

    return matched.to_numpy()[0]

# TODO: Paralellize this
def tile_cleaner(data, tile_size=1000, min_feature_size_px=16, foreground_val=1, background_val=0):
    '''
    Removes small noise specks from a two-class classification map by
    area-opening the foreground class (erasing small isolated foreground
    blobs) and area-closing it (filling small background holes), so noise
    is reclassified into whichever class surrounds it rather than being
    zeroed out into a third, undefined value. Runs in overlapping tiles to
    bound memory use on large rasters; pixel values other than
    foreground_val/background_val (e.g. true nodata) are left untouched by
    the caller applying its own nodata mask afterwards.
    '''
    foreground = data==foreground_val
    result = np.zeros(data.shape, dtype=bool)

    for i in tqdm(range(0, data.shape[0], tile_size-min_feature_size_px), ncols=70):
        if i+tile_size>result.shape[0]:
            i = result.shape[0]-tile_size
        for j in range(0, data.shape[1], tile_size-min_feature_size_px):
            if j+tile_size > result.shape[1]:
                j = result.shape[1]-tile_size

            if np.any(foreground[i:i+tile_size, j:j+tile_size]):
                result[i:i+tile_size, j:j+tile_size] |= area_closing(
                    area_opening(
                        foreground[i:i+tile_size, j:j+tile_size], min_feature_size_px
                    ), min_feature_size_px
                )

            if j == result.shape[1]-tile_size:
                break

        if i == result.shape[0]-tile_size:
            break

    cleaned = np.where(result, foreground_val, background_val).astype(data.dtype)
    # pixel values that were neither the foreground nor background class
    # (e.g. true nodata already encoded in `data`) are preserved as-is
    other = (data!=foreground_val) & (data!=background_val)
    cleaned[other] = data[other]
    return cleaned