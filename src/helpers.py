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
import subprocess
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

# TODO: Paralellize this
def tile_cleaner(data, tile_size=1000, min_feature_size_px=16, nodata_val=255):
    result = np.zeros(data.shape, dtype=np.uint8)
    data2 = data.copy()
    data2[data2==nodata_val] = 0

    for i in tqdm(range(0, data.shape[0], tile_size-min_feature_size_px), ncols=70):
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