import numpy as np
from distancemap import distance_map

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