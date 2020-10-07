from pypcd import pypcd
import numpy as np

from  tqdm import tqdm

import sys

from os import listdir, mkdir
from os.path import join, isdir

import argparse

EPS = np.finfo(np.float32).eps

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def pca_numpy(array):
    S, U = np.linalg.eig(array.T @ array)
    return S, U

def add_noise(array, limit = 0.01):
    points = array[:, :3]
    normals = array[:, 3:]
    noise = normals * np.random.uniform(-limit, limit, (points.shape[0],1))
    points = points + noise.astype(np.float32)
    #not adding noise on normals yet
    noise_array = np.concatenate((points, normals), axis=1)
    return noise_array

def align_canonical(array):
    points = array[:, :3]
    normals = array[:, 3:]
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    points = (R @ points.T).T
    normals= (R @ normals.T).T
    aligned_array = np.concatenate((points, normals), axis=1)
    return aligned_array

def centralize(array):
    points = array[:, :3]
    mean = np.mean(points, 0)
    centralized_points = points - mean
    array[:, :3] = centralized_points
    return array

def rescale(array, factor = 1000):
    points = array[:, :3]
    scaled_points = points / (factor + EPS)
    array[:, :3] = scaled_points
    return array

def cube_rescale(array, factor = 1):
    points = array[:, :3]
    std = np.max(points, 0) - np.min(points, 0)
    scaled_points = (points / (np.max(std) + EPS))*factor
    array[:, :3] = scaled_points
    return array

def iterative_mkdir(folder_list):
    dir = ''
    for folder in folder_list:
        dir = join(dir, folder)
        if(not isdir(dir)):
            mkdir(dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud dataset generator.')
    parser.add_argument('input', type=str, help='input file in .pcd format or a folder with .pcd files. The .pcd files must have the normals too.')
    parser.add_argument('output', type=str, help='output folder.')
    parser.add_argument('--rescale_factor', type = float, default = 1, help='first rescale applied to the point cloud, used to change the measurement unit.')
    parser.add_argument('--noise_limit', type = float, default=0.01, help='limit noise applied to the point cloud.')
    parser.add_argument('--centralize', type = bool, default=True, help='bool to centralize or not.')
    parser.add_argument('--align', type = bool, default=True, help='bool to canonical alignment or not.')
    parser.add_argument('--cube_rescale_factor', type = float, default=1, help='argument to make the point cloud lie in a unit cube, the factor multiplies all the dimensions of result cube.')
    args = vars(parser.parse_args())

    inputname = args['input']
    outputname = args['output']
    rescale_factor = args['rescale_factor']
    noise_limit = args['noise_limit']
    is_centralize = args['centralize']
    is_align = args['align']
    cube_rescale_factor = args['cube_rescale_factor']

    if inputname[-4:] == '.pcd':
        files = [inputname]
    else:
        files = [join(inputname, f) for f in listdir(inputname) if f[-4:] == '.pcd']

    xyznormal_folder = join(outputname, 'xyz_normal')
    iterative_mkdir(xyznormal_folder.split('/'))     
    xyz_folder = join(outputname, 'xyz')
    iterative_mkdir(xyz_folder.split('/'))
    
    for f in tqdm(files):
        pc = pypcd.PointCloud.from_path(f)
        #array = pc.pc_data.view(np.float32).reshape(pc.pc_data.shape + (-1,))[:,:6]
        array = np.zeros((pc.pc_data.shape[0], 6), dtype = float)
        array[:,0] = pc.pc_data['x']
        array[:,1] = pc.pc_data['y']
        array[:,2] = pc.pc_data['z']
        array[:,3] = pc.pc_data['normal_x']
        array[:,4] = pc.pc_data['normal_y']
        array[:,5] = pc.pc_data['normal_z']
        if rescale_factor != 1:
            array = rescale(array, rescale_factor)
        if noise_limit != 0.0:
            array = add_noise(array, noise_limit)
        if is_centralize == True:
            array = centralize(array)
        if is_align == True:
            array = align_canonical(array)
        if cube_rescale_factor != 0:
            array = cube_rescale(array, cube_rescale_factor)
    
        filename = f.split('/')[-1][:-4] + '.xyz'
        np.savetxt(join(xyznormal_folder, filename), array)
        np.savetxt(join(xyz_folder, filename), array[:, :3])