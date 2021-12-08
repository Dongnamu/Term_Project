import numpy as np
from one_euro_filter import OneEuroFilter
import os
from natsort import natsorted
from multiprocessing import Pool
import math
import cv2
import torch
from triangles import mouthPoints, chins, rest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--fps', type=int, default=25)
args = parser.parse_args()

image_height = args.image_size
image_width = args.image_size
fps = args.fps

def applyFilter(points, t, min_cutoff, beta, skipPoints = []):
    filtered = np.empty_like(points)
    filtered[0] = points[0]
    one_euro_filter = OneEuroFilter(t[0], points[0], min_cutoff, beta)
    
    for i in range(1, points.shape[0]):
        filtered[i] = one_euro_filter(t[i], points[i])
        
    for i in range(1, points.shape[0]):
        for skipPoint in skipPoints:
            filtered[i, skipPoint] = points[i, skipPoint]

    return filtered

def removeJitter(data_dir):
    normalised_mesh_files = natsorted([os.path.join(data_dir, 'mesh_dict', x) for x in os.listdir(os.path.join(data_dir, 'mesh_dict'))])

    landmarks = []

    for file in normalised_mesh_files:
        landmark = torch.load(file)
        R = landmark['R']
        t = landmark['t']
        c = landmark['c']
        keys = natsorted([x for x in landmark.keys() if type(x) is int])
        vertices = []
        for key in keys:
            vertice = np.array(landmark[key]).reshape(3,1)
            norm_vertice = (c * np.matmul(R, vertice) + t).squeeze()
            x_px = min(math.floor(norm_vertice[0]), image_width - 1)
            y_px = min(math.floor(norm_vertice[1]), image_height - 1)
            z_px = min(math.floor(norm_vertice[2]), image_width - 1)
            vertices.append([x_px, y_px, z_px])
            # vertices.append(norm_vertice)
        landmarks.append(vertices)
        
    landmarks = np.array(landmarks)

    shape_1, shape_2, shape_3 = landmarks.shape

    xs = landmarks[:,:,0].reshape((shape_1, shape_2))
    ys = landmarks[:,:,1].reshape((shape_1, shape_2))
    zs = landmarks[:,:,2].reshape((shape_1, shape_2))

    t = np.linspace(0, xs.shape[0]/fps, xs.shape[0])


    xs_hat = applyFilter(xs, t, 0.005, 0.7)
    ys_hat = applyFilter(ys, t, 0.005, 0.7, mouthPoints + chins)
    ys_hat = applyFilter(ys_hat, t, 0.000001, 1.5, rest)
    zs_hat = applyFilter(zs, t, 0.005, 0.7)
    combine = np.stack(((xs_hat, ys_hat, zs_hat)), axis=2)

    count = [i for i in range(combine.shape[0])]

    os.makedirs(os.path.join(data_dir, 'stabilized_norm_mesh'),exist_ok=True)
    for i in range(combine.shape[0]):
        torch.save(combine[i], os.path.join(data_dir, 'stabilized_norm_mesh', '{}.pt'.format(count[i])))


if __name__ == '__main__':
    data_dirs = args.data_dir

    videos = [x for x in natsorted(os.listdir(data_dirs)) if os.path.isdir(os.path.join(data_dirs, x))]
    valid_videos = [os.path.join(data_dirs, video) for video in videos if 'failed.txt' not in os.listdir(os.path.join(data_dirs, video))]
    
    # print(valid_videos[0])
    pool = Pool(processes=40)
    pool.map(removeJitter, valid_videos)
    pool.terminate()
    pool.join()
    
    

    



