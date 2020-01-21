from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os
from perception import ColorImage

path = '/nfs/diskstation/projects/mech_search/prob_pose/datasets/multi_target_dataset/images/target_ims'

for f in os.listdir(path):
    x = imread(os.path.join(path, f))
    mask = ~x[:,:,1].astype(np.bool)
    mean_px = np.mean(np.nonzero(mask), axis=1).astype(np.int)
    x = np.pad(x, ((64,), (64,), (0,)), 'constant', constant_values=np.iinfo('uint8').max)
    crop = x[mean_px[0]:mean_px[0]+128, mean_px[1]:mean_px[1]+128, :]
    imsave(os.path.join(path, '..', 'target_ims_test', f), crop)
