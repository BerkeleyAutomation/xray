import collections
import os.path as osp
import numpy as np
from skimage.io import imread
import skimage.morphology as skm
import torch
from torch.utils import data

class FCNDataset(data.Dataset):
    class_names = np.array([
            'background',
            'obj',
        ])
    
    mean_bgr = np.array([189.0995733, 189.0995733, 189.30626962])

    def __init__(self, root, split='train', soft=False, transform=False):
        self.root = root
        self.split = split
        self._soft = soft
        self._transform = transform

        self.files = collections.defaultdict(list)
        inds = np.load(osp.join(root, '{}_indices.npy'.format(split)))
        for i in inds:
            img_file = osp.join(root, 'color_ims', 'image_{:06d}.png'.format(i))
            if self._soft:
                lbl_file = osp.join(root, 'soft_dist_ims', 'image_{:06d}.png'.format(i))
            else:
                lbl_file = osp.join(root, 'dist_ims', 'image_{:06d}.png'.format(i))
            self.files[split].append({
                'img': img_file,
                'lbl': lbl_file,
            })
        

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        
        # load image
        img_file = data_file['img']
        img = imread(img_file)
        
        # load label
        lbl_file = data_file['lbl']
        lbl = imread(lbl_file)
        if not self._soft:
            lbl = skm.binary_dilation(lbl, selem=np.ones((11,11)))
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
