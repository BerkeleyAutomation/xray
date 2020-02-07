import collections
import os.path as osp
import numpy as np
from skimage.io import imread
import torch
from torch.utils import data


class FCNDataset(data.Dataset):
    class_names = np.array([
            'background',
            'obj',
        ])
    
    mean_bgr = np.array([189.0995733, 189.0995733, 189.30626962])

    def __init__(self, root, split='train', 
                 imgs='color_ims', lbls='soft_dist_ims', 
                 transform=False, max_inds=0):
        self.root = root
        self.split = split
        self._transform = transform
        self._soft = ('soft' in lbls)

        self.files = collections.defaultdict(list)
        inds = np.load(osp.join(root, '{}_indices.npy'.format(split)))
        if max_inds:
            inds = inds[:max_inds]
        for i in inds:
            img_file = osp.join(root, imgs, 'image_{:06d}.png'.format(i))
            lbl_file = osp.join(root, lbls, 'image_{:06d}.png'.format(i))
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
        if self._transform:
            img, lbl = self.transform(img, lbl)
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


class FCNTargetDataset(FCNDataset):

    def __init__(self, root, split='train', 
                 imgs='color_ims', targs='target_ims', lbls='soft_dist_ims', 
                 transform=False, max_inds=0):
        self.root = root
        self.split = split
        self._transform = transform
        self._soft = ('soft' in lbls)

        self.files = collections.defaultdict(list)
        inds = np.load(osp.join(root, '{}_indices.npy'.format(split)))
        if max_inds:
            inds = inds[:max_inds]
        for i in inds:
            img_file = osp.join(root, imgs, 'image_{:06d}.png'.format(i))
            targ_file = osp.join(root, targs, 'image_{:06d}.png'.format(i))
            lbl_file = osp.join(root, lbls, 'image_{:06d}.png'.format(i))
            self.files[split].append({
                'img': img_file,
                'targ': targ_file,
                'lbl': lbl_file,
            })
        

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        
        # load image and label
        img, lbl = super().__getitem__(index)

        # load target
        targ_file = data_file['targ']
        targ = imread(targ_file)
        
        if self._transform:
            img, targ, label = self.transform(img, targ, lbl)
        return img, targ, lbl

    def transform(self, img, targ, lbl):
        img, lbl = super().transform(img, lbl)
        targ = targ.transpose(2, 0, 1)
        targ = torch.from_numpy(targ.astype(np.float64)).float()
        return img, targ, lbl

    def untransform(self, img, targ, lbl):
        img, lbl = super().untransform(img, lbl)
        targ = targ.numpy()
        targ = targ.transpose(1, 2, 0).astype(np.uint8)
        return img, targ, lbl
