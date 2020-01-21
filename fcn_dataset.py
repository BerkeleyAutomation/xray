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

    def __init__(self, root, split='train', imgs=['color_ims'], lbls='soft_dist_ims', transform=False, max_inds=0):
        self.root = root
        self.split = split
        self._transform = transform
        self._soft = ('soft' in lbls)

        self.files = collections.defaultdict(list)
        inds = np.load(osp.join(root, '{}_indices.npy'.format(split)))
        if max_inds:
            inds = inds[:max_inds]
        for i in inds:
            img_files = [osp.join(root, img_folder, 'image_{:06d}.png'.format(i)) for img_folder in imgs]
            lbl_file = osp.join(root, lbls, 'image_{:06d}.png'.format(i))
            self.files[split].append({
                'img': img_files,
                'lbl': lbl_file,
            })
        

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        
        # load image
        img_files = data_file['img']
        img = np.concatenate([imread(img_file) for img_file in img_files], axis=-1)
    
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
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img[:, :, :3] -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img[:, :, :3] += self.mean_bgr
        img = img.astype(np.uint8)
        img[:, :, :3] = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
