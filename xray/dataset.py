import collections
import os.path as osp

import numpy as np
import tqdm
from skimage.io import imread
from torch.utils import data

from . import utils


class XrayDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        imgs,
        lbls,
        ratios,
        mean_bgr,
        max_ind=0,
    ):
        self.root = root
        self.split = split
        self.mean_bgr = np.array(mean_bgr)

        self.files = collections.defaultdict(list)
        inds = np.load(osp.join(root, f"{split}_indices.npy"))
        if max_ind:
            inds = inds[inds < max_ind]
        for i in tqdm.tqdm(inds, desc="Processing Dataset"):
            for r in ratios:
                im_path = osp.join(root, imgs, "image_{:06d}_{:02d}.png".format(i, r))
                if osp.exists(im_path):
                    self.files[split].append(
                        {
                            "img": im_path,
                            "lbl": osp.join(root, lbls, "image_{:06d}_{:02d}.png".format(i, r)),
                            "ratio": ratios.index(r),
                        }
                    )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        # load image
        img_file = data_file["img"]
        img = imread(img_file)
        img = np.array(img)
        img = utils.transform(img, self.mean_bgr)

        # load label
        lbl_file = data_file["lbl"]
        lbl = imread(lbl_file)
        lbl = np.array(lbl)

        ratio = data_file["ratio"]
        return img, lbl, ratio
