from skimage.io import imread
import numpy as np
import os
import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="find mean of images")
    parser.add_argument("directory", type=str, help="directory with images")
    args = parser.parse_args()

    means = []
    inds = np.load(os.path.join(args.directory, "..", "train_indices.npy"))
    for f in tqdm.tqdm(os.listdir(args.directory)):
        fn = os.path.join(args.directory, f)
        if int(os.path.splitext(fn)[0].split("_")[-1]) in inds:
            means.append(imread(fn).mean(axis=(0,1)))

    print(np.mean(means, axis=0))
