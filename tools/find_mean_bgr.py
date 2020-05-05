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
    for f in tqdm.tqdm(os.listdir(args.directory)):
        means.append(imread(os.path.join(args.directory, f)).mean(axis=(0,1)))

    print(np.mean(means, axis=0))
