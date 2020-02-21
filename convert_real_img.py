from perception import WebcamSensor
import matplotlib.pyplot as plt

import datetime
import math
import os
import os.path as osp
import shutil
import argparse
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import tqdm
from torchvision.models.segmentation import fcn_resnet50
from autolab_core import YamlConfig
from prettytable import PrettyTable
import cv2

import utils
import fcn_dataset
from siamese_fcn import siamese_fcn

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    if zoom_factor == 1:
        return img

    height, width = img.shape[-2:] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img.numpy()[..., y1:y2, x1:x2]
    cropped_img = cropped_img.reshape((-1, *cropped_img.shape[-2:])).transpose(1,2,0)

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2), (0,0)]

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec[:result.ndim], mode='edge')
    assert result.shape[0] == height and result.shape[1] == width
    if result.ndim == 3:
        result = result.transpose(2,0,1)
    return torch.from_numpy(result.reshape(img.shape)).float()

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark FCN model on real images")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark_real_fcn.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    print("Benchmarking model")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['model']['gpu'])
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.benchmark = True

    # 1. model
    siamese = ('siamese' in config['model']['type'])
    unet = ('unet' in config['model']['type'])
    if siamese and unet:
        model = SiameseUNet(3, 1)
    elif siamese:
        model = siamese_fcn()
    else:
        model = fcn_resnet50(num_classes=1)
    checkpoint = torch.load(osp.join(config['model']['path'], 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model = model.cuda()

    # 2. camera
    cam = WebcamSensor(device_id=config['device_id'])
    cam.start()

    # If using mixed precision training, initialize here
    if APEX_AVAILABLE:
        # amp.load_state_dict(checkpoint['amp'])
        model = amp.initialize(
            model, opt_level="O3", 
            keep_batchnorm_fp32=True
        )
    model.eval()

    while True:
        # Read frame from camera and convert
        frame = cam.frames()[0]
        gray_frame = np.repeat(frame.to_grayscale().data[...,None], 3, axis=-1)
        red_mask = np.logical_and(frame.data[:,:,0] > 250, np.logical_and(frame.data[:,:,1] < 150, frame.data[:,:,2] < 150))
        gray_frame[red_mask, :] = frame.data[red_mask, :]

        img = cv2.resize(gray_frame, (512, 384))
        mean_rgb = img.mean(axis=(0,1))

        img = img.astype(np.float64)
        img -= mean_rgb
        img = img[:, :, ::-1].copy()  # RGB -> BGR
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img[None,...]).float()
        img = cv2_clipped_zoom(img, 1 / config['scale'])
        if cuda:
            img = img.cuda()
        with torch.no_grad():
            score = model(img)
        score = score['out'].squeeze()
        score = cv2_clipped_zoom(score.cpu()[None,...], config['scale'])
        img = cv2_clipped_zoom(img.cpu(), config['scale'])

        img = img.numpy().squeeze()
        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1]
        img += mean_rgb
        img = img.astype(np.uint8)

        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(frame.data)
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(score.squeeze().numpy(), cmap='jet')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(img)
        plt.imshow(score.squeeze().numpy(), cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()
