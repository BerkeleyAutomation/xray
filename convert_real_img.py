from phoxipy import ColorizedPhoXiSensor
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
from autolab_core import YamlConfig, Logger
from autolab_core.utils import keyboard_input
from prettytable import PrettyTable
import cv2
from perception import ColorImage

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

    # set up logger
    logger = Logger.get_logger(__file__)

    logger.info("Benchmarking model")

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
    cam = ColorizedPhoXiSensor(config['phoxi_name'], 
                               config['device_id'], 
                               '/nfs/diskstation/calib/')
    cam.start()
    
    # If using mixed precision training, initialize here
    if APEX_AVAILABLE:
        model = amp.initialize(
            model, opt_level="O1" 
        )
        amp.load_state_dict(checkpoint['amp'])
    model.eval()

    response = keyboard_input('Ready to take images?', yesno=True)
    while response.lower() != 'y':
        response = keyboard_input('Ready to take images?', yesno=True)

    while True:
        # Read frame from camera and convert
        color_im, depth_im, _ = cam.read()
        depth_im = depth_im.crop(512, 683, center_i=356, center_j=542).inpaint()
        color_im = color_im.crop(512, 683, center_i=356, center_j=542).inpaint()
        color_im = cv2.resize(color_im.data, (512, 384))
        depth_im = cv2.resize(depth_im.data, (512, 384))

        # Create combo image from depth image and opencv red detector
        color_hsv = cv2.cvtColor(np.flip(color_im, axis=-1), cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(color_hsv,  np.array([0,120,70]), np.array([10,255,255]))
        mask2 = cv2.inRange(color_hsv, np.array([170,120,70]), np.array([180,255,255]))
        color_mask = (mask1 + mask2).astype(np.bool)
        combo_im = np.iinfo(np.uint8).max * (depth_im - 0.25) / (1.0 - 0.25)
        combo_im = np.repeat(combo_im[...,None], 3, axis=-1).astype(np.uint8)
        combo_im[color_mask, 0] = 255
        combo_im[~color_mask, 0] = 0

        mean_bgr = np.array([0.0, 181.0, 181.0])
        img = combo_im[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img[None,...]).float()        
        img = cv2_clipped_zoom(img, 1 / config['scale'])

        # np.savez('test_ds/test_{:03d}'.format(i), img.numpy())
        if cuda:
            img = img.cuda()
        with torch.no_grad():
            score = model(img)
        score = score['out'].squeeze()
        score = cv2_clipped_zoom(score.cpu()[None,...], config['scale'])
        img = cv2_clipped_zoom(img.cpu(), config['scale'])

        img = img.numpy().squeeze()
        img = img.transpose(1, 2, 0)
        img += mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]

        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(score.squeeze().numpy(), cmap='jet')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(color_im)
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(color_im)
        plt.imshow(score.squeeze().numpy(), cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.02, hspace=0.02)
        plt.show()
