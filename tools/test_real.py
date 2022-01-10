import argparse
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from autolab_core import Logger, YamlConfig
from autolab_core.utils import keyboard_input
from phoxipy import ColorizedPhoXiSensor

from xray import XrayModel, utils

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Test model on real images")
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/test_real.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    # set up logger
    logger = Logger.get_logger(__file__)
    logger.info("Testing model")

    # 1. model
    model_cfg = YamlConfig(osp.join(config["model"]["path"], f"{osp.split(config['model']['path'])[-1]}.yaml"))
    device = torch.device(config["device"])
    model = XrayModel(ratios=model_cfg["model"]["ratios"]).to(device)
    checkpoint = torch.load(osp.join(config["model"]["path"], "model_best.pth.tar"))
    model.load(checkpoint["model_state_dict"])
    model.eval()

    # 2. camera
    cam = ColorizedPhoXiSensor(config["phoxi"]["name"], config["phoxi"]["device_id"], "/nfs/diskstation/calib/")
    cam.start()

    response = keyboard_input("Ready to take images?", yesno=True)
    while response.lower() != "y":
        response = keyboard_input("Ready to take images?", yesno=True)

    while True:
        # Read frame from camera and convert
        color_im, depth_im, _ = cam.read()
        depth_im = depth_im.crop(512, 683, center_i=356, center_j=542).inpaint()
        color_im = color_im.crop(512, 683, center_i=356, center_j=542).inpaint()
        color_im = cv2.resize(color_im.data, (512, 384))
        depth_im = cv2.resize(depth_im.data, (512, 384))

        # Create combo image from depth image and opencv red detector
        color_hsv = cv2.cvtColor(np.flip(color_im, axis=-1), cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(color_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(color_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        color_mask = (mask1 + mask2).astype(np.bool)
        combo_im = np.iinfo(np.uint8).max * (depth_im - 0.25) / (1.0 - 0.25)
        combo_im = np.repeat(combo_im[..., None], 3, axis=-1).astype(np.uint8)
        combo_im[color_mask, 0] = 255
        combo_im[~color_mask, 0] = 0

        # Transform image
        img = utils.transform(combo_im, np.array(model_cfg["dataset"]["mean"]))
        img = torch.from_numpy(img[None, ...])
        img = utils.cv2_clipped_zoom(img, 1 / config["scale"])
        img = img.to(device)

        with torch.no_grad():
            score = model(img.float())["out"]
        score = utils.cv2_clipped_zoom(score.cpu(), config["scale"])

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(combo_im)
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(score.squeeze().numpy(), cmap="jet")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(color_im)
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.imshow(color_im)
        plt.imshow(score.squeeze().numpy(), cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.02, hspace=0.02)
        plt.show()
