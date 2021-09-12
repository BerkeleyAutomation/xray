import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse
import numpy as np
import skimage.io
import torch
from torchvision.models.segmentation import fcn_resnet50
from autolab_core import YamlConfig, Logger
import cv2


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
    conf_parser.add_argument("--config", action="store", default="cfg/test_real_image.yaml",
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
    ratios = config["ratios"]
    model = fcn_resnet50(num_classes=len(ratios) if ratios is not None else 1)
    checkpoint = torch.load(osp.join(config['model']['path'], 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model = model.cuda()

    # 2. files
    depth_image_fns = sorted([osp.join(config["dataset"], f) for f in os.listdir(config["dataset"]) if "Depth" in f])
    color_image_fns = sorted([osp.join(config["dataset"], f) for f in os.listdir(config["dataset"]) if "Color" in f])
    results_folder = osp.join(config["dataset"], "results")
    if not osp.exists(results_folder):
        os.makedirs(results_folder)

    model.eval()
    for c_fn, d_fn in zip(color_image_fns, depth_image_fns):
        # Read frame from camera and convert
        color_im = skimage.io.imread(c_fn)
        depth_im = skimage.io.imread(d_fn)
        color_im = cv2.resize(color_im, (640, 480))
        depth_im = cv2.resize(depth_im, (640, 480))

        img = depth_im.astype(np.float64)
        img -= np.array(config["mean_bgr"])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img[None,...]).float()        
        # img = cv2_clipped_zoom(img, 1 / config['scale'])

        # np.savez('test_ds/test_{:03d}'.format(i), img.numpy())
        if cuda:
            img = img.cuda()
        with torch.no_grad():
            score = model(img)
        score = score['out'].squeeze()
        score /= score.max()
        # score = cv2_clipped_zoom(score.cpu()[None,...], config['scale'])
        # img = cv2_clipped_zoom(img.cpu(), config['scale'])

        plt.figure(frameon=False)
        # plt.subplot(2,2,1)
        plt.imshow(depth_im)
        # plt.axis('off')
        # plt.subplot(2,2,2)
        plt.imshow(score[0].squeeze().cpu().numpy(), cmap='jet', alpha=0.5)
        plt.axis('off')
        # plt.subplot(2,2,3)
        #plt.imshow(color_im)
        #plt.axis('off')
        #plt.subplot(2,2,4)
        #plt.imshow(color_im)
        #plt.imshow(score.squeeze().numpy(), cmap='jet', alpha=0.5)
        #plt.axis('off')
        # plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.02, hspace=0.02)
        plt.savefig(osp.join(results_folder, f"result_{osp.basename(osp.splitext(c_fn)[0])}.png"))
        plt.figure(frameon=False)
        plt.imshow(depth_im)
        plt.bar(np.arange(640), -480 * score[0].cpu().numpy().sum(axis=0) / score[0].cpu().numpy().sum(axis=0).max(), width=1.0, alpha=0.5, bottom=480)
        plt.axis('off')
        plt.savefig(osp.join(results_folder, f"result1d_{osp.basename(osp.splitext(c_fn)[0])}.png"))
