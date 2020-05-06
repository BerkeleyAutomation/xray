"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Author: Mike Danielczuk

Benchmark Usage Notes:

Please edit "cfg/benchmark.yaml" to specify the necessary parameters for that task.

Run this file with the tag --config [config file name] if different config from the default location (cfg/benchmark.yaml).

Here is an example run command (GPU selection included):
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py
"""

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
import matplotlib.pyplot as plt

from xray import utils
from xray import FCNRatioDataset

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def cv2_blur(img, kernel_size=5):
    result = img.numpy().reshape((-1, *img.shape[-2:])).transpose(1,2,0)
    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    return torch.from_numpy(result.transpose(2,0,1).reshape(img.shape)).float()

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
    result = np.pad(result, pad_spec, mode='edge')
    assert result.shape[0] == height and result.shape[1] == width
    return torch.from_numpy(result.transpose(2,0,1).reshape(img.shape)).float()

def benchmark(output_dir, model, data_loader, config, cuda=False, use_amp=False):
    """Benchmarks a model."""

    model.eval()

    if config['vis']:
        out = osp.join(output_dir, 'viz')
        utils.mkdir_if_missing(out)

    benchmark_loss = 0
    label_trues, label_preds = [], []
    for batch_idx, data in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader),
            desc='Benchmark Progress', ncols=80,
            leave=False):

        data[0] = cv2_clipped_zoom(data[0], 1 / config['scale'])
        if cuda:
            data = [d.cuda() for d in data]
        data = [Variable(d) for d in data]
        imgs, lbls, ratios = data
        with torch.no_grad():
            score = model(imgs)
        if isinstance(score, dict):
            score = score['out']
        score = cv2_clipped_zoom(score.cpu(), config['scale']).cuda()
        data[0] = cv2_clipped_zoom(data[0].cpu(), config['scale']).cuda()
        loss_score = score[range(len(imgs)), ratios]

        loss = torch.nn.MSELoss()(loss_score, lbls)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while benchmarking')
        benchmark_loss += loss_data / len(loss_score)
        
        data = [d.data.cpu() for d in data]
        lbl_pred = loss_score.data.cpu().numpy()
        all_lbls_pred = score.data.cpu().numpy()
        for img, lbl, pred in zip(data[0], data[1], lbl_pred):
            img, lbl = data_loader.dataset.untransform(img, lbl)
            label_trues.append(lbl)
            label_preds.append(pred)
        
        if config['vis'] and batch_idx % config['vis_interval'] == 0:
            out_file = osp.join(out, 'vis_{{}}_{num:03d}.png'.format(num=int(batch_idx / config['vis_interval'])))
            gt = np.iinfo(np.uint8).max * plt.cm.jet(lbl.astype(np.uint8))
            all_pred = np.iinfo(np.uint8).max * plt.cm.jet(all_lbls_pred[-1] / all_lbls_pred[-1].max(axis=(1,2), keepdims=True))

            if 'vis_block' not in config.keys() or config['vis_block']:
                viz = utils.visualize_segmentation(lbl_pred=all_lbls_pred[-1], lbl_true=lbl, img=img)        
            else:
                skimage.io.imsave(out_file.format('input'), img)
                skimage.io.imsave(out_file.format('gt'), gt.astype(np.uint8))
                for j, ap in enumerate(all_pred):
                    skimage.io.imsave(out_file.format('pred_{:02d}'.format(config['model']['ratios'][j])), ap.astype(np.uint8))
            if 'vis_block' not in config.keys() or config['vis_block']:
                skimage.io.imsave(out_file.format('block'), viz)

    metrics = utils.label_accuracy_score(label_trues, label_preds)
    benchmark_loss /= len(data_loader)
    
    t = PrettyTable(['N', 'Loss', 'Acc', 'Bal-Acc', 'IoU'])
    t.add_row([data_loader.batch_size * len(data_loader)] + [np.round(benchmark_loss, 1)] + list(np.round(metrics, 2)))
    res = {k: v if isinstance(v, int) else float(v) for (k,v) in zip(t._field_names, t._rows[0])}
    results = YamlConfig()
    results.update(res)
    results.save(os.path.join(output_dir, 'results.yaml'))
    print(t)


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark FCN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark_fcn.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    print("Benchmarking model")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['model']['gpu'])
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.benchmark = True

    # 1. model
    ratios = config['model']['ratios']
    ratio_map = {k:v for k,v in zip(ratios, range(len(ratios)))}
    model = fcn_resnet50(num_classes=len(ratios))
    checkpoint = torch.load(osp.join(config['model']['path'], 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model = model.cuda()

    # 2. dataset
    root = config['dataset']['path']
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    test_set = FCNRatioDataset(root, split='test', imgs=config['dataset']['imgs'], lbls=config['dataset']['lbls'], 
                               mean=config['dataset']['mean'], ratio_map=ratio_map, transform=True)
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=config['model']['batch_size'], shuffle=False, **kwargs)

    # If using mixed precision training, initialize here
    if APEX_AVAILABLE and False:
        model = amp.initialize(
            model, opt_level="O1")
        amp.load_state_dict(checkpoint['amp'])

    benchmark(output_dir, model, data_loader, config, cuda=cuda, use_amp=APEX_AVAILABLE and False)
