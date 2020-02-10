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

import utils
import fcn_dataset
from siamese_fcn import siamese_fcn

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def benchmark(output_dir, model, data_loader, config, cuda=False, use_amp=False):
    """Benchmarks a model."""

    model.eval()

    if config['vis']:
        out = osp.join(output_dir, 'visualization_viz')
        utils.mkdir_if_missing(out)

    benchmark_loss = 0
    label_trues, label_preds = [], []
    for batch_idx, data in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader),
            desc='Benchmark Progress', ncols=80,
            leave=False):
        if cuda:
            data = tuple([d.cuda() for d in data])
        data = tuple([Variable(d) for d in data])
        lbl = data[-1]
        with torch.no_grad():
            score = model(*data[:-1])
        score = score['out'].squeeze()
        
        loss = torch.nn.MSELoss()(score, lbl)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while benchmarking')
        benchmark_loss += loss_data / len(score)
        
        data = tuple([d.data.cpu() for d in data])
        lbl_pred = score.data.cpu().numpy()
        for d in zip(*data, lbl_pred):
            dd = data_loader.dataset.untransform(*d[:-1])
            label_trues.append(dd[-1])
            label_preds.append(d[-1])
        
        if config['vis'] and batch_idx % config['vis_interval'] == 0:
            if len(dd) > 2:
                viz = utils.visualize_segmentation(lbl_pred=d[-1], lbl_true=dd[-1], img=dd[0], targ=dd[1])     
            else:
                viz = utils.visualize_segmentation(lbl_pred=d[-1], lbl_true=dd[-1], img=dd[0])        
            out_file = osp.join(out, 'vis_sample_{:03d}.jpg'.format(int(batch_idx / config['vis_interval'])))
            skimage.io.imsave(out_file, viz)

    metrics = utils.label_accuracy_score(label_trues, label_preds)
    benchmark_loss /= len(data_loader)
    
    t = PrettyTable(['#Img', 'Loss', 'Acc', 'Bal-Acc', 'IoU'])
    t.add_row([len(data_loader)] + [benchmark_loss] + list(metrics))
    print(t)


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark FCN model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark_fcn.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    print("Benchmarking model.")

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
    siamese = (config['model']['type'] == 'siamese_fcn')
    model = siamese_fcn() if siamese else fcn_resnet50(num_classes=1)
    checkpoint = torch.load(osp.join(config['model']['path'], 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model = model.cuda()

    # 2. dataset
    root = config['dataset']['path']
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    test_set = fcn_dataset.FCNTargetDataset(root, split='test', imgs=config['dataset']['imgs'], targs=config['dataset']['targs'], lbls=config['dataset']['lbls'], transform=True) if siamese else fcn_dataset.FCNDataset(root, split='test', imgs=config['dataset']['imgs'], lbls=config['dataset']['lbls'], transform=True)
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=config['model']['batch_size'], shuffle=True, **kwargs)

    # If using mixed precision training, initialize here
    if APEX_AVAILABLE:
        # amp.load_state_dict(checkpoint['amp'])
        model = amp.initialize(
            model, opt_level="O3", 
            keep_batchnorm_fp32=True
        )

    benchmark(output_dir, model, data_loader, config, cuda=cuda, use_amp=APEX_AVAILABLE)
