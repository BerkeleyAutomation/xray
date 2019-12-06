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

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def benchmark(output_dir, model, data_loader, config, cuda=False, use_amp=False):
    """Benchmarks a model."""

    model.eval()

    benchmark_loss = 0
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader),
            desc='Benchmark Progress', ncols=80,
            leave=False):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        with torch.no_grad():
            score = model(data)
        score = score['out'].squeeze()
        
        loss = torch.nn.MSELoss()(score, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while benchmarking')
        benchmark_loss += loss_data / len(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.cpu().numpy()
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = data_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img)
                visualizations.append(viz)

    metrics = utils.label_accuracy_score(label_trues, label_preds)

    out = osp.join(output_dir, 'visualization_viz')
    if not osp.exists(out):
        os.makedirs(out)
    out_file = osp.join(out, 'vis_sample.jpg')
    skimage.io.imsave(out_file, utils.get_tile_image(visualizations))

    benchmark_loss /= len(data_loader)
    
    t = PrettyTable(['#Img', 'Loss', 'Acc', 'Bal-Acc', 'IoU'])
    t.add_row([len(data_loader)*config['model']['batch_size']] + [benchmark_loss] + list(metrics))
    print(t)

    ######## BENCHMARK JUST CREATES THE RUN DIRECTORY ########
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    # Create predictions and record where everything gets stored.
    # pred_info_dir = detect(config['output_dir'], inference_config, model, test_dataset)

    # ap, ap50, ap75, ar = calculate_metrics(test_dataset, inference_config, pred_info_dir, display=True)

    # pred_config = config['vis']['pred']
    # gt_config = config['vis']['gt']
    # diff_config = config['vis']['diffs']
    # if pred_config['boxes'] or pred_config['dists']:
    #     visualize_predictions(config['output_dir'], test_dataset, inference_config, pred_info_dir, show_dists=pred_config['dists'],
    #                           show_boxes=pred_config['boxes'], show_scores=pred_config['scores'], show_class=pred_config['classes'])
    # if gt_config['boxes'] or gt_config['dists']:
    #     visualize_gts(config['output_dir'], test_dataset, inference_config, show_dists=gt_config['dists'],
    #                   show_boxes=gt_config['boxes'], show_scores=gt_config['scores'], show_class=gt_config['classes'])
    
    # if diff_config['boxes'] or diff_config['dists']:
    #     visualize_differences(config['output_dir'], test_dataset, inference_config, pred_info_dir, show_dists=diff_config['dists'],
    #                           show_boxes=diff_config['boxes'], show_scores=diff_config['scores'], show_class=diff_config['classes'])
    
    # print("Saved benchmarking output to {}.\n".format(config['output_dir']))
    # return ap, ar


def calculate_metrics(dataset, inference_config, pred_info_dir, display=True):
    """Calculates AP and AR from predictions."""

    total_image_ids = len(dataset.image_ids)
    mAPrange = 0
    mAR = 0
    mAP50 = 0
    mAP75 = 0
    dist_metric = 0.0
    
    for image_id in tqdm(dataset.image_ids):
        
        # Load image and ground truth data
        image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, inference_config, image_id)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load info and calculate AP/AR
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id)), allow_pickle=True).item()
        mAPrange += utils.compute_ap_range(gt_bbox, gt_class_id, r['rois'], r['class_ids'], r['scores'], verbose=False)
        mAP50 += utils.compute_ap(gt_bbox, gt_class_id, r['rois'], r['class_ids'], r['scores'])[0]
        mAP75 += utils.compute_ap(gt_bbox, gt_class_id, r['rois'], r['class_ids'], r['scores'], iou_threshold=0.75)[0]
        AR = 0.0
        for iou in np.linspace(0.5, 0.95, 10):
            AR += utils.compute_recall(r['rois'], gt_bbox, iou)[0] / 10.0
        mAR += AR

        gt_y_centers = np.mean(gt_bbox[:,::2], axis=1, dtype=np.int)
        gt_x_centers = np.mean(gt_bbox[:,1::2], axis=1, dtype=np.int)
        gt_dist = np.zeros_like(image[:,:,0], dtype=np.bool)
        gt_dist[gt_y_centers, gt_x_centers] = True
        gt_dist = skm.binary_dilation(gt_dist, selem=np.ones((8,8))).astype(np.uint8)

        pred_y_centers = np.mean(r['rois'][:,::2], axis=1, dtype=np.int)
        pred_x_centers = np.mean(r['rois'][:,1::2], axis=1, dtype=np.int)
        pred_dist = np.zeros_like(image[:,:,0], dtype=np.bool)
        pred_dist[pred_y_centers, pred_x_centers] = True
        pred_dist = 2 * (skm.binary_dilation(pred_dist, selem=np.ones((8,8))).astype(np.uint8))

        both_dist = pred_dist + gt_dist
        intersection = np.sum(both_dist == 3)
        union = np.sum(both_dist > 0)
        dist_metric += float(intersection) / float(union)

    mAPrange /= total_image_ids
    mAP50 /= total_image_ids
    mAP75 /= total_image_ids
    mAR /= total_image_ids
    dist_metric /= total_image_ids
    
    if display:
        table = [['AP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AR', 'Dist IoU'],
                 ['{:.3f}'.format(mAPrange), '{:.3f}'.format(mAP50), '{:.3f}'.format(mAP75), 
                  '{:.3f}'.format(mAR), '{:.3f}'.format(dist_metric)]]
        visualize.display_table(table)
    
    return mAPrange, mAP50, mAP75, mAR

def visualize_predictions(run_dir, dataset, inference_config, pred_info_dir, 
                          show_dists=True, show_boxes=True, show_scores=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectories for visualizations
    if show_dists:
        dist_dir = os.path.join(run_dir, 'pred_dists')
        utils.mkdir_if_missing(dist_dir)
    if show_boxes:
        box_dir = os.path.join(run_dir, 'pred_boxes')
        utils.mkdir_if_missing(box_dir)

    # Feed images into model one by one. For each image visualize predictions
    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(dataset.image_ids):
        # Load image
        image, _, _, _ = modellib.load_image_gt(dataset, inference_config, image_id)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load predicted boxes
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id)), allow_pickle=True).item()

        if show_dists:
            y_centers = np.mean(r['rois'][:,::2], axis=1, dtype=np.int)
            x_centers = np.mean(r['rois'][:,1::2], axis=1, dtype=np.int)
            dist = np.zeros_like(image[:,:,0], dtype=np.bool)
            dist[y_centers, x_centers] = True
            dist = skm.binary_dilation(dist, selem=np.ones((8,8)))
            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            ax.imshow(dist, cmap=plt.cm.gray)
            file_name = os.path.join(dist_dir, 'dist_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close()

        if show_boxes:
            # Visualize boxes
            scores = r['scores'] if show_scores else None
            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            visualize.display_instances(image, ax, r['rois'], r['class_ids'], ['bg', 'obj'], 
                                        colors=['b']*len(r['class_ids']), scores=scores, show_class=show_class)
            file_name = os.path.join(box_dir, 'boxes_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close()

def visualize_gts(run_dir, dataset, inference_config, show_dists=True, 
                  show_boxes=True, show_scores=False, show_class=True):
    """Visualizes gts."""
    # Create subdirectories for gt visualizations
    if show_dists:
        dist_dir = os.path.join(run_dir, 'gt_dists')
        utils.mkdir_if_missing(dist_dir)
    if show_boxes:
        box_dir = os.path.join(run_dir, 'gt_boxes')
        utils.mkdir_if_missing(box_dir)

    print('VISUALIZING GROUND TRUTHS')
    for image_id in tqdm(dataset.image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, inference_config, image_id)

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        if show_dists:
            y_centers = np.mean(gt_bbox[:,::2], axis=1, dtype=np.int)
            x_centers = np.mean(gt_bbox[:,1::2], axis=1, dtype=np.int)
            dist = np.zeros_like(image[:,:,0], dtype=np.bool)
            dist[y_centers, x_centers] = True
            dist = skm.binary_dilation(dist, selem=np.ones((8,8)))

            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            ax.imshow(dist, cmap=plt.cm.gray)
            file_name = os.path.join(dist_dir, 'dist_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close()

        if show_boxes:
            # Visualize boxes
            scores = np.ones(gt_class_id.size) if show_scores else None
            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            visualize.display_instances(image, ax, gt_bbox, gt_class_id, ['bg', 'obj'], 
                                        colors=['g']*len(gt_class_id), scores=scores, show_class=show_class)
            file_name = os.path.join(box_dir, 'boxes_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close()     


def visualize_differences(run_dir, dataset, inference_config, pred_info_dir, show_dists=True,
                          show_boxes=True, show_scores=True, show_class=True):
    """Visualizes differences in predictions and ground truth."""
    # Create subdirectories for visualizations
    if show_dists:
        dist_dir = os.path.join(run_dir, 'diff_dists')
        utils.mkdir_if_missing(dist_dir)
    if show_boxes:
        box_dir = os.path.join(run_dir, 'diff_boxes')
        utils.mkdir_if_missing(box_dir)

    print('VISUALIZING DIFFERENCES')
    for image_id in tqdm(dataset.image_ids):
        
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, inference_config, image_id)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id)), allow_pickle=True).item()

        if show_dists:
            gt_y_centers = np.mean(gt_bbox[:,::2], axis=1, dtype=np.int)
            gt_x_centers = np.mean(gt_bbox[:,1::2], axis=1, dtype=np.int)
            gt_dist = np.zeros_like(image[:,:,0], dtype=np.bool)
            gt_dist[gt_y_centers, gt_x_centers] = True
            gt_dist = skm.binary_dilation(gt_dist, selem=np.ones((8,8))).astype(np.uint8)

            pred_y_centers = np.mean(r['rois'][:,::2], axis=1, dtype=np.int)
            pred_x_centers = np.mean(r['rois'][:,1::2], axis=1, dtype=np.int)
            pred_dist = np.zeros_like(image[:,:,0], dtype=np.bool)
            pred_dist[pred_y_centers, pred_x_centers] = True
            pred_dist = 2 * (skm.binary_dilation(pred_dist, selem=np.ones((8,8))).astype(np.uint8))
 
            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            ax.imshow(gt_dist+pred_dist)
            file_name = os.path.join(dist_dir, 'dist_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close()

        if show_boxes:
            fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
            ax = plt.Axes(fig, [0.,0.,1.,1.])
            fig.add_axes(ax)
            visualize.display_differences(image, ax, gt_bbox, gt_class_id, r['rois'], 
                                          r['class_ids'], r['scores'], ['bg', 'obj'], 
                                          show_scores=False)

            file_name = os.path.join(box_dir, 'diff_vis_{:06d}'.format(image_id))
            fig.savefig(file_name, transparent=True, dpi=300)
            plt.close() 

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

    # 1. dataset
    root = config['dataset']['path']
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(
        fcn_dataset.FCNDataset(root, split='test', soft=config['dataset']['soft_masks'], 
                               transform=True, max_inds=config['dataset']['max_inds']),
            batch_size=config['model']['batch_size'], shuffle=False, **kwargs)

    # 2. model
    model = fcn_resnet50(num_classes=1)
    checkpoint = torch.load(osp.join(config['model']['path'], 'checkpoint.pth.tar'))
    if cuda:
        model = model.cuda()

    # If using mixed precision training, initialize here
    if APEX_AVAILABLE:
        model = amp.initialize(
            model, opt_level="O3", 
            keep_batchnorm_fp32=True
        )

    benchmark(model, data_loader, config, cuda=cuda, use_amp=APEX_AVAILABLE)
