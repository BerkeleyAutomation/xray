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
CUDA_VISIBLE_DEVICES=0 python tools/benchmark.py --config cfg/benchmark.yaml
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import skimage.io as io
from copy import copy
import matplotlib.pyplot as plt
import tensorflow as tf

from autolab_core import YamlConfig

import utils
from config import ProbabilityConfig
from dataset import ImageDataset
from sd_maskrcnn.coco_benchmark import coco_benchmark

import model as modellib
import visualize

def benchmark(config):
    """Benchmarks a model, computes and stores model predictions and then
    evaluates them on COCO metrics and supplementary benchmarking script."""

    print("Benchmarking model.")

    # Create new directory for outputs
    output_dir = config['output_dir']
    utils.mkdir_if_missing(output_dir)

    # Save config in output directory
    config.save(os.path.join(output_dir, config['save_conf_name']))
    image_shape = config['model']['settings']['image_shape']
    config['model']['settings']['image_min_dim'] = min(image_shape)
    config['model']['settings']['image_max_dim'] = max(image_shape)
    config['model']['settings']['gpu_count'] = 1
    config['model']['settings']['images_per_gpu'] = 1
    inference_config = ProbabilityConfig(config['model']['settings'])
    
    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                              model_dir=model_dir)

    # Load trained weights
    print("Loading weights from ", config['model']['path'])
    model.load_weights(config['model']['path'], by_name=True)

    # Create dataset
    test_dataset = ImageDataset(config)
    test_dataset.load(config['dataset']['indices'])
    test_dataset.prepare()

    ######## BENCHMARK JUST CREATES THE RUN DIRECTORY ########
    # code that actually produces outputs should be plug-and-play
    # depending on what kind of benchmark function we run.

    # Create predictions and record where everything gets stored.
    pred_info_dir = detect(config['output_dir'], inference_config, model, test_dataset)

    # ap, ar = coco_benchmark(pred_mask_dir, pred_info_dir, gt_mask_dir)
    if config['vis']['predictions']:
        visualize_predictions(config['output_dir'], test_dataset, inference_config, pred_info_dir, 
                              show_bbox=config['vis']['show_bbox_pred'], show_scores=config['vis']['show_scores_pred'], show_class=config['vis']['show_class_pred'])
    if config['vis']['ground_truth']:
        visualize_gts(config['output_dir'], test_dataset, inference_config, show_scores=False, show_bbox=config['vis']['show_bbox_gt'], show_class=config['vis']['show_class_gt'])
    
    if config['vis']['diffs']:
        visualize_differences(config['output_dir'], test_dataset, inference_config, pred_info_dir, 
                              show_bbox=config['vis']['show_bbox_pred'], show_scores=config['vis']['show_scores_pred'], show_class=config['vis']['show_class_pred'])
    
    print("Saved benchmarking output to {}.\n".format(config['output_dir']))
    # return ap, ar

def detect(run_dir, inference_config, model, dataset):
    """
    Given a run directory, a MaskRCNN config object, a MaskRCNN model object,
    and a Dataset object,
    - Loads and processes ground-truth masks, saving them to a new directory
      for annotation
    - Makes predictions on images
    - Saves prediction masks in a certain directory
    - Saves other prediction info (scores, bboxes) in a separate directory

    Returns paths to directories for prediction masks, prediction info, and
    modified GT masks.

    If bin_mask_dir is specified, then we will be checking predictions against
    the "bin-vs-no bin" mask for the test case.
    For each predicted instance, if less than overlap_thresh of the mask actually
    consists of non-bin pixels, we will toss out the mask.
    """

    # Create subdirectory for prediction scores & bboxes
    pred_info_dir = os.path.join(run_dir, 'pred_info')
    utils.mkdir_if_missing(pred_info_dir)

    # Feed images into model one by one. For each image, predict and save.
    image_ids = dataset.image_ids
    indices = dataset.indices
    times = []
    print('MAKING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, _, _ = modellib.load_image_gt(dataset, inference_config, image_id)
    
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        times.append(r['time'])

        # Save info
        r_info = {
            'rois': r['rois'],
            'scores': r['scores'],
            'class_ids': r['class_ids']
        }
        r_info_path = os.path.join(pred_info_dir, 'image_{:06d}.npy'.format(image_id))
        np.save(r_info_path, r_info)
    print('Took {} s'.format(sum(times)))
    print('Saved prediction info (bboxes, scores, classes) to:\t {}'.format(pred_info_dir))

    return pred_info_dir

def visualize_predictions(run_dir, dataset, inference_config, pred_info_dir, show_bbox=True, show_scores=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'vis')
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.image_ids

    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, _, _ = modellib.load_image_gt(dataset, inference_config, image_id)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()

        # Visualize
        scores = r['scores'] if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        fig.add_axes(ax)
        visualize.display_instances(image, ax, r['rois'], r['class_ids'], ['bg', 'obj'], 
                                    scores=scores, show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'vis_{:06d}'.format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()

def visualize_gts(run_dir, dataset, inference_config, show_bbox=True, show_scores=False, show_class=True):
    """Visualizes gts."""
    # Create subdirectory for gt visualizations
    vis_dir = os.path.join(run_dir, 'gt_vis')
    utils.mkdir_if_missing(vis_dir)

    # Feed images one by one
    image_ids = dataset.image_ids

    print('VISUALIZING GROUND TRUTHS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, inference_config, image_id)

        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # Visualize
        scores = np.ones(gt_class_id.size) if show_scores else None
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        fig.add_axes(ax)
        visualize.display_instances(image, ax, gt_bbox, gt_class_id, ['bg', 'obj'], 
                                    scores, show_bbox=show_bbox, show_class=show_class)
        file_name = os.path.join(vis_dir, 'gt_vis_{:06d}'.format(image_id))
        height, width = image.shape[:2]
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()


def visualize_differences(run_dir, dataset, inference_config, pred_info_dir, show_bbox=True, show_scores=True, show_class=True):
    """Visualizes predictions."""
    # Create subdirectory for prediction visualizations
    vis_dir = os.path.join(run_dir, 'diff_vis')
    utils.mkdir_if_missing(vis_dir)

    # Feed images into model one by one. For each image visualize predictions
    image_ids = dataset.image_ids

    print('VISUALIZING PREDICTIONS')
    for image_id in tqdm(image_ids):
        # Load image and ground truth data and resize for net
        image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, inference_config, image_id)
        if inference_config.IMAGE_CHANNEL_COUNT == 1:
            image = np.repeat(image, 3, axis=2)

        # load mask and info
        r = np.load(os.path.join(pred_info_dir, 'image_{:06}.npy'.format(image_id))).item()

        # Visualize
        fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        fig.add_axes(ax)
        visualize.display_differences(image, ax, gt_bbox, gt_class_id, r['rois'], 
                                      r['class_ids'], r['scores'], ['bg', 'obj'], 
                                      show_bbox=show_bbox)

        file_name = os.path.join(vis_dir, 'diff_vis_{:06d}'.format(image_id))
        fig.savefig(file_name, transparent=True, dpi=300)
        plt.close()

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    
    # Set up tf session to use what GPU mem it needs and benchmark
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as sess:
    #     set_session(sess)
    benchmark(config)
