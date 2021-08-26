"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
from __future__ import division
import os
import math
import random
import numpy as np
import scipy.ndimage
from matplotlib.pyplot import cm
import skimage
import skimage.transform
import copy
import six
import warnings
from distutils.version import LooseVersion

def mkdir_if_missing(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print("Something went wrong in mkdir_if_missing. "
                  "Probably some other process created the directory already.")


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0)]
    if image.ndim == 3:
        padding.append((0, 0))
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / float(min(h, w)))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / float(image_max)

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        if image.ndim == 3:
            padding.append((0, 0))
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        if image.ndim == 3:
            padding.append((0, 0))
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_dist(dist, scale, padding, crop=None):
    """Resizes a dist image using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the dist image, are resized consistently.

    scale: scaling factor
    padding: Padding to add in the form
            [(top, bottom), (left, right)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = scipy.ndimage.zoom(dist, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        dist = dist[y:y + h, x:x + w]
    else:
        dist = np.pad(dist, padding, mode='constant', constant_values=0)
    return dist

############################################################
#  Miscellaneous
############################################################

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def label_accuracy_values(label_trues, label_preds, pos_thresh=25.5, diff_thresh=51):
    """Returns accuracy score evaluation result.

      - mean accuracy
      - mean IoU
      - balanced accuracy
    """
    lp = np.array(label_preds)
    lt = np.array(label_trues)
    
    true_pos = np.sum(np.logical_and(np.abs(lp - lt) < diff_thresh, lt > pos_thresh))
    true_neg = np.sum(np.logical_and(np.abs(lp - lt) < diff_thresh, lt <= pos_thresh))
    num_pred_pos = np.sum(lt > pos_thresh)
    num_total_pos = np.sum(np.logical_or(lt > pos_thresh, lp > pos_thresh))
    return true_pos, true_neg, num_pred_pos, num_total_pos, lt.size

def label_accuracy_scores(true_pos, true_neg, num_pred_pos, num_total_pos, total_size):
    acc = (true_pos + true_neg) / total_size
    bal_acc = 0.5 * ((true_pos / num_pred_pos) + (true_neg / (total_size - num_pred_pos)))
    iou = true_pos / num_total_pos
    
    return acc, bal_acc, iou


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical + h,
               pad_horizontal:pad_horizontal + w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            n_channels = imgs[0].shape[2]
            assert all(im.shape[2] == n_channels for im in imgs)
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, n_channels),
                dtype=np.uint8,
            )
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in six.moves.range(y_num):
        for x in six.moves.range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                                   x * one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    def resize(*args, **kwargs):
        # anti_aliasing arg cannot be passed to skimage<0.14
        # use LooseVersion to allow 0.14dev.
        if LooseVersion(skimage.__version__) < LooseVersion('0.14'):
            kwargs.pop('anti_aliasing', None)
        return skimage.transform.resize(*args, **kwargs)

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return y_num, x_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(
            image=img,
            output_shape=(h, w),
            mode='reflect',
            preserve_range=True,
            anti_aliasing=True,
        ).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)


def label2rgb(lbl, img=None, alpha=0.5, thresh_suppress=0):

    cmap = cm.jet
    lbl_viz = cmap(lbl / lbl.max())[...,:-1]
    lbl_viz = np.iinfo(np.uint8).max * lbl_viz

    if img is not None:
        img_gray = skimage.color.rgb2gray(img)
        img_gray = skimage.color.gray2rgb(img_gray)
        img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray

    return lbl_viz.astype(np.uint8)


def visualize_segmentation(**kwargs):
    """Visualize segmentation.

    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.

    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred = copy.deepcopy(lbl_pred)

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            label2rgb(lbl_true),
            label2rgb(lbl_true, img),
        ]
        vizs.append(get_tile_image(viz_trues, (1, len(viz_trues))))

    if lbl_pred is not None:
        if isinstance(lbl_pred, np.ndarray) and lbl_pred.ndim == 2:
            lbl_pred = [lbl_pred]
        for lp in lbl_pred:
            viz_preds = [
                img,
                label2rgb(lp),
                label2rgb(lp, img),
            ]
            vizs.append(get_tile_image(viz_preds, (1, len(viz_preds))))

    if len(vizs) == 1:
        return vizs[0]
    else:
        return get_tile_image(vizs, (len(vizs), 1))
