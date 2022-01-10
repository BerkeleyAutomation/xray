"""
Author: Mike Danielczuk

Benchmark Usage Notes:
Please edit "cfg/benchmark.yaml" to specify the necessary parameters.
Run this file with the tag --config [config file name] if different
config from the default location (cfg/benchmark.yaml).
"""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
import tqdm
from autolab_core import YamlConfig
from prettytable import PrettyTable
from torch.autograd import Variable

from xray import XrayDataset, XrayModel, utils


def benchmark(output_dir, model, device, data_loader, config):
    """Benchmarks a model."""

    model.eval()

    if config["viz"]:
        out = osp.join(output_dir, "viz")
        utils.mkdir_if_missing(out)

    benchmark_loss = 0
    batch_metrics = []
    for batch_idx, data in tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Benchmark Progress", ncols=80, leave=False
    ):

        label_trues, label_preds = [], []
        data[0] = utils.cv2_clipped_zoom(data[0], 1 / config["scale"])
        data = [Variable(d.to(device)) for d in data]
        imgs, lbls, ratios = data
        with torch.no_grad():
            score = model(imgs.float())["out"]
        score = utils.cv2_clipped_zoom(score.cpu(), config["scale"]).to(device)
        data[0] = utils.cv2_clipped_zoom(data[0].cpu(), config["scale"]).to(device)

        loss_score = score.squeeze()[range(len(imgs)), ratios]
        loss = torch.nn.MSELoss()(loss_score, lbls.float())
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError("loss is nan while benchmarking")
        benchmark_loss += loss_data / len(loss_score)

        data = [d.data.cpu().numpy() for d in data]
        lbl_pred = loss_score.data.cpu().numpy()
        all_lbls_pred = score.data.cpu().numpy()
        for img, lbl, pred in zip(data[0], data[1], lbl_pred):
            img = utils.untransform(img, data_loader.dataset.mean_bgr)
            label_trues.append(lbl)
            label_preds.append(pred)
        batch_metrics.append(utils.label_accuracy_values(label_trues, label_preds))

        if config["viz"] and batch_idx % config["viz_interval"] == 0:
            out_file = osp.join(out, "viz_{{}}_{num:03d}.png".format(num=int(batch_idx / config["viz_interval"])))
            gt = np.iinfo(np.uint8).max * plt.cm.jet(lbl.astype(np.uint8))
            all_pred = np.iinfo(np.uint8).max * plt.cm.jet(
                all_lbls_pred[-1] / all_lbls_pred[-1].max(axis=(1, 2), keepdims=True)
            )

            if "viz_block" not in config.keys() or config["viz_block"]:
                viz = utils.visualize_segmentation(lbl_pred=all_lbls_pred[-1], lbl_true=lbl, img=img)
            else:
                skimage.io.imsave(out_file.format("input"), img)
                skimage.io.imsave(out_file.format("gt"), gt.astype(np.uint8))
                for j, ap in enumerate(all_pred):
                    skimage.io.imsave(
                        out_file.format("pred_{:02d}".format(config["model"]["ratios"][j])), ap.astype(np.uint8)
                    )
            if "vis_block" not in config.keys() or config["viz_block"]:
                skimage.io.imsave(out_file.format("block"), viz)

    metrics = utils.label_accuracy_scores(*np.array(batch_metrics).sum(axis=0))
    benchmark_loss /= len(data_loader)

    t = PrettyTable(["N", "Loss", "Acc", "Bal-Acc", "IoU"])
    t.add_row([data_loader.batch_size * len(data_loader)] + [np.round(benchmark_loss, 1)] + list(np.round(metrics, 2)))
    res = {k: v if isinstance(v, int) else float(v) for (k, v) in zip(t._field_names, t._rows[0])}
    results = YamlConfig()
    results.update(res)
    results.save(os.path.join(output_dir, "results.yaml"))
    print(t)


if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and benchmark
    conf_parser = argparse.ArgumentParser(description="Benchmark FCN model")
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/benchmark.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    print("Benchmarking model")

    # Create new directory for outputs
    output_dir = osp.join(config["output_dir"], osp.split(config["model"]["path"])[-1])
    utils.mkdir_if_missing(output_dir)

    model_cfg = YamlConfig(osp.join(config["model"]["path"], f"{osp.split(config['model']['path'])[-1]}.yaml"))
    config["model"]["ratios"] = model_cfg["model"]["ratios"]

    # Save config in output directory
    config.save(os.path.join(output_dir, "benchmark.yaml"))

    # 1. model
    device = torch.device(config["model"]["device"])
    if "cuda" in config["model"]["device"]:
        torch.backends.cudnn.benchmark = True
    model = XrayModel(ratios=config["model"]["ratios"]).to(device)
    checkpoint = torch.load(osp.join(config["model"]["path"], "model_best.pth.tar"))
    model.load(checkpoint["model_state_dict"])

    # 2. dataset
    test_set = XrayDataset(
        config["dataset"]["path"],
        split="test",
        imgs=config["dataset"]["imgs"],
        lbls=config["dataset"]["lbls"],
        ratios=config["model"]["ratios"],
        mean_bgr=config["dataset"]["mean"],
        max_ind=config["dataset"]["max_ind"],
    )
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=config["model"]["batch_size"], shuffle=False)

    benchmark(output_dir, model, device, data_loader, config)
