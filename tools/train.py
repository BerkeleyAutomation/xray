import argparse
import datetime
import os
import os.path as osp
import shutil
import sys

import numpy as np
import skimage.io
import torch
import tqdm
from autolab_core import YamlConfig
from autolab_core.utils import keyboard_input
from torch.autograd import Variable

from xray import XrayDataset, XrayModel, utils


class Trainer(object):
    def __init__(
        self,
        model,
        device,
        optimizer,
        scaler,
        train_loader,
        val_loader,
        out,
        epochs,
        reduction="mean",
        use_amp=False,
    ):
        self.model = model
        self.device = device
        self.optim = optimizer
        self.scaler = scaler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()
        self.reduction = reduction
        self.use_amp = use_amp

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            "epoch",
            "iteration",
            "train/loss",
            "train/acc",
            "train/bal_acc",
            "train/mean_iou",
            "valid/loss",
            "valid/acc",
            "valid/bal_acc",
            "valid/mean_iou",
            "elapsed_time",
        ]
        if not osp.exists(osp.join(self.out, "log.csv")):
            with open(osp.join(self.out, "log.csv"), "w") as f:
                f.write(",".join(self.log_headers) + "\n")

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = epochs
        self.best_mean_iou = 0

    def validate(self):
        self.model.eval()

        val_loss = 0
        visualizations = []
        batch_metrics = []
        for _, data in tqdm.tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="Valid iteration=%d" % self.iteration,
            ncols=80,
            leave=False,
        ):

            label_trues, label_preds = [], []
            data = [Variable(d.to(self.device)) for d in data]
            imgs, lbls, ratios = data
            with torch.no_grad():
                score = self.model(imgs.float())
                score = score["out"][range(len(imgs)), ratios]

            loss = torch.nn.MSELoss()(score, lbls.float())
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError("loss is nan while validating")
            val_loss += loss_data / len(score)

            data = [d.data.to(torch.device("cpu")) for d in data]
            lbl_pred = score.data.to(torch.device("cpu")).numpy()
            for img, lbl, pred in zip(data[0], data[1], lbl_pred):
                img = utils.untransform(img.numpy(), self.val_loader.dataset.mean_bgr)
                label_trues.append(lbl.numpy())
                label_preds.append(pred)
                if len(visualizations) < 9:
                    viz = utils.visualize_segmentation(lbl_pred=pred, lbl_true=lbl, img=img)
                    visualizations.append(viz)
            batch_metrics.append(utils.label_accuracy_values(label_trues, label_preds))

        metrics = utils.label_accuracy_scores(*np.array(batch_metrics).sum(axis=0))

        out = osp.join(self.out, "validation_viz")
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, "iter%012d.jpg" % self.iteration)
        skimage.io.imsave(out_file, utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, "log.csv"), "a") as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [""] * 4 + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(",".join(log) + "\n")

        mean_iou = metrics[-1]
        is_best = mean_iou > self.best_mean_iou
        if is_best:
            self.best_mean_iou = mean_iou
        torch.save(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_iou": self.best_mean_iou,
            },
            osp.join(self.out, "checkpoint.pth.tar"),
        )
        if is_best:
            shutil.copy(osp.join(self.out, "checkpoint.pth.tar"), osp.join(self.out, "model_best.pth.tar"))

    def train_epoch(self):
        train_bar = tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Train epoch=%d" % self.epoch,
            ncols=80,
            leave=False,
        )

        for batch_idx, data in train_bar:
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            assert self.model.training

            data = [Variable(d.to(self.device)) for d in data]
            imgs, lbls, ratios = data
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                score = self.model(imgs.float())
                score = score["out"][range(len(imgs)), ratios]
                loss = torch.nn.MSELoss()(score, lbls.float())

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError("loss is nan while training")

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()

            loss_data /= len(score)
            train_bar.set_postfix_str(f"Loss: {loss_data:.3f}")

            lbls_pred = score.data.cpu().numpy()
            lbls_true = lbls.data.cpu().numpy()
            metrics = utils.label_accuracy_scores(*utils.label_accuracy_values(lbls_true, lbls_pred))

            with open(osp.join(self.out, "log.csv"), "a") as f:
                elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + list(metrics) + [""] * 4 + [elapsed_time]
                log = map(str, log)
                f.write(",".join(log) + "\n")

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc="Train", ncols=80):
            self.epoch = epoch
            self.model.train()
            self.train_epoch()
            self.model.eval()
            self.validate()


if __name__ == "__main__":

    conf_parser = argparse.ArgumentParser(description="Train model")
    conf_parser.add_argument(
        "--config",
        action="store",
        default="cfg/train.yaml",
        dest="conf_file",
        type=str,
        help="path to the configuration file",
    )
    conf_parser.add_argument("--resume", action="store_true", help="resume previous training")
    conf_parser.add_argument("--amp", action="store_true", help="use automatic mixed precision during training")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    resume = conf_args.resume
    use_amp = conf_args.amp
    device = torch.device(config["model"]["device"])
    if "cuda" in config["model"]["device"]:
        torch.backends.cudnn.benchmark = True

    out = osp.join(config["model"]["path"], config["model"]["name"])
    interactive = bool(hasattr(sys, "ps1"))
    if osp.exists(out) and not resume:
        if not interactive:
            print(f"A model folder already exists at {out}. Quitting...")
            sys.exit()
        response = keyboard_input(f"A model folder already exists at {out}. Would you like to overwrite?", yesno=True)
        if response.lower() == "n":
            sys.exit()
    elif osp.exists(out) and resume:
        resume = resume and osp.exists(osp.join(out, "checkpoint.pth.tar"))
    else:
        os.makedirs(out)

    # 1. model
    model = XrayModel(ratios=config["model"]["ratios"]).to(device)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(osp.join(out, "checkpoint.pth.tar"))
        model.load(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"]

    # 2. dataset
    train_set = XrayDataset(
        config["dataset"]["path"],
        split="train",
        imgs=config["dataset"]["imgs"],
        lbls=config["dataset"]["lbls"],
        ratios=config["model"]["ratios"],
        mean_bgr=config["dataset"]["mean"],
        max_ind=config["dataset"]["max_ind"],
    )
    val_set = XrayDataset(
        config["dataset"]["path"],
        split="test",
        imgs=config["dataset"]["imgs"],
        lbls=config["dataset"]["lbls"],
        ratios=config["model"]["ratios"],
        mean_bgr=config["dataset"]["mean"],
        max_ind=config["dataset"]["max_ind"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config["model"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config["model"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"]
    )

    # 3. optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=config["model"]["lr"],
        momentum=config["model"]["momentum"],
        weight_decay=config["model"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if resume:
        optim.load_state_dict(checkpoint["optim_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 4. save training cfg
    config.save(os.path.join(out, f"{config['model']['name']}.yaml"))

    # 5. train
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optim,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        epochs=config["model"]["epochs"],
        use_amp=use_amp,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
