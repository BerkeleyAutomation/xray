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
from autolab_core.utils import keyboard_input

import utils
import fcn_dataset
from siamese_fcn import siamese_fcn

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter,
                 reduction='mean', interval_validate=None,
                 use_amp=False):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()
        self.reduction = reduction
        self.use_amp = use_amp

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/bal_acc',
            'train/mean_iou',
            'valid/loss',
            'valid/acc',
            'valid/bal_acc',
            'valid/mean_iou',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iou = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, data in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data = tuple([d.cuda() for d in data])
            data = tuple([Variable(d) for d in data])
            lbl = data[-1]
            with torch.no_grad():
                score = self.model(*data[:-1])
            score = score['out'].squeeze()
            
            # pos_weight = (target.nelement() - target.sum()) / target.sum()
            # x_ent_2D = torch.nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
            # loss = x_ent_2D(score, target)
            loss = torch.nn.MSELoss()(score, lbl)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(score)

            data = tuple([d.data.cpu() for d in data])
            lbl_pred = score.data.cpu().numpy()
            for d in zip(*data, lbl_pred):
                dd = self.val_loader.dataset.untransform(*d[:-1])
                label_trues.append(dd[-1])
                label_preds.append(d[-1])
                if len(visualizations) < 9:
                    viz = utils.visualize_segmentation(lbl_pred=d[-1], lbl_true=dd[-1], img=dd[0])
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(label_trues, label_preds)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 4 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iou = metrics[-1]
        is_best = mean_iou > self.best_mean_iou
        if is_best:
            self.best_mean_iou = mean_iou
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'amp': amp.state_dict(),
            'best_mean_iou': self.best_mean_iou,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        for batch_idx, data in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration > 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data = tuple([d.cuda() for d in data])
            data = tuple([Variable(d) for d in data])
            lbl = data[-1]
            self.optim.zero_grad()
            score = self.model(*data[:-1])['out'].squeeze()

            # pos_weight = (target.nelement() - target.sum()) / target.sum()
            # x_ent_2D = torch.nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
            # loss = x_ent_2D(score, target)
            loss = torch.nn.MSELoss()(score, lbl)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            if self.use_amp:
                with amp.scale_loss(loss, self.optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optim.step()
            loss_data /= len(score)

            # lbl_pred = torch.sigmoid(score).data.cpu().numpy()
            lbl_pred = score.data.cpu().numpy()
            lbl_true = lbl.data.cpu().numpy()
            metrics = utils.label_accuracy_score(lbl_true, lbl_pred)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now() - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                       list(metrics) + [''] * 4 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    
    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

if __name__ == "__main__":

    conf_parser = argparse.ArgumentParser(description="Train model")
    conf_parser.add_argument("--config", action="store", default="cfg/train_fcn.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_parser.add_argument("--resume", action="store_true", help='resume previous training')
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    resume = conf_args.resume

    now = datetime.datetime.now()
    
    out = osp.join(config['model']['path'], config['model']['name'])
    if osp.exists(out) and not resume:
        response = keyboard_input('A model folder already exists at {}. Would you like to overwrite?'.format(out), yesno=True)
        if response.lower() == 'n':
            os.exit()
    elif osp.exists(out) and resume:
        resume = resume and osp.exists(osp.join(out, 'checkpoint.pth.tar'))
    else osp.exists(out):
        os.makedirs(out)
    config.save(os.path.join(out, config['save_conf_name']))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['model']['gpu'])
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.benchmark = True

    # 1. model
    siamese = (config['model']['type'] == 'siamese_fcn')
    model = siamese_fcn() if siamese else fcn_resnet50(num_classes=1)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(osp.join(out, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']

    if cuda:
        model = model.cuda()

    # 2. dataset
    root = config['dataset']['path']
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_set = fcn_dataset.FCNTargetDataset(root, split='train', imgs=config['dataset']['imgs'], targs=config['dataset']['targs'],lbls=config['dataset']['lbls'], transform=True) if siamese else fcn_dataset.FCNDataset(root, split='train', imgs=config['dataset']['imgs'], lbls=config['dataset']['lbls'], transform=True)
    val_set = fcn_dataset.FCNTargetDataset(root, split='test', imgs=config['dataset']['imgs'], targs=config['dataset']['targs'],lbls=config['dataset']['lbls'], transform=True) if siamese else fcn_dataset.FCNDataset(root, split='test', imgs=config['dataset']['imgs'], lbls=config['dataset']['lbls'], transform=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['model']['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['model']['batch_size'], shuffle=True, **kwargs)

    # 3. optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=config['model']['lr'],
        momentum=config['model']['momentum'],
        weight_decay=config['model']['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # If using mixed precision training, initialize here
    if APEX_AVAILABLE:
        if resume:
            amp.load_state_dict(checkpoint['amp'])
        model, optim = amp.initialize(
            model, optim, opt_level="O3", 
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=config['model']['max_iteration'],
        use_amp=APEX_AVAILABLE
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
