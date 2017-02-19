import itertools
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Trainer(object):

    def __init__(self, device_ids, model, optimizer,
                 train_loader, val_loader, out, max_iter):
        self.device_ids = device_ids
        assert len(self.device_ids) == 1  # TODO(wkentaro): support multi-gpu

        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
        ]
        with open(osp.join(self.out, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        self.model.eval()
        self.iteration = self.epoch * len(self.train_loader)

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        metrics = []
        visualizations = []
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.device_ids[0] >= 0:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)

            n, c, h, w = output.size()
            logit = output.transpose(1, 2).transpose(2, 3).contiguous()
            labels = target
            logit = logit[labels.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
            logit = logit.view(-1, c)
            labels = labels[labels >= 0]
            val_loss += F.cross_entropy(logit, labels,
                                        size_average=False).data[0]

            imgs = data.data.cpu()
            lbl_pred = output.data.max(1)[1].cpu().numpy()[:, 0, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                acc, acc_cls, mean_iu, fwavacc = \
                    fcn.utils.label_accuracy_score(lt, lp, n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lp, lt, img, n_class=n_class)
                    visualizations.append(viz)
        metrics = np.mean(metrics, axis=0)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, '%08d.jpg' % self.epoch)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + metrics.tolist()
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'arch': self.model.__class__.__name__,
            'state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.iteration = batch_idx + self.epoch * len(self.train_loader)

            if self.device_ids[0] >= 0:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)

            n, c, h, w = output.size()
            logit = output.transpose(1, 2).transpose(2, 3).contiguous()
            labels = target
            logit = logit[labels.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
            logit = logit.view(-1, c)
            labels = labels[labels >= 0]
            loss = F.cross_entropy(logit, labels, size_average=False)
            loss.backward()
            self.optimizer.step()

            metrics = []
            lbl_pred = output.data.max(1)[1].cpu().numpy()[:, 0, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    fcn.utils.label_accuracy_score(lt, lp, n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                      metrics.tolist() + [''] * 5
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        for epoch in itertools.count():
            self.epoch = epoch
            self.validate()
            if self.iteration >= self.max_iter:
                break
            self.train_epoch()
