import os
import os.path as osp
import shutil

import fcn
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchfcn import datasets
from torchfcn import models


cuda = True
seed = 1
max_iter = 100000
best_mean_iu = 0
log_headers = [
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

with open('log.csv', 'w') as f:
    f.write(','.join(log_headers) + '\n')

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# 1. dataset

root = '/home/wkentaro/.chainer/dataset'
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.PascalVOC2012ClassSeg(
        root, train=True, transform=True),
    batch_size=1, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
    datasets.PascalVOC2012ClassSeg(
        root, train=False, transform=True),
    batch_size=1, shuffle=False, **kwargs)

# 2. model

model = models.FCN32s(n_class=21)
pth_file = '/home/wkentaro/.torch/models/vgg16-00b39a1b.pth'
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load(pth_file))
for l1, l2 in zip(vgg16.features, model.features):
    try:
        if l1.weight.size() == l2.weight.size():
            l2.weight = l1.weight
        if l1.bias.size() == l2.bias.size():
            l2.bias = l1.bias
    except AttributeError:
        pass
if cuda:
    model = model.cuda()

# 3. optimizer

optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.99,
                      weight_decay=0.0005)


def validate(epoch):
    model.eval()

    val_loss = 0
    metrics = []
    vizs = []
    for batch_idx, (data, target) in enumerate(val_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        n, c, h, w = output.size()
        logit = output.transpose(1, 2).transpose(2, 3).contiguous()
        labels = target
        logit = logit[labels.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        logit = logit.view(-1, c)
        labels = labels[labels >= 0]
        val_loss += F.cross_entropy(logit, labels, size_average=False).data[0]

        data = data.data.cpu()
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, 0, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(data, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(
                lt, lp, n_class=21)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            if len(vizs) < 9:
                viz = fcn.utils.visualize_segmentation(lp, lt, img, n_class=21)
                vizs.append(viz)
    metrics = np.mean(metrics, axis=0)
    scipy.misc.imsave('%08d.jpg' % epoch, fcn.utils.get_tile_image(vizs))

    val_loss /= len(val_loader)

    with open('log.csv', 'a') as f:
        iteration = epoch * len(train_loader)
        log = [epoch, iteration] + [''] * 5 + [val_loss] + metrics.tolist()
        log = map(str, log)
        f.write(','.join(log) + '\n')

    global best_mean_iu
    mean_iu = metrics[2]
    is_best = mean_iu > best_mean_iu
    if is_best:
        best_mean_iu = mean_iu
    torch.save({
        'epoch': epoch,
        'arch': 'FCN32s',
        'state_dict': model.state_dict(),
        'best_mean_iu': best_mean_iu,
    }, 'checkpoint.pth.tar')
    if is_best:
        shutil.copy('checkpoint.pth.tar', 'model_best.pth.tar')


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        n, c, h, w = output.size()
        logit = output.transpose(1, 2).transpose(2, 3).contiguous()
        labels = target
        logit = logit[labels.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        logit = logit.view(-1, c)
        labels = labels[labels >= 0]
        loss = F.cross_entropy(logit, labels, size_average=False)
        loss.backward()
        optimizer.step()

        metrics = []
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, 0, :, :]
        lbl_true = target.data.cpu().numpy()
        for lt, lp in zip(lbl_true, lbl_pred):
            acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(
                lt, lp, n_class=21)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
        metrics = np.mean(metrics, axis=0)

        iteration = batch_idx + epoch * len(train_loader)

        with open('log.csv', 'a') as f:
            log = [epoch, iteration] + [loss.data[0]] + metrics.tolist() + [''] * 5
            log = map(str, log)
            f.write(','.join(log) + '\n')

        if iteration >= max_iter:
            break
    return iteration


epoch = 0
while True:
    validate(epoch)
    iteration = train(epoch)
    if iteration >= max_iter:
        break
    epoch += 1
