import fcn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchfcn import models
from torchfcn import datasets


cuda = True
seed = 1
log_interval = 10
max_iter = 100000

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

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


model = models.FCN32s(n_class=21)
if cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.99,
                      weight_decay=0.0005)


def validate(epoch):
    model.eval()
    val_loss = 0
    metrics = []
    for data, target in val_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        lbl_pred = output.data.max(1)[1]

        n, c, h, w = output.size()
        output = output.transpose(1, 2).transpose(2, 3).contiguous()
        output = output.view(-1, c)
        target = target.view(-1)
        val_loss += F.cross_entropy(output, target, size_average=False)[0]

        lbl_true = target.data.cpu()
        for lt, lp in zip(lbl_true, lbl_pred):
            acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(
                lt, lp, n_class=21)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
    metrics = np.mean(metrics, axis=0)

    val_loss /= len(val_loader)
    print('Val Epoch: {0:08}, loss={1:.2f}, acc={2:.4f}, acc_cls={3:.4f}, mean_iu={4:.4f}'
          .format(epoch, val_loss, metrics[0], metrics[1], metrics[3]))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        n, c, h, w = output.size()
        loss = F.cross_entropy(
            output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c),
            target.view(-1),
            size_average=False)
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
        if batch_idx % log_interval == 0:
            print('Train epoch={0:08}, iter={1:012}, loss={2:.2f}, acc={3:.4f}, acc_cls={4:.4f}, mean_iu={5:.4f}'
                  .format(epoch, iteration, loss.data[0], metrics[0], metrics[1], metrics[2], metrics[3]))
        if iteration >= max_iter:
            break
    return iteration


epoch = 0
while True:
    iteration = train(epoch)
    test(epoch)
    if iteration >= max_iter:
        break
    epoch += 1
