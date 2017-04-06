# pytorch-fcn


Fully Convolutional Networks implemented with PyTorch.


## TODO

- Support FCN16s and FCN8s.


## Accuracy

**FCN32s**

- `deconv=False`
- `train=SBDClassSeg(split='train')`
- `val=VOC2011(split='seg11val')`
- `batch_size=1`
- `MomentumSGD(lr=1e-10, momentum=0.99, weight_decay=0.0005)`

|   epoch |   iteration |   valid/loss |   valid/acc |   valid/acc_cls |   valid/mean_iu |   valid/fwavacc |
|--------:|------------:|-------------:|------------:|----------------:|----------------:|----------------:|
|       9 |       76482 | 59656.847812 |    0.897753 |        0.780288 |        0.628707 |        0.844420 |

<img src="_static/fcn32s_voc2012_best_epoch9.jpg" width="40%" />
<img src="_static/fcn32s_voc2012_visualization_val.gif" width="40%" />


## Speed

PyTorch implementation is a little slower than [Chainer one](https://github.com/wkentaro/fcn) at test time.
(In the previous performance, Chainer one was slower, but it was fixed via [wkentaro/fcn#90](https://github.com/wkentaro/fcn/pull/90).)

```bash
% ./speedtest.py
==> Running on GPU: 0 to evaluate 1000 times
==> Testing FCN32s with Chainer
Elapsed time: 52.03 [s / 1000 evals]
Hz: 19.22 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 58.78 [s / 1000 evals]
Hz: 17.01 [hz]
```
