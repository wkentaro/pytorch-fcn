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

It is ~4 times faster than [FCN implemented with Chainer](https://github.com/wkentaro/fcn),
measuring on Titan X Pascal.

```bash
% ./speedtest.py --gpu 0 --times 1000
==> Running on GPU: 0 to evaluate 1000 times
==> Testing FCN32s with Chainer
Elapsed time: 208.34 [s / 1000 evals]
Hz: 4.80 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 56.30 [s / 1000 evals]
Hz: 17.76 [hz]
```
