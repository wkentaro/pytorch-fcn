# pytorch-fcn

[![PyPI Version](https://img.shields.io/pypi/v/torchfcn.svg)](https://pypi.python.org/pypi/torchfcn)
[![Build Status](https://travis-ci.org/wkentaro/pytorch-fcn.svg?branch=master)](https://travis-ci.org/wkentaro/pytorch-fcn)

Fully Convolutional Networks implemented with PyTorch.


## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)


## Installation

```bash
git clone https://github.com/wkentaro/pytorch-fcn.git
cd pytorch-fcn

conda install pytorch cuda80 torchvision -c soumith
pip install .
```


## Training

See [VOC example](examples/voc).


## Accuracy

At `pytorch-fcn==1.7.0`.

| Model | Implementation |   epoch |   iteration | Accuracy | Accuracy Class | Mean IU | FWAV Accuracy |
|:-----:|:--------------:|:-------:|:-----------:|:--------:|:--------------:|:-------:|:-------------:|
|FCN32s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s)       | - | -     | 90.49 | 76.48 | 63.63 | 83.47 |
|FCN32s      | Ours                                                                                         |10 | 92000 | 90.63 | 72.36 | 63.13 | 83.36 |
|FCN16s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn16s)       | - | -     | 91.00 | 78.07 | 65.01 | 84.27 |
|FCN16s      | Ours                                                                                         | 5 | 44000 | 91.03 | 77.38 | 64.80 | 84.23 |
|FCN8s       | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s)        | - | -     | 91.23 | 77.61 | 65.51 | 84.55 |
|FCN8s       | Ours                                                                                         | 3 | 28000 | 91.24 | 77.12 | 65.39 | 84.55 |
|FCN8sAtOnce | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s-atonce) | - | -     | 91.13 | 78.50 | 65.40 | 84.44 |
|FCN8sAtOnce | Ours                                                                                         | 6 | 56000 | 91.12 | 76.42 | 65.10 | 84.36 |

<img src="static/fcn8s_iter28000.jpg" width="50%" />
Visualization of validation result of FCN8s.
