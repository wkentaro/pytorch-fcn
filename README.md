# pytorch-fcn

[![PyPI Version](https://img.shields.io/pypi/v/torchfcn.svg)](https://pypi.python.org/pypi/torchfcn)
[![Python Versions](https://img.shields.io/pypi/pyversions/torchfcn.svg)](https://pypi.org/project/torchfcn)
[![GitHub Actions](https://github.com/wkentaro/pytorch-fcn/workflows/CI/badge.svg)](https://github.com/wkentaro/pytorch-fcn/actions)

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org).


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
pip install .

# or

pip install torchfcn
```


## Training

See [VOC example](examples/voc).


## Accuracy

At `10fdec9`.

| Model | Implementation |   epoch |   iteration | Mean IU | Pretrained Model |
|:-----:|:--------------:|:-------:|:-----------:|:-------:|:----------------:|
|FCN32s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/main/voc-fcn32s)       | - | -     | **63.63** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/45c6b2d3f553cbe6369822d17a7a51dfe9328662/torchfcn/models/fcn32s.py#L34) |
|FCN32s      | Ours                                                                                         |11 | 96000 | 62.84 | |
|FCN16s      | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/main/voc-fcn16s)       | - | -     | **65.01** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/45c6b2d3f553cbe6369822d17a7a51dfe9328662/torchfcn/models/fcn16s.py#L17) |
|FCN16s      | Ours                                                                                         |11 | 96000 | 64.91 | |
|FCN8s       | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/main/voc-fcn8s)        | - | -     | **65.51** | [Download](https://github.com/wkentaro/pytorch-fcn/blob/45c6b2d3f553cbe6369822d17a7a51dfe9328662/torchfcn/models/fcn8s.py#L17) |
|FCN8s       | Ours                                                                                         | 7 | 60000 | 65.49 | |
|FCN8sAtOnce | [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/main/voc-fcn8s-atonce) | - | -     | **65.40** | |
|FCN8sAtOnce | Ours                                                                                         |11 | 96000 | 64.74 | |

<img src=".readme/fcn8s_iter28000.jpg" width="50%" />
Visualization of validation result of FCN8s.


## Cite This Project

If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entry.

```bash
@misc{pytorch-fcn2017,
  author =       {Ketaro Wada},
  title =        {{pytorch-fcn: PyTorch Implementation of Fully Convolutional Networks}},
  howpublished = {\url{https://github.com/wkentaro/pytorch-fcn}},
  year =         {2017}
}
```
