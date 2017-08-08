# VOC Example


## Training


```bash
./download_dataset.sh

./train_fcn32s.py -g 0
./train_fcn16s.py -g 0
./train_fcn8s.py -g 0
./train_fcn8s_atonce.py -g 0
```


## Speed

PyTorch implementation is faster for static inputs and slower for dynamic ones than [Chainer one](https://github.com/wkentaro/fcn) at test time.  
(In the previous performance, Chainer one was much slower, but it was fixed via [wkentaro/fcn#90](https://github.com/wkentaro/fcn/pull/90).)  

```bash
# Titan X (Pascal)
# chainer==2.0.2
# pytorch==0.2.0.post2
# pytorch-fcn==1.7.0

% cd examples/voc

% ./speedtest.py --gpu 2
==> Benchmark: gpu=2, times=1000, dynamic_input=False
==> Testing FCN32s with Chainer
Elapsed time: 45.95 [s / 1000 evals]
Hz: 21.76 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 42.63 [s / 1000 evals]
Hz: 23.46 [hz]

% ./speedtest.py --gpu 3 --dynamic-input
==> Benchmark: gpu=3, times=1000, dynamic_input=True
==> Testing FCN32s with Chainer
Elapsed time: 47.68 [s / 1000 evals]
Hz: 20.97 [hz]
==> Testing FCN32s with PyTorch
Elapsed time: 54.49 [s / 1000 evals]
Hz: 18.35 [hz]
```


## Caffe to PyTorch model

```
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
vim Makefile.config  # edit as you like
make -j
make pycaffe
export PYTHONAPTH=$(pwd)/python:$PYTHONPATH
cd ..

cd pytorch-fcn
cd examples/voc
./model_caffe_to_pytorch.py
```
