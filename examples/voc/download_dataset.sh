#!/bin/bash

DIR=~/data/datasets/VOC

mkdir -p $DIR
cd $DIR

if [ ! -e benchmark_RELEASE ]; then
  wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz -O benchmark.tar
  tar -xvf benchmark.tar
fi

if [ ! -e VOCdevkit/VOC2012 ]; then
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_11-May-2012.tar
fi
