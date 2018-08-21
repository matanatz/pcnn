# PCNN - Point Convolutional Neural Networks by Extension Operators 

<p align="center">
  <img src="teaser.png"/>
</p>


This repository contains an implementation to the SIGGRAPH 2018 paper: PCNN - Point Convolutional Neural Networks by Extension Operators.

PCNN is a novel framework for applying convolutional neural networks to point clouds. The framework consists of two operators: extension and restriction, mapping point cloud functions to volumetric functions and vise-versa. A point cloud convolution is defined by pull-back of the Euclidean volumetric convolution via an extension-restriction mechanism. 

For more details visit: https://arxiv.org/abs/1803.10091.

### Installation Requirmenets
The code is compatible with python 3.5 + tensorflow 1.8. In addition, the following packages are required:  
pyhocon, h5py.

### Usage
* To run the training procedure on the ModelNet40 classification task:  
	python train.py

* Training outputs are saved in:  
	./exp_results/pcnn/[host_name]/[gpu]/[timestamp]

* To run evaluation:  
	python ./exp_results/pcnn/[host_name]/[gpu]/[timestamp]/evaluate.py
  
The file pointconv.conf containts additional confguration parameters. To train with a different config file:  
  python train.py --config file_name
