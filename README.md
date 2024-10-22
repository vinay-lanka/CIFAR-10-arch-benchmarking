# Image classification on the CIFAR10 dataset

This repository contains the scripts required for the homework 0 submission for the CMSC733 (Classical and Deep Learning Approaches for Geometric Computer Vision) course. Made by Vinay Lanka (120417665) at the University of Maryland. 

This contains a python script for the deep learning introduction. It has training and testing scripts for a variety of deep learning architectures and can train 5 models in the `Phase2/Code/Networks/Networks.py` file.
It deals with the classification of images in the CIFAR10 dataset which needs to be present in the Phase2 directory. 

The different model architectures discussed in this assignment include a basic CNN, an optimized CNN with more layers and batch normalization, a ResNet-based model, a ResNext-based model, and a DenseNet-based model. 

Works on tensorflow 1 and was developed using the tensorflow 2 compatibility layer. Was developed by training using Tensorflow 2.14, CUDA 11.8 and cuDNN 8.7.

To train any network change to the Phase 2 directory and run

```bash
#Change to the Phase 2 directory
$ cd Phase2
$ python3 Code/Train.py --Model <Model Name> 
```

To test any network run 

```bash
$ cd Phase2
$ python3 Code/Test.py --Model <Model Name> 
```