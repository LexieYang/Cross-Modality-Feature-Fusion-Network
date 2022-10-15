# Cross-Modality Feature Fusion Network for Few-Shot 3D Point Cloud Classification

![overview (1)](https://user-images.githubusercontent.com/63827451/195998009-4463e9c3-af93-4ae1-b47c-6f5472c7e0a3.png)

## Introduction

This repository includes the PyTorch implementation for our WACV 2023 Paper "**Cross-Modality Feature Fusion Network for Few-Shot 3D Point Cloud Classification**" by Minmin Yang, Jiajing Chen and Senem Velipasalar.

Recent years have witnessed significant progress in the field of few-shot image classification while few-shot 3D point cloud classification still remains under-explored. Real-world 3D point cloud data often suffers from occlusions, noise and deformation, which makes the few-shot 3D point cloud classification even more challenging. In this paper, we propose a cross-modality feature fusion network, for few-shot 3D point cloud classification, which aims to recognize an object given only a few labeled samples, and provides better performance even with point cloud data with missing points. More specifically, we train two models in parallel. One is a projection-based model with ResNet-18 as the backbone and the other one is a point-based model with a DGCNN backbone. Moreover, we design a Support-Query Mutual Attention (sqMA) module to fully exploit the correlation between support and query features. Extensive experiments on three datasets, namely ModelNet40, ModelNet40-C and ScanObjectNN, show the effectiveness of our method, and its robustness to missing points. Our proposed method outperforms different state-of-the-art baselines on all datasets. The margin of improvement is even larger on the ScanObjectNN dataset, which is collected from real-world scenes and is more challenging with objects having missing points.

## Installation

This project is built upon the following environment:
* Install Python 3.6
* Install CUDA 11.0
* Install PyTorch 1.10.2

The package requirements include:
* pytorch==1.10.2
* tqdm==4.63.1
* tensorboard==2.8.0

## Datasets

* Download [ModelNet40](https://modelnet.cs.princeton.edu/)
* Download [ModelNet40-C from Google Drive](https://drive.google.com/drive/folders/10YeQRh92r_WdL-Dnog2zQfFr03UW4qXX)
* Download [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)

All data used in this project is in .npy format.
## Train
Train a model on the ModelNet40 dataset by
```
python main.py --dataset modelnet40 --fs_head crossAtt_mixmodel
```

## Evaluate
```
python main.py --train False
```
