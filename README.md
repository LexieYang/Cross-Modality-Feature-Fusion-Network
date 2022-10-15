# Cross-Modality Feature Fusion Network for Few-Shot 3D Point Cloud Classification

## Installation

* Install Python 3.6
* Install CUDA 11.0
* Install PyTorch 1.10.2


## Dataset

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
