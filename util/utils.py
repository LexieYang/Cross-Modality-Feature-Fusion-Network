import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


colors_per_class = {
    'airplane' : [110, 19, 207], # 0
    'bathtub' : [0, 204, 255], # 1
    'bed' : [55, 159, 43], #  2 
    'bench' : [153, 51, 0], # 3
    'bookshelf' : [254, 202, 87], # 4
    'bottle' : [10, 189, 227], # 5
    'bowl' : [0, 0, 0], # 6
    'car' : [0, 255, 255], # 7
    'chair' : [20, 102, 187], # 8
    'cone' : [204, 204, 255], # 9

    'cup' : [153, 150, 90],  # 10
    'curtain' : [255, 128, 128], # 11
    'desk' : [128, 128, 128], # 12
    'door' : [255, 204, 0], # 13
    'dresser' : [110, 189, 0], # 14
    'flower_pot' : [87, 101, 116], # 15
    'glass_box' : [0, 159, 243], # 16
    'guitar' : [0, 151, 102], # 17
    'keyboard' : [52, 31, 151], # 18
    'lamp' : [255, 157, 204], # 19
    
    'laptop' : [102, 0, 102], # 20
    'mantel' : [152, 231, 90], # 21
    'monitor' : [216, 72, 32], # 22
    'night_stand' : [16, 172, 132], # 23
    'person' : [100, 200, 0], # 24
    'piano' : [255, 159, 243], # 25
    'plant' : [100, 100, 255], # 26
    'radio' : [0, 128, 0], # 27
    'range_hood' : [128, 80, 128], # 28
    'sink' : [110, 110, 210], # 29
    
    'sofa' : [255, 107, 107], # 30
    'stairs' : [153, 204, 0], # 31
    'stool' : [0, 51, 102], # 32
    'table' : [128, 180, 28], # 33
    'tent' : [0, 128, 128], # 34
    'toilet' : [50, 201, 99], # 35
    'tv_stand' : [70, 70, 210], # 36
    'vase' : [255, 0, 107], # 37
    'wardrobe' : [51, 51, 0], # 38
    'xbox' : [128, 0, 0] # 39
    
}
# TESTING_SET = ['bookshelf', 'vase', 'bottle', 'piano', 'night_stand', 'range_hood', 'flower_pot', 'keyboard', 'sink', 'person']



    # scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def cos_sim(a, b, eps=1e-6):
    a = a.unsqueeze(-1)
    b = b.unsqueeze(0).transpose(1, 2)
    output = F.cosine_similarity(a, b, dim=1)
    return output

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.squeeze()

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)