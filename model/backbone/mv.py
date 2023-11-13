import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pcview import PCViews
from util.utils import Squeeze


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)




# class view_pooling(nn.Module):
#     def __init__(self,inchannel=32,out_channel=32):
#         super().__init__()
#         self.net=nn.Sequential(nn.Conv2d(6*inchannel,out_channel,kernel_size=3,padding=1),
#                                nn.ReLU())
    
    
#     def forward(self,x):
#         '''
#         x's shape is (bs,6,32,64,64)
#         '''
#         lr=torch.max(x[:,[0,2],:,:],1)[0] # left and right
#         fb=torch.max(x[:,[1,3],:,:],1)[0] # front and back
#         tb=torch.max(x[:,[4,5],:,:],1)[0] # top and bottom
        
#         lft=torch.max(x[:,[0,1,4],:,:],1)[0] # left front and top
#         rbb=torch.max(x[:,[2,3,5],:,:],1)[0] # right back and bottom
        
#         al=torch.max(x,1)[0]
        
#         feat=torch.cat([al,lr,fb,tb,lft,rbb],1)
#         feat=self.net(feat)
        
#         return feat







class MVModel(nn.Module):
    def __init__(self, feat_size=32, backbone='resnet18'):
        super().__init__()

        self.hidden_dim=256

        self.pcview=PCViews()
        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = nn.Sequential(*img_layers)
        # ===== bin number ======
        self.bin_num = [1, 2, 4, 8, 16, 32]

        self.final=nn.Linear(128, self.hidden_dim)


    def frame_max(self, x):
        return torch.max(x, 1)




    def get_img(self,inpt): # inpt.shape: torch.Size([20, 3, 1024])
        bs=inpt.shape[0]
        imgs=self.pcview.get_img(inpt.permute(0,2,1))
        _,h,w=imgs.shape # torch.Size([120, 128, 128])
        
        imgs=imgs.reshape(bs, 6, -1)
        max=torch.max(imgs,-1,keepdim=True)[0]
        min=torch.min(imgs,-1,keepdim=True)[0]
        
        nor_img=(imgs-min)/(max-min+0.0001)
        nor_img=nor_img.reshape(bs,6,h,w)
        return nor_img

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from model.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-3]
        img_layers = [
            nn.Conv2d(6, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers
            # Squeeze()
        ]

        return img_layers, in_features

    def forward(self,inpt):
        '''
        norm_img shape is (20,6,128,128)
        20 is the batch_size
        6 is the view number
        128 is the image size
        '''
        norm_img=self.get_img(inpt) # (20,6,128,128)
        # norm_img=norm_img.unsqueeze(2)
        
        feat = self.img_model(norm_img) # torch.Size([20, 128, 32, 32])
        # feat, att = self.CSAM(feat)
        # return feat
        n, c, h, w = feat.size()
        feat = feat.view(n, c, -1)
        ###### fc layer #########
        # x1 = self.fc(feat)
        # x1 = x1.permute(2, 0, 1).contiguous()
        # feature=self.final(x1)
        # return feature
        #########################
        ###### bins #########
        feature=[]
        
        for num_bin in self.bin_num:
            # feat = feat[:, :, torch.randperm(feat.shape[-1])]
            z = feat.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            # z = torch.cat([z.mean(3), z.max(3)[0]], 2)
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous() # torch.Size([31, 30, 128])
        feature=self.final(feature) # torch.Size([31, 30, 128])
        return feature
        #########################


if __name__=='__main__':
    '''
    5 way
    1 shot
    3 query
    '''

    inpt=torch.randn((20,3,1024))
    network=MVModel()
    out=network(inpt)
