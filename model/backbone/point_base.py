import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # batch, num_dims, num_points, k_neighbors
  
    return feature



class point_backbone(nn.Module):
    def __init__(self):
        super(point_backbone, self).__init__()
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), # !!!! modify this by ymm
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        

        
        # self.fc = nn.Linear(1024, 32)
        self.feat_transform = nn.Linear(1024, 256)
        self.bin_num = [1, 2, 4, 8, 16, 32]


    def forward(self, x): #(B, 3, 1024)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x) # torch.Size([30, 64, 1024, 20])
        x1 = x.max(dim=-1, keepdim=False)[0] # torch.Size([30, 64, 1024])
   

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # torch.Size([30, 64, 1024])
    


        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([30, 128, 1024])

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x) 
        x4 = x.max(dim=-1, keepdim=False)[0]  # torch.Size([30, 256, 1024])

        x = torch.cat((x1, x2, x3, x4), dim=1) # torch.Size([30, 512, 1024])

        x = self.conv5(x)  # torch.Size([30, 1024, 1024])


        batch, feat_dim, n_points = x.shape
        # x1 = self.fc(x)
        # x1 = x1.permute(2, 0, 1).contiguous()
        # feature = self.feat_transform(x1)
        # return feature

        bin_feat = []


        for bin in self.bin_num:
            # x = x[:, :, torch.randperm(x.shape[-1])]
            z = x.view(batch, feat_dim, bin, -1)
            z_max, _ = z.max(3)
            # z = torch.cat([z.mean(3), z_max], 2)
            z = z.mean(3)+z_max
            bin_feat.append(z)

        bin_feat = torch.cat(bin_feat, 2).permute(2, 0, 1).contiguous() # 31/62, 6, 1024
        bin_feat = self.feat_transform(bin_feat) # bins, batch, 1024

        return bin_feat
      






if __name__=='__main__':
    inpt=torch.rand((10,3,1024))
    network=DGCNN_fs()
    out_feat, _=network(inpt) #out_feat shape is (10,1024)
    print(out_feat.shape)
    
