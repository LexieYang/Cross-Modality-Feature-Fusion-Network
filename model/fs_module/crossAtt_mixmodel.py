import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from util.utils import cos_sim

class crossAtt_mixmodel(nn.Module):
    def __init__(self, n_way, k_shot, query):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.feat_dim = 256
        self.bin_num = [1, 2, 4, 8, 16]
        
        '''
        the first try is [1], exp0
        the second try is [1, 16], exp0_1
        '''
        
        self.trip_loss = trip(n_way, k_shot, query)
        self.transform = nn.Sequential(
                            nn.Linear(256, 256),
                            nn.ReLU(),
                            nn.Linear(256, 1),
                            nn.Sigmoid()
        )
        # self.transform2 = nn.Sequential(
        #                     nn.Linear(1024, 1024),
        #                     nn.Softmax(dim=1)
        # )
        self.to_qk=nn.Linear(256,2*256)
        self.k1 = (self.n_way*self.query) // 2
        self.k2 = 2
        self.support_weight=nn.Sequential(nn.Linear(256*2, 256),
                                          nn.ReLU())
    
        self.query_weight=nn.Sequential(nn.Linear(256*2, 256),
                                        nn.ReLU())


    def getout(self,support,queries):
        dist=torch.cdist(queries,support)
        dist=torch.mean(dist,0)
        y_pred=(-dist).softmax(1)
        return y_pred

    # def self_interaction(self,v): # v.shape: torch.Size([62, 30, 256])
    #     feat=self.to_qk(v)
    #     q,k=torch.split(feat,self.feat_dim,-1)

    #     # === get R matrix ====
    #     R_mat=torch.einsum('bijk,bikx->bijx',q.unsqueeze(-1),k.unsqueeze(-2)) # torch.Size([62, 30, 256, 256])
    #     R_mat=nn.functional.softmax(R_mat,1)
    #     # R_mat=nn.functional.softmax(torch.div(R_mat, torch.sqrt(256)),1)
    #     # =====================

    #     final_feat=torch.einsum('bijk,bikx->bijx', v.unsqueeze(2), R_mat) # torch.Size([62, 30, 1, 256])
    #     final_feat=final_feat.squeeze(2)+v
    #     return final_feat


    def get_bin_sim(self,a,b,eps=1e-6):
        norm_a,norm_b=torch.norm(a,dim=-1),torch.norm(b,dim=-1)
        prod_norm=norm_a.unsqueeze(-1)*norm_b.unsqueeze(1)
        prod_norm[prod_norm<eps]=eps

        prod_mat=torch.bmm(a,b.permute(0,2,1))
        cos_sim=prod_mat/prod_norm
        return cos_sim

    def cross_fusion(self,feat):
        bin_num=feat.shape[0]
        support=feat[:,:self.n_way*self.k_shot,:]
        queries=feat[:,self.n_way*self.k_shot:,:]
        
        # === get distance ===
        bin_sim=self.get_bin_sim(support,queries) # (31,5,25)
        # dist=-bin_sim
        # ====================

        # === obtain fused support ====
        s2q_sim = F.softmax(bin_sim, dim=-1)
        s_cat_q = torch.cat([support, torch.bmm(s2q_sim, queries)], dim=-1)
        cross_support = self.support_weight(s_cat_q) + support
        # =============================

        # ==== obtain fused query ======
        bin_sim = bin_sim.permute(0,2,1)
        q2s_sim = F.softmax(bin_sim, dim=-1)
        q_cat_s = torch.cat([queries, torch.bmm(q2s_sim, support)], dim=-1)
        cross_query = self.support_weight(q_cat_s) + queries
        # ===============================

        return cross_support, cross_query

    def forward(self, feat, label):

        nbins, _, fd = feat.shape


        
        support, query = self.cross_fusion(feat)
        # support, query = self_feat[:, :self.n_way*self.k_shot, :], self_feat[:, self.n_way*self.k_shot:, :]

        support = support.reshape(nbins, self.n_way, self.k_shot, fd)
        proto = torch.mean(support, 2)
        feat = torch.cat([proto, query], dim=1)

        # bin_feat = self.self_interaction(feat)

        y_pred, loss = self.trip_loss(feat, label)

        return y_pred, loss



class trip(nn.Module):
    def __init__(self,k_way,n_shot,query):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.q=query
        self.margin=0.2


    def batch_dist(self,feat):
        return torch.cdist(feat,feat)



    def trip_loss(self, feature, label):
        '''
        feature: shape[batchsize, 512]
        '''
        # ==== get label ====
        bins, batch_size, fd = feature.shape
        # bin_num,sample_num,fd=feature.shape
        label=label.unsqueeze(0).repeat(bins, 1)
        label=label.to(feature.device) # [bin, batchsize]
        # ===================

        # ==== get mask and dist ====
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        dist=self.batch_dist(feature) # [bin, batchsize, batchsize]
        dist=dist.view(-1) # [6291456]


        full_hp_dist=torch.masked_select(dist, hp_mask).reshape(bins, batch_size, -1, 1) 
        full_hn_dist=torch.masked_select(dist, hn_mask).reshape(bins, batch_size, 1, -1)

        # self.adp_margin = torch.masked_select(self.adp_margin, hn_mask).reshape(bins, batch_size, 1, -1)
        full_loss_metric=F.relu(self.margin + full_hp_dist-full_hn_dist).view(bins, -1)
        
        full_loss_metric_sum=torch.sum(full_loss_metric, 1)
        full_loss_num=(full_loss_metric!=0).sum(1).float()
        full_loss_mean=full_loss_metric_sum/full_loss_num
        full_loss_mean[full_loss_num == 0] = 0
        
        return full_loss_mean.mean()


    def getout(self,support,queries):
        dist=torch.cdist(queries,support)
        dist=torch.mean(dist,0)
        y_pred=(-dist).softmax(1)
        return y_pred

    # def adpMargin_loss(self, feature, label, label_sim):
        

    def forward(self,inpt,label):
        label=torch.cat(label)

        proto, query = inpt[:, :self.k, :], inpt[:, self.k:, :]
        loss = self.trip_loss(inpt, label) # label.shape: torch.Size([20])
        y_pred = self.getout(proto, query)

        return y_pred, loss




# class adpMargin_loss(nn.Module):
#     def __init__(self):
        





if __name__=='__main__':
#     '''
#     If backbone is the gait related network
#     the embeding shape is (bin,sample_num,feat_dim), like (62,20,256)
#     '''
    k=5
    q=1
    n=5

    inpt=torch.randn((30, 10, 1024))
    query_label=torch.tensor([0, 1, 2, 3, 4])
    sup_label=torch.arange(k)
    fs=dgcnn_trip(n_way=k,k_shot=n,query=q)
    fs(inpt,[sup_label,query_label])