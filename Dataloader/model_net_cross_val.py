import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
np.random.seed(0)



class ModelNet40_fs(Dataset):
    def __init__(self,root,split='train',fold=0,num_point=1024,data_aug=True):
        super().__init__()
        self.root=root
        self.fold=fold
        self.split=split
        self.num_point=num_point
        self.data_aug=data_aug

        self.point_path,self.point_label=self.get_point()
        
    
    def get_point(self):
        #== will be returned later ==
        point_path_list=[]
        label_list=[]
        #============================

        picked_index=np.zeros(40)
        picked_index[self.fold*10:(self.fold+1)*10]=1
        
        class_list=np.arange(40)
        if self.split=='train':
            picked_index=(1-picked_index).astype(bool)
        else:
            picked_index=picked_index.astype(bool)
        
        class_list=class_list[picked_index]
        for c in class_list:
            class_fold=os.path.join(self.root,str(c))
            for i in os.listdir(class_fold):
                point_path_list.append(os.path.join(class_fold,i))
                label_list.append(c)
        
        return point_path_list,label_list


    def __len__(self):
        return len(self.point_path)
    

    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

    
    
    def __getitem__(self, index):
        point=np.load(self.point_path[index])[:self.num_point]
        label=self.point_label[index]

        if self.split == 'train' and self.data_aug:
            point = self.translate_pointcloud(point)
            np.random.shuffle(point)
        
        pointcloud=torch.FloatTensor(point)
        label=torch.LongTensor([label])
        pointcloud=pointcloud.permute(1,0)
        return pointcloud, label



'''
In the WACV paper
- Totoal 80 epochs used for training
- 400 training episodes and 600 validating episodes for each epoch
- For testing, episodes=700
- n_way=5. k_shot=1. query=15 for each classes

'''


class NShotTaskSampler(Sampler):
    def __init__(self,dataset,episode_num,k_way,n_shot,query_num):
        super().__init__(dataset)
        self.dataset=dataset
        self.episode_num=episode_num
        self.k_way=k_way
        self.n_shot=n_shot
        self.query_num=query_num
        self.label_set=self.get_label_set()
        self.data,self.label =self.dataset.point_path, self.dataset.point_label
    
    def get_label_set(self):
        point_label_set=np.unique(self.dataset.point_label)
        return point_label_set
    
    
    
    
    def __iter__(self):
        for _ in range(self.episode_num):
            support_list=[]
            query_list=[]
            picked_cls_set=np.random.choice(self.label_set,self.k_way,replace=False)
            
            for picked_cls in picked_cls_set:
                target_index=np.where(self.label==picked_cls)[0]
                picked_target_index=np.random.choice(target_index,self.n_shot+self.query_num,replace=False)
                
                support_list.append(picked_target_index[:self.n_shot])
                query_list.append(picked_target_index[self.n_shot:])
                
            s=np.concatenate(support_list)
            q=np.concatenate(query_list)
            
            
            '''
            For epi_index
            - it's the index used for each batch
            - the first k_way*n_shot images is the support set
            - the last k_way*query images is for the query set 
            '''    
            epi_index=np.concatenate((s,q))
            # np.random.shuffle(epi_index[self.n_way:])
            yield epi_index
            

    
    
    def __len__(self):
        return self.episode_num


def get_sets(data_path,fold=0,k_way=5,n_shot=1,query_num=15,data_aug=True):
    train_dataset=ModelNet40_fs(root=data_path,split='train',fold=fold,data_aug=data_aug)
    train_sampler=NShotTaskSampler(dataset=train_dataset,episode_num=400,k_way=k_way,n_shot=n_shot,query_num=query_num)
    train_loader=DataLoader(train_dataset,batch_sampler=train_sampler)
    
    val_dataset=ModelNet40_fs(root=data_path,split='test',fold=fold,data_aug=data_aug)
    val_sampler=NShotTaskSampler(dataset=val_dataset,episode_num=700,k_way=k_way,n_shot=n_shot,query_num=query_num)
    val_loader=DataLoader(val_dataset,batch_sampler=val_sampler)
    
    return train_loader,val_loader

if __name__=='__main__':
    root='/data1/jiajing/dataset/ModelNet40_fewshot/modelnet40_fs_crossvalidation'
    dataset=ModelNet40_fs(root)
    
    train_loader,test_loader=get_sets(data_path=root)
    for (x,y) in train_loader:
        '''
        x' shape is (80,3,1024)
        y's shpae is (80,1)
        '''
        pass
