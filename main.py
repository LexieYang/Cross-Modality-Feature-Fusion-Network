from logging import raiseExceptions
import torch
from tqdm import tqdm
import argparse
import numpy as np
# from Dataloader.scanobjectnn_cross_val import get_sets
from Dataloader.model_net_cross_val import get_sets
from util.get_acc import cal_cfm
import torch.nn as nn
from model.network import fs_network
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
import scipy.io
import logging

# ============== Get Configuration =================
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--data_aug',default=True)
    cfg.add_argument('--epochs',default=100)
    cfg.add_argument('--decay_ep',default=5)
    cfg.add_argument('--gamma',default=0.5)
    cfg.add_argument('--lr',default=0.0008) # 0.001
    cfg.add_argument('--train',action='store_true',default=True)
    cfg.add_argument('--seed',default=0)
    cfg.add_argument('--dataset', default='modelnet40')
    cfg.add_argument('--data_path',default='./modelnet40_fs_crossvalidation')
    cfg.add_argument('--device',default='cuda')
    cfg.add_argument('--fs_head', default='crossAtt_mixmodel')
    cfg.add_argument('--backbone', default='pointAndview') 
    cfg.add_argument('--exp', default='Debug') # exp
    cfg.add_argument('--exp_description', default='')#MetaOpt
    cfg.add_argument('--val_epoch_size', type=int, default=700)
    # ======== few shot cfg =============#
    cfg.add_argument('--k_way',default=5)
    cfg.add_argument('--n_shot',default=1)
    cfg.add_argument('--query',default=10)
    # ======== cross validation =========#
    cfg.add_argument('--fold',default=0)
    # ===================================#
    
    return cfg.parse_args()
cfg=get_arg()
print(cfg)

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False
# ==================================================

np.random.seed(cfg.seed)


fsl_config = str(cfg.k_way) + "way_" + str(cfg.n_shot) + "shot_" + str(cfg.query) + "query_" + cfg.exp
exp_path = os.path.join('Exp_CrossVal', cfg.fs_head, cfg.dataset+"_fold_{}".format(str(cfg.fold)), fsl_config)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

# ============= create logging ==============
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s, %(name)s, %(message)s')

    ########### this is used to set the log file ##########
    logName = os.path.join(exp_path, 'accuracy.log')
    file_handler = logging.FileHandler(logName)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    #######################################################


    ######### this is used to set the output in the terminal/screen ########
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    #################################################################

    ####### add the log file handler and terminal handerler to the logger #######
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    ##############################################################################

    return logger

logger=get_logger()
# ============================================
logger.debug(cfg)


def main(cfg):
    
    # train_loader,val_loader=get_sets(data_path=cfg.data_path,k_way=cfg.k_way,n_shot=cfg.n_shot,query_num=cfg.query)
    train_loader,val_loader = get_sets(data_path=cfg.data_path,fold=cfg.fold,k_way=cfg.k_way,n_shot=cfg.n_shot,query_num=cfg.query,data_aug=cfg.data_aug)

    model = fs_network(k_way=cfg.k_way,n_shot=cfg.n_shot,query=cfg.query, backbone=cfg.backbone, fs=cfg.fs_head)
    logger.debug(model)

    if cfg.train:
        train_model(model,train_loader,val_loader,cfg)
    
    else:
        test_model(model,val_loader,cfg)

def train_model(model, train_loader, val_loader, cfg):
    device=torch.device(cfg.device)
    model=model.to(device)
    
    #====== loss and optimizer =======
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(5, cfg.epochs, cfg.decay_ep), gamma=cfg.gamma)
    
    
    def train_one_epoch():
        bar=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model, bar,'train',loss_func=loss_func,optimizer=optimizer)
        
        mean_acc=np.mean(epsum['acc'])
        summary={'meac/train':mean_acc}
        summary["loss/train"] = np.mean(epsum['loss'])
        return summary
        
        
    def eval_one_epoch():
        bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model, bar,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        summary={'meac/valid':mean_acc}
        test_accuracies = np.array(epsum['acc'])
        test_accuracies = np.reshape(test_accuracies, -1)
        stds = np.std(test_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(cfg.val_epoch_size)
        summary['std/valid'] = ci95
        return summary,epsum['cfm']
    
    
    # ======== define exp path ===========
    fsl_config = str(cfg.k_way) + "way_" + str(cfg.n_shot) + "shot_" + str(cfg.query) + "query_" + cfg.exp
    exp_path = os.path.join('Exp_CrossVal', cfg.fs_head, cfg.dataset+"_fold_{}".format(str(cfg.fold)), fsl_config)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    # save config into json #
    cfg_dict=vars(cfg)
    yaml_file=os.path.join(exp_path,'config.yaml')
    with open(yaml_file,'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)

    tensorboard = SummaryWriter(log_dir=os.path.join(exp_path,'TB'))
    pth_path = os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    # =====================================
    
    # ========= train start ===============
    acc_list=[]
    tqdm_epochs=tqdm(range(cfg.epochs),unit='epoch',ncols=100)
    for e in tqdm_epochs:
        train_summary=train_one_epoch()
        val_summary,conf_mat=eval_one_epoch()
        summary={**train_summary,**val_summary}
        lr_schedule.step()
    
        accuracy=val_summary['meac/valid']
        std = val_summary['std/valid']
        acc_list.append(val_summary['meac/valid'])
        logger.debug('epoch {}: acc: {:.2%}, std: {:.2%}. Highest: {:.2%}'.format(e, accuracy, std, np.max(acc_list)))
        if np.max(acc_list)==acc_list[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),
                            'cfm':conf_mat}
            torch.save(summary_saved,os.path.join(pth_path,'best.pth'))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    # =======================================    
    


def test_model(model,val_loader,cfg):
    global logger
    logger=get_logger(file_name='testing_result.log')
    fsl_config = str(cfg.k_way) + "way_" + str(cfg.n_shot) + "shot_" + str(cfg.query) + "query_" + cfg.exp

    ckpt_path = os.path.join('Exp_CrossVal', cfg.fs_head, cfg.dataset+"_fold_{}".format(str(cfg.fold)), fsl_config, 'pth_file', 'best.pth')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    

    model=model.cuda()
    bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
    summary=run_one_epoch(model,bar,'test',loss_func=None)

    acc_list=summary['acc']

    mean_acc=np.mean(acc_list)
    std_acc=np.std(acc_list)

    interval=1.960*(std_acc/np.sqrt(len(acc_list)))
    logger.debug('Mean: {}, Interval: {}'.format(mean_acc*100,interval*100))

    



def run_one_epoch(model, bar, mode, loss_func, optimizer=None, show_interval=10):
    val_confusion_mat=np.zeros((cfg.k_way,cfg.k_way))
    train_confusion_mat = np.zeros((cfg.k_way,cfg.k_way))
    
    device=next(model.parameters()).device
    
    if mode=='train':
        model.train()
        summary={"acc":[],"loss":[]}
    else:
        model.eval()
        summary={"acc":[]}
    test_accuracies = []
    for i, (x_cpu, y_cpu) in enumerate(bar):
        # epoch_size += 1
        x, y = x_cpu.to(device),y_cpu.to(device)
        # label = torch.zeros(x.shape[0]-1, 1)
        # label[(y[:-1]==y[-1])[:, 0]] = 1
        num = x.shape[0]
        y = torch.reshape(y, (num,))

        if mode=='train':
            optimizer.zero_grad()
            if cfg.fs_head in ['protonet', 'CIA', 'crossAtt_mixmodel']:
                q_label = torch.arange(cfg.k_way).repeat_interleave(cfg.query).to(device)
                s_label = torch.arange(cfg.k_way).to(device)
                pred,loss = model(x, [s_label, q_label])

            elif cfg.fs_head == 'MetaOpt':
                q_label = torch.arange(cfg.k_way).repeat_interleave(cfg.query).to(device)
                s_label = torch.arange(cfg.k_way).repeat_interleave(cfg.n_shot).to(device)
                pred,loss = model(x, [s_label, q_label])
            else:
                raise Exception("Not implemented error")
            #==take one step==#
            loss.backward()
            optimizer.step()
            #=================#
        else:
            with torch.no_grad():
                if cfg.fs_head in ['protonet', 'CIA', 'crossAtt_mixmodel']:
                    q_label = torch.arange(cfg.k_way).repeat_interleave(cfg.query).to(device)
                    s_label = torch.arange(cfg.k_way).to(device)
                    pred,loss = model(x, [s_label, q_label])

                elif cfg.fs_head in ['MetaOpt']:
                    q_label = torch.arange(cfg.k_way).repeat_interleave(cfg.query).to(device)
                    s_label = torch.arange(cfg.k_way).repeat_interleave(cfg.n_shot).to(device)
                    pred,loss = model(x, [s_label, q_label])
                else:
                    pred,loss=model(x, q_label)
        
        if mode=='train':
            summary['loss']+=[loss.item()]
        
        if mode=='train':
            batch_cfm=cal_cfm(pred, q_label, ncls=cfg.k_way)
            batch_acc=np.trace(batch_cfm)/np.sum(batch_cfm)
            summary['acc'].append(batch_acc)
            if i%show_interval==0:
                bar.set_description("Loss: %.3f"%(np.mean(summary['loss'])))
                # bar.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))
            
            train_confusion_mat+=batch_cfm
        else:
            
            batch_cfm=cal_cfm(pred, q_label, ncls=cfg.k_way)
            batch_acc=np.trace(batch_cfm)/np.sum(batch_cfm)
            test_accuracies.append(batch_acc)
            summary['acc'].append(batch_acc)
            if i%show_interval==0:
                bar.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))
            
            val_confusion_mat+=batch_cfm
    
    if mode!='train':
        summary['cfm']=val_confusion_mat

    return summary
            

    









if __name__=='__main__':
    main(cfg)
    