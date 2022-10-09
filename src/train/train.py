import os
import tempfile
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union
from torch.cuda.amp import autocast, GradScaler

import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchinfo import summary as TISummary
import numpy as np
from datetime import datetime
from tqdm import trange
from tqdm import tqdm, tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from logging import handlers

# from PyUtils.PLModuleInterface import PLMInterface, GraphCallback
from PyUtils.logs.print import *

import os
import argparse
import torch
from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

import torch.utils.tensorboard.writer as TBWriter
from prettytable import PrettyTable

from PyUtils.utils.meter import AverageMeter
from PyUtils.pytorch.callback import LogCallback, TrainCallback, GraphCallback, Callback
from PyUtils.pytorch.module import TrainModule, Trainer

from src.nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from src.loss.centernet_training import get_lr_scheduler, set_optimizer_lr
from src.dataset.dataloader import CenternetDataset
from src.dataset.utils import download_weights, get_classes, show_config
from src.loss.centernet_training import focal_loss, reg_l1_loss
from src.config.config_v1 import CONFIGS



class LRSchedulerCB(Callback):
    def on_train_epoch_end(self, local_rank, epoch, module: "TrainModule", trainer: "Trainer"):
        is_change = False
        epoch = trainer.current_epoch - trainer.start_epoch
        if epoch >= 0 and epoch < 10:
            learning_rate=0.0001  #0.01
            is_change = True
        if epoch >= 10 and epoch < 30:
            learning_rate=0.00001  #0.001
            is_change = True
        if epoch >= 30:
            learning_rate=0.000001  #0.0001
            is_change = True
        if is_change:
            for param_group in module.optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f'[************]  Learning Rate for this epoch=[{epoch}/{trainer.configs.TRAIN_START_EPOCHS}]  learning_rate: {module.optimizer.param_groups[0]["lr"]} --> {learning_rate}')
        else:
            print(f'[************]  Learning Rate for this epoch=[{epoch}/{trainer.configs.TRAIN_START_EPOCHS}]  {module.optimizer.param_groups[0]["lr"]=}')


class SaveModelCB(Callback):
    def on_train_epoch_end(self, local_rank, epoch, module: "TrainModule", trainer: "Trainer"):
        if local_rank == 0 and epoch % module.configs.TRAIN_SAVE_MODEL_INTERVAL == 0:
            torch.save(module.model.module.state_dict(), os.path.join(f'{module.configs.CHECKPOINT_DIR}', f'{epoch}.pth'))


class CenterNetTrainModule(TrainModule):

    def __init__(self, configs=None, pretrained=None):
        super(CenterNetTrainModule, self).__init__(pretrained)
        self.configs = configs
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None
        self.val_dataset = None
        self.val_sampler = None
        self.val_loader = None
        self.test_dataset = None
        self.test_sampler = None
        self.test_loader = None
        self.pretrained = pretrained

    def create_model(self, local_rank):
        # create local model
        if self.configs.NET_STRUCETURE == 'resnet':
            self.model = CenterNet_Resnet50(self.configs.VOC_CLS_NUM)

            # resnet = models.resnet50(pretrained=True)
            # new_state_dict = resnet.state_dict()
            # dd = self.model.state_dict()
            # for k in new_state_dict.keys():
            #     print(k)
            #     if k in dd.keys() and not k.startswith('fc'):
            #         print('yes')
            #         dd[k] = new_state_dict[k]
            # self.model.load_state_dict(dd)
        else:
            self.model = CenterNet_HourglassNet({'hm': self.configs.VOC_CLS_NUM, 'wh': 2, 'reg':2})
        
        if self.pretrained and os.path.exists(self.pretrained) and local_rank == 0:
            sllog << f'[------------][{local_rank}]  loading pre-trained model[{self.pretrained}] ...'
            # self.model.load_state_dict(torch.load(pretrained).module.state_dict())
            self.model.load_state_dict(torch.load(self.pretrained))
            sllog << f'[------------][{local_rank}]  load pre-trained model complete.'
            
        self.model.to(local_rank)
        dist.barrier()

    def create_loss(self):
        ...

    def create_optim(self, model):
        
        nbs             = 64
        lr_limit_max    = 5e-3 if self.configs.TRAIN_OPTIM_TYPE == 'adam' else 5e-2
        lr_limit_min    = 2.5e-4 if self.configs.TRAIN_OPTIM_TYPE == 'adam' else 5e-4
        Init_lr_fit     = min(max(self.configs.TRAIN_BATCH_SIZE / nbs * self.configs.TRAIN_LR_INIT, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(self.configs.TRAIN_BATCH_SIZE / nbs * self.configs.TRAIN_LR_END, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        self.optimizer = {
            'adam'  : optim.Adam(self.model.parameters(), Init_lr_fit, betas = (self.configs.TRAIN_OPTIM_MOMENTUM, 0.999), weight_decay = self.configs.TRAIN_OPTIM_WEIGHT_DECAY),
            'sgd'   : optim.SGD(self.model.parameters(), Init_lr_fit, momentum = self.configs.TRAIN_OPTIM_MOMENTUM, nesterov=True, weight_decay = self.configs.TRAIN_OPTIM_WEIGHT_DECAY)
        }[self.configs.TRAIN_OPTIM_TYPE]
        self.lr_scheduler = None

    def create_data_loader(self):
        
        with open(self.configs.TRAIN_DATA_FILES) as f:
            train_lines = f.readlines()
        num_train   = len(train_lines)

        self.train_dataset   = CenternetDataset(train_lines, self.configs.TRAIN_INPUT_SHAPE, self.configs.VOC_CLS_NUM, train = True)
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.configs.TRAIN_BATCH_SIZE,
            num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            sampler=self.train_sampler,
            collate_fn=self.train_dataset.dataset_collate
        )
        
        with open(self.configs.VAL_DATA_FILES) as f:
            val_lines   = f.readlines()
        num_val     = len(val_lines)
        self.val_dataset = CenternetDataset(val_lines, self.configs.TRAIN_INPUT_SHAPE, self.configs.VOC_CLS_NUM, train = True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.configs.TRAIN_BATCH_SIZE,
            num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            sampler=self.val_sampler,
            collate_fn=self.val_dataset.dataset_collate
        )

        with open(self.configs.TEST_DATA_FILES) as f:
            test_lines = f.readlines()
        self.test_dataset = CenternetDataset(test_lines, self.configs.TRAIN_INPUT_SHAPE, self.configs.VOC_CLS_NUM, train = True)
        self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.configs.TEST_BATCH_SIZE,
            num_workers=self.configs.TRAIN_NUMBER_WORKERS,
            sampler=self.test_sampler,
            collate_fn=self.test_dataset.dataset_collate
        )

    def train_step(self, batch_idx, batch, local_rank):
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        batch_images = batch_images.to(self.model.device)
        batch_hms = batch_hms.to(self.model.device)
        batch_whs = batch_whs.to(self.model.device)
        batch_regs = batch_regs.to(self.model.device)
        batch_reg_masks = batch_reg_masks.to(self.model.device)

        hm, wh, offset = self.model(batch_images)
        c_loss          = focal_loss(hm, batch_hms)
        wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        
        # loss            = c_loss + wh_loss + off_loss
        loss            = (c_loss*c_loss + wh_loss*wh_loss + off_loss*off_loss) / (c_loss + wh_loss + off_loss)

        return {'loss': loss, 'c_loss': c_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}

    def train_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    def train_epoch_end(self):
        ...
        
    @torch.no_grad()
    def eval_step(self, batch_idx, batch, local_rank):
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        batch_images = batch_images.to(self.model.device)
        batch_hms = batch_hms.to(self.model.device)
        batch_whs = batch_whs.to(self.model.device)
        batch_regs = batch_regs.to(self.model.device)
        batch_reg_masks = batch_reg_masks.to(self.model.device)

        hm, wh, offset = self.model(batch_images)
        c_loss          = focal_loss(hm, batch_hms)
        wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        
        loss            = c_loss + wh_loss + off_loss

        return {'eval_loss': loss, 'c_loss': c_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}

    def eval_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def test_step(self, batch_idx, batch, local_rank):
        # images, targets, y_trues = batch[0], batch[1], batch[2]

        # outputs = self.model(images.to(local_rank))
        # loss_all  = 0
        # losses = [0, 0, 0]
        # for l, output in enumerate(outputs):
        #     loss_item = self.criterion(
        #         l,
        #         output,
        #         targets,
        #         y_trues[l],
        #         batch
        #     )
        #     loss_all  += loss_item
        #     losses[l] += loss_item.item()
        # return {'loss': loss_all, 'loss1':losses[0], 'loss2':losses[1], 'loss3':losses[2]}
        return {'test_loss': torch.FloatTensor([0])}
    

    def test_step_end(self, step_output: Union[None, torch.Tensor, dict], batch_idx=None, batch=None, local_rank=None):
        ...

    @torch.no_grad()
    def predict(self, images, device_id):
        ...

    def predict_end(self, predicts, device_id):
        ...

    def set_callbacks(self):
        graph_cb = TrainCallback(
            interval=30, 
            log_dir=self.configs.LOGS_DIR, 
            dummy_input=np.zeros(shape=(2, 3, 446, 446))
        )
        log_cbs = LogCallback(
            train_meters={
                'loss': AverageMeter(name='loss', fmt=':4f'),
                'c_loss': AverageMeter(name='c_loss', fmt=':4f'),
                'wh_loss': AverageMeter(name='wh_loss', fmt=':4f'),
                'off_loss': AverageMeter(name='off_loss', fmt=':4f')
            }, 
            val_meters={
                'eval_loss': AverageMeter(name='loss', fmt=':4f'),
                'c_loss': AverageMeter(name='c_loss', fmt=':4f'),
                'wh_loss': AverageMeter(name='wh_loss', fmt=':4f'),
                'off_loss': AverageMeter(name='off_loss', fmt=':4f')
            }, 
            test_meters={
                'test_loss': AverageMeter(name='loss', fmt=':4f'),
                'c_loss': AverageMeter(name='c_loss', fmt=':4f'),
                'wh_loss': AverageMeter(name='wh_loss', fmt=':4f'),
                'off_loss': AverageMeter(name='off_loss', fmt=':4f')
            }, 
            log_dir=self.configs.LOGS_DIR, log_surfix='default',
            interval=30
        )
        lr_cb = LRSchedulerCB()
        save_model = SaveModelCB()
        return [graph_cb, log_cbs, lr_cb, save_model]



if __name__=="__main__":
    train_module = CenterNetTrainModule(configs=CONFIGS, pretrained=os.path.join(CONFIGS.TRAIN_PRETRAINED_DIR, CONFIGS.TRAIN_PRETRAINED_MODEL_NAME))
    
    trainer = Trainer(
        train_module=train_module, 
        configs=CONFIGS,
        mode='train', 
        accelarate=1, 
        precision=False,
        grad_average=False,
        sync_bn=True,
        limit_train_iters=10,
        limit_val_iters=10,
        limit_test_iters=10
    )
    trainer.fit()
