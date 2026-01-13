from turtle import shape
import torch
import torch.distributed as dist
from .base import BaseTrainer
from srcs.utils import inf_loop, collect, util
from srcs.logger import BatchMetrics, MetricLogger, SmoothedValue
import time
import datetime
import numpy as np
import math
import sys

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, data_loader,
                 val_data_loader=None, lr_adjust=None, loss_scaler=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        
        self.len_epoch = len(self.data_loader)
        
        self.log_step = self.log_step = config.print_freq
        # print('len_epoch: {:f} log_step: {:f}'.format(self.len_epoch, self.log_step))
        self.val_data_loader = val_data_loader
        self.lr_adjust = lr_adjust
        self.loss_scaler = loss_scaler

        self.train_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/valid', writer=self.writer)

    


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        accum_iter = 1
        self.optimizer.zero_grad()

        metric_logger = MetricLogger(self.logger, delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
        
        for batch_idx, out_dic in enumerate(metric_logger.log_every(self.data_loader, self.log_step, f'Epoch:[{epoch}]')):
            
            if batch_idx % accum_iter == 0:
                self.lr_adjust(epoch + batch_idx / len(self.data_loader))

            texture = out_dic['texture']
            depth = out_dic['depth']
            thermal = out_dic['thermal']
            img_name = out_dic['name']

            texture = texture.to(self.device, non_blocking=True)
            depth = depth.to(self.device, non_blocking=True)
            thermal = thermal.to(self.device, non_blocking=True)

        
            with torch.cuda.amp.autocast():
                loss, x_mix, mask_mixing, pred, mask_l, mask = self.model(texture, depth, thermal, self.config.mask_ratio)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=(batch_idx + 1) % accum_iter == 0)
            if (batch_idx + 1) % accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss)
            

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        
            if self.device ==0 and (batch_idx + 0) % self.config.print_freq == 0:
        
                # save img
                pred = self.model.module.unpatchify(pred) # (N, C, H, W)

                mask_l = mask_l.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 16**2) # n l c d
                mask_l = self.model.module.unpatchify(torch.einsum('nlcd->ncld', mask_l))
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, 16**2)    # n c l d    
                mask = self.model.module.unpatchify(mask)  # (N, C, H, W)

                util.run_one_image(texture, depth, thermal, x_mix, pred, mask_l, mask, img_name, self.config.save_dir+f'/Epoch {epoch:03d}_{batch_idx:04d}.jpg')


        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        
        self.logger.info((f"Averaged: {metric_logger}"))

        self.train_metrics.update('lr', metric_logger.lr.value)
        self.train_metrics.update('loss', metric_logger.loss.value)

        log = self.train_metrics.result()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k+'/epoch', v)
        return log

