


#####################################only bmode finetune################################################################################


import torch
import torch.distributed as dist
from .base import BaseTrainer
from srcs.utils import inf_loop, collect
from srcs.logger import BatchMetrics, MetricLogger, SmoothedValue
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import time
import csv
import datetime
from typing import Iterable, Optional
import re
import pandas as pd
import numpy as np
import math
import sys, os
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, data_loader,
                 val_data_loader=None, test_loader=None, lr_adjust=None, loss_scaler=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.train_criterion, self.validate_criterion = [c.to(self.device) for c in criterions]
        
        self.len_epoch = len(self.data_loader)
        
        self.log_step = self.log_step = config.print_freq
        # print('len_epoch: {:f} log_step: {:f}'.format(self.len_epoch, self.log_step))
        self.val_data_loader = val_data_loader
        self.test_loader = test_loader
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
        # metric_logger.add_meter('accuracy', SmoothedValue(window_size=1, fmt='{value:.2f}'))
        
        for batch_idx, out_dic in enumerate(metric_logger.log_every(self.data_loader, self.log_step, f'Epoch:[{epoch}]')):
            
            if batch_idx % accum_iter == 0:
                self.lr_adjust(epoch + batch_idx / len(self.data_loader))

            depth, targets, name = out_dic['depth'], out_dic['label'], out_dic['name']
            
            targets = targets.to(self.device, non_blocking=True)
            targets = targets.squeeze(1).long()
            #print("TARGETS*******************",targets)
            #print("NAMES**********************",name)
            #targets = torch.cat((targets, 1-targets), dim=1)
            #targets = torch.as_tensor(targets, dtype=torch.long).squeeze()
           # targets = targets.argmax(dim=1)
            
           
            depth = depth.to(self.device, non_blocking=True)
        
            with torch.autocast("cuda"):
                outputs = self.model(depth, self.config.channel_mixing)
                
                #print(outputs.shape, targets.shape)
                loss = self.train_criterion(outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            self.loss_scaler(loss, self.optimizer, clip_grad=None,
                        parameters=self.model.parameters(), create_graph=False,
                        update_grad=(batch_idx + 1) % accum_iter == 0)
            if (batch_idx + 1) % accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            min_lr = 10.
            max_lr = 0.
            for group in self.optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            self.train_metrics.update('lr', max_lr)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            loss = loss.detach().cpu().item()   #cc
          
            self.train_metrics.update('loss',loss)   #cc

            # preds = torch.where(outputs.sigmoid()>0.5, 1.0, 0.0)
            # preds = outputs
            preds = torch.argmax(outputs, dim=1)
            preds = preds.detach().cpu().numpy()
            y = targets.detach().cpu().numpy()

            # for met in self.metric_ftns:
            met = self.metric_ftns[0]
            f1_arr, mean_f1, _, mean_prec, _, mean_recall = met(preds, y)
            # accuracy = met(preds, y)
            # mean_f1 = accuracy
            metric = mean_f1 # average metric between processes
            self.train_metrics.update(met.__name__, metric)
            
            # update logger
            metric_logger.update(lr=max_lr)
            metric_logger.update(loss=loss_value)   
            metric_logger.update(F1=torch.tensor(mean_f1))
        
        self.logger.info((f"Averaged: {metric_logger}"))
        log = self.train_metrics.result()

        if (self.val_data_loader is not None) and (self.device==0):
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

    
        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k+'/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.logger.info('--'*55)
        metric_logger = MetricLogger(self.logger, delimiter="  ")
        # metric_logger.add_meter('accuracy', SmoothedValue(window_size=1, fmt='{value:.2f}'))
        self.model.eval()
        self.valid_metrics.reset()

        all_preds, all_label = [], []
        with torch.no_grad():
            all_preds, all_label = [], []
            for batch_idx, out_dic in enumerate(metric_logger.log_every(self.val_data_loader, 200, f'Test [{epoch}]')):
                
                depth, targets = out_dic['depth'], out_dic['label']
                
                targets = targets.to(self.device, non_blocking=True)
                targets = targets.squeeze(1).long()
                #targets = torch.cat((targets, 1-targets), dim=1)
               # targets = torch.as_tensor(targets, dtype=torch.long).squeeze()
                 # targets = targets.argmax(dim=1)
              
                
                depth = depth.to(self.device, non_blocking=True)
                
                # compute output
                output = self.model(depth, self.config.channel_mixing)
               # print("OUTPUT******************************************", output)
                loss = self.validate_criterion(output, targets)
                
                loss = loss.detach().cpu().item()  # CPU + float
                self.valid_metrics.update('loss',loss)
                metric_logger.update(loss=loss)

                # preds = torch.where(output.sigmoid()>0.5, 1.0, 0.0)
                preds = torch.argmax(output, dim=1)
                # preds = output
                
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(targets.detach().cpu().numpy())

                # if (batch_idx % 200 == 0) or (batch_idx == len(self.val_data_loader) - 1):
                #     self.logger.info(f'Test: Epoch: {epoch} {self._progress(batch_idx, self.val_data_loader)} Loss: {loss.item():.6f}')
                
            
            all_preds, all_label = np.concatenate(all_preds, 0), np.concatenate(all_label, 0)
            
            f1_arr, mean_f1, _, mean_prec, _, mean_recall = self.metric_ftns[0](all_preds, all_label)
            accuracy = self.metric_ftns[1](all_preds, all_label)
            self.valid_metrics.update('F1', mean_f1)
            self.valid_metrics.update('accuracy', accuracy)
            metric_logger.update(F1=torch.tensor(mean_f1))
            # metric_logger.update(F1=torch.tensor(accuracy))

            # labels = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU14','AU15','AU17','AU23','AU24']
            # labels = ['0','1']
            # for j in range(len(f1_arr)):
            #     au_f1 = f1_arr[j]
            #     au_name = labels[j]
            #     self.valid_metrics.update(au_name, au_f1)
            
            self.logger.info(f'Epoch [{epoch}] Test F1 ===> {mean_f1:.5f}')
     
    
        if (epoch+1) == self.epochs:
            #csv_file = open("/IMAGE_LEVEL_CHANNEL-MIX.csv", mode='w', newline='')
            #writer = csv.writer(csv_file)
            #header_csv = ["Patient ID", "True Diagnosis","Final Diagnosis","pred_prob_class_0", "pred_prob_class_1"]
            #writer.writerow(header_csv)
            print(self.valid_metrics.result())
            all_preds, all_label = [], []            
            checkpoint_dir = "/home/ec2-user/SageMaker/channel-mix/Mul-AE/cp/model_best.pth"
            checkpoint = torch.load(checkpoint_dir, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.eval()
            all_preds, all_label = [], []
            with torch.no_grad():
                all_preds, all_label,all_names,all_names2,all_probs  = [], [], [], [], []
                for batch_idx, out_dic in enumerate(self.test_loader):
                    depth, targets, name = out_dic['depth'], out_dic['label'], out_dic['name']
                    
                   # print("*****************TARGET**************",targets)
                    targets = targets.to(self.device, non_blocking=True)
                    targets = targets.squeeze(1).long()
                     # targets = torch.cat((targets, 1-targets), dim=1)
                    #  targets = torch.as_tensor(targets, dtype=torch.long).squeeze()
                     # targets = targets.argmax(dim=1)
                   

                    depth =depth.to(self.device, non_blocking=True)

                    # compute output
                    output = self.model(depth, self.config.channel_mixing)
                    print("OUTPUT******************************************", output)
                   
                    # preds = torch.where(output.sigmoid()>0.5, 1.0, 0.0)
                    preds = torch.argmax(output, dim=1)
                    # preds = output
                    probs = F.softmax(output, dim=1)

                    all_preds.append(preds.detach().cpu().numpy())
                    all_label.append(targets.detach().cpu().numpy())
                    all_probs.append(probs.detach().cpu().numpy())
                
                    all_names.extend(name)

                    # if (batch_idx % 200 == 0) or (batch_idx == len(self.val_data_loader) - 1):
                    #     self.logger.info(f'Test: Epoch: {epoch} {self._progress(batch_idx, self.val_data_loader)} Loss: {loss.item():.6f}')

                all_preds, all_label, all_probs = np.concatenate(all_preds, 0), np.concatenate(all_label, 0), np.concatenate(all_probs, 0)
                
                """
                for img_name, pred, true_label, prob in zip(all_names, all_preds, all_label, all_probs):
                            row = [img_name, true_label, pred, prob[0], prob[1]]
                            writer.writerow(row)
                """
                accuracy = np.mean(all_preds == all_label) * 100  
                print(f"ACCC: {accuracy:.2f}%")

                # Pozitif sınıf olasılıkları
                #print("ALL LABEL", all_label)
                y_true = all_label.flatten()
                #print("Y TRUE", y_true)
                #print("ALL PROBS", all_probs)
                y_score = all_probs[:, 1]   # class 1'in olasılığı
                #print(y_score)
                auc = roc_auc_score(y_true, y_score)
                print(f"AUC: {auc:.4f}")
                
                f1_arr, mean_f1, _, mean_prec, _, mean_recall = self.metric_ftns[0](all_preds, all_label)
                accuracy2 = self.metric_ftns[1](all_preds, all_label)
                print(f"ACCC2: {accuracy2:.2f}%")
                conf_matrix = confusion_matrix(all_label, all_preds)
                print("Confusion Matrix:")
                print(conf_matrix)
                #self.valid_metrics.update('F1', mean_f1)
                #self.valid_metrics.update('accuracy', accuracy)
                #metric_logger.update(F1=torch.tensor(mean_f1))
                # metric_logger.update(F1=torch.tensor(accuracy))

                # labels = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU14','AU15','AU17','AU23','AU24']
                # labels = ['0','1']
                # for j in range(len(f1_arr)):
                #     au_f1 = f1_arr[j]
                #     au_name = labels[j]
                #     self.valid_metrics.update(au_name, au_f1)

                #self.logger.info(f'Epoch [{epoch}] LAST F1 ===> {mean_f1:.5f}')
              
        
        return self.valid_metrics.result()