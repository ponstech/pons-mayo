import imp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from .bp4d import BP4D
from .disfa import DISFA
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import random
def seed_worker(worker_id):  #cc
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        


class AUDataLoader:
    def __init__(self, config):
        self.config = config
        
        self.transform_train = {
            'rgb':A.Compose([
                
                A.RandomResizedCrop(
                            size=(config.input_size, config.input_size),                           
                            scale=(0.3, 1.0),
                            interpolation=cv2.INTER_CUBIC  # torchvision'daki interpolation=3 ile aynı
                        ),
                A.HorizontalFlip(p=0.5),
                #A.Resize(config.input_size, config.input_size),
                
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD),
                ToTensorV2()]
            ),
            'gray':A.Compose([
               
                A.RandomResizedCrop(
                            size=(config.input_size, config.input_size), 
                            scale=(0.3, 1.0),
                            interpolation=cv2.INTER_CUBIC  # torchvision'daki interpolation=3 ile aynı
                        ),
                A.HorizontalFlip(p=0.5),
                #A.Resize(config.input_size, config.input_size),
                
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ],
            )

        }
        self.transform_val = {
            'rgb':A.Compose([
                A.Resize(config.input_size, config.input_size),   
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD),
                ToTensorV2()], 
            ),
            'gray':A.Compose([
                A.Resize(config.input_size, config.input_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ],
            )

        }
 

    def _get_data_loaders(self, dataset, batch_size, num_workers=1, training=True):
        g = torch.Generator()
        g.manual_seed(42)  # global deterministik seed
        if training:
            
            train_sampler = RandomSampler(dataset,generator=torch.Generator().manual_seed(42))
            return  DataLoader(dataset,
                                sampler=train_sampler,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=seed_worker,    
                                pin_memory=True, drop_last=False)
        else:
            test_sampler = SequentialSampler(dataset)
            return DataLoader(dataset,
                                sampler=test_sampler,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=seed_worker,   
                                pin_memory=True)
    
    def get_data_loaders(self):
        data_name = self.config.data_name
        batch_size = self.config.batch_size
        num_workers = self.config.workers
        training = self.config.training
        csv_root = self.config.csv_root
        
        # CLASS = Thermal
        CLASS = BP4D
        if data_name == 'DISFA':
            CLASS = DISFA
        if training:               
            if self.config.task == 'Finetune':
            
                train_dataset = CLASS(os.path.join(csv_root, f'{self.config.fold}_train.csv'),
                                self.config, self.transform_train)
                validate_dataset = CLASS(os.path.join(csv_root, f'{self.config.fold}_val.csv'),
                                self.config, self.transform_val)
                test_dataset = CLASS(os.path.join(csv_root, f'{self.config.fold}_test.csv'),
                                self.config, self.transform_val)
                
                # self.config.texture_root = '/home/jupyter/MultiMAE/breast_bus/enhanced/'
                # self.config.depth_root = '/home/jupyter/MultiMAE/breast_bus/bmode/'
                # self.config.thermal_root = '/home/jupyter/MultiMAE/breast_bus/improved/'
                # test_dataset = CLASS(os.path.join(csv_root, f'bus.csv'),
                #                 self.config, self.transform_val)
                
                # self.config.texture_root = '/home/jupyter/MultiMAE/BUSBRA/enhanced/'
                # self.config.depth_root = '/home/jupyter/MultiMAE/BUSBRA/bmode/'
                # self.config.thermal_root = '/home/jupyter/MultiMAE/BUSBRA/improved/'
                # test_dataset = CLASS(os.path.join(csv_root, f'busbra.csv'),
                                # self.config, self.transform_val)
                
                train_loader = self._get_data_loaders(train_dataset, batch_size, num_workers=num_workers, training=True)

                validate_loader = self._get_data_loaders(validate_dataset, batch_size, num_workers=num_workers, training=False) #batch_size*4 tü değiştim
                test_loader = self._get_data_loaders(test_dataset, batch_size, num_workers=num_workers, training=False)
                
                return train_loader, validate_loader, test_loader
            else:
                csv = f'{self.config.fold}_train.csv'
                train_dataset = CLASS(os.path.join(csv_root, csv),
                            self.config, self.transform_train)
                return self._get_data_loaders(train_dataset, batch_size, num_workers=num_workers, training=True)

        else:
            csv = 'sample.csv' if self.config.task == 'Visualize' else f'{self.config.fold}_test.csv'
            test_dataset = CLASS(os.path.join(csv_root, csv),
                            self.config, self.transform_val)
            return self._get_data_loaders(test_dataset, batch_size, num_workers=num_workers, training=False)


