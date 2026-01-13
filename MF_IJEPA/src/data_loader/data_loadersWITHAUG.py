import imp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from .bp4d import BP4D

import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.masks.multiblock import MaskCollator as MBMaskCollator
import cv2
class AUDataLoader:
    def __init__(self, crop_size, patch_size, pred_mask_scale, enc_mask_scale, aspect_ratio,
                 num_enc_masks, num_pred_masks, allow_overlap, min_keep,
                 batch_size, num_workers, csv_root, modalities,
                 texture_root, depth_root, thermal_root):
        
        # 🔹 self’e atamalar
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.pred_mask_scale = pred_mask_scale
        self.enc_mask_scale = enc_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.allow_overlap = allow_overlap
        self.min_keep = min_keep
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_root = csv_root
        self.modalities = modalities
        self.texture_root = texture_root
        self.depth_root = depth_root
        self.thermal_root = thermal_root

        input_size_ = 224
        self.transform_train = {
            'rgb':A.Compose([
                
                A.RandomResizedCrop(
                            size=(input_size_, input_size_),                           
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
                            size=(input_size_, input_size_), 
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
            'rgb': A.Compose([
                A.Resize(input_size_, input_size_),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ]),
            'gray': A.Compose([
                A.Resize(input_size_, input_size_),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN[0], std=IMAGENET_DEFAULT_STD[0]),
                ToTensorV2()
            ])
        }

    def _get_data_loaders(self, dataset, num_workers=1, training=True):
        """
        mask_collator = MBMaskCollator(
            input_size=self.crop_size,
            patch_size=self.patch_size,
            pred_mask_scale=self.pred_mask_scale,
            enc_mask_scale=self.enc_mask_scale,
            aspect_ratio=self.aspect_ratio,
            nenc=self.num_enc_masks,
            npred=self.num_pred_masks,
            allow_overlap=self.allow_overlap,
            min_keep=self.min_keep
        )
        """
        
        if training:
            train_sampler = RandomSampler(dataset)
            return DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        else:
            test_sampler = SequentialSampler(dataset)
            return DataLoader(dataset,
                                sampler=test_sampler,
                                batch_size=self.batch_size,
                                num_workers=num_workers,
                                pin_memory=True)

    def get_data_loaders(self):
        CLASS = BP4D
        csv_path = self.csv_root


        train_dataset = CLASS(os.path.join(csv_path, "busi_fold1_train.csv"), 
                                 self.modalities,
                                 self.texture_root, 
                                 self.depth_root,
                                 self.thermal_root,
                                 self.transform_train)
        validate_dataset = CLASS(os.path.join(csv_path, "busi_fold1_val.csv"),
                                 self.modalities,
                                 self.texture_root, 
                                 self.depth_root,
                                 self.thermal_root,
                                 self.transform_val)
        test_dataset = CLASS(os.path.join(csv_path, "busi_fold1_test.csv"),
                                 self.modalities,
                                 self.texture_root, 
                                 self.depth_root,
                                 self.thermal_root,
                                self.transform_val)

        
        train_loader = self._get_data_loaders(train_dataset, self.num_workers, True)
        validate_loader = self._get_data_loaders(validate_dataset,  self.num_workers, False)
        test_loader = self._get_data_loaders(test_dataset, self.num_workers,False)


        return train_loader, validate_loader, test_loader


