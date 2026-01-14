from __future__ import print_function
import torch.utils.data as data
import random
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from parameters import *


def get_breast_imagenet(root_dir, fold=1, split='train', target_transform=None):
    """
    Get breast cancer dataset similar to get_imagenet in dataloader.py
    Returns base dataset (transform will be applied in MultiFeatureDataset)
    """
    # Setup paths - fold structure
    fold_dir = os.path.join(root_dir, f'fold{fold}')
    split_root = os.path.join(fold_dir, split)
    
    # Use ImageFolder WITHOUT transform (will apply in MultiFeatureDataset for 5 views)
    # Same structure as get_imagenet but transform is None
    dataset = datasets.ImageFolder(
        root=split_root,
        transform=None,  # Will apply in MultiFeatureDataset
        target_transform=target_transform
    )
    
    return dataset


class MultiFeatureDataset(data.Dataset):
    """
    Wrapper to return 5 augmented versions of each image for fusion
    Applies transforms similar to get_imagenet but 5 times
    """
    def __init__(self, base_dataset, split='train'):
        self.base_dataset = base_dataset
        self.split = split
        
        # Setup transforms - same as get_imagenet
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:  # val or test
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __getitem__(self, index):
        """
        Returns 5 augmented versions of the same image
        Each call to transform will produce different augmentation due to randomness
        """
        # Get original PIL image and label from ImageFolder
        # ImageFolder returns (PIL Image, label) when transform=None
        img, label = self.base_dataset[index]
        
        # Generate 5 different augmentations
        # Each transform call will produce different result due to RandomResizedCrop etc.
        images = []
        for _ in range(5):
            transformed_img = self.transform(img)
            images.append(transformed_img)
        
        # Return as tuple: (x0, x1, x2, x3, x4), label
        return tuple(images), label
    
    def __len__(self):
        return len(self.base_dataset)


def get_breast_loader(root_dir, fold=1, split='train', batch_size=32, num_workers=4):
    """
    Get DataLoader for breast cancer dataset - similar to get_loader in dataloader.py
    """
    # Get base dataset using ImageFolder (like get_imagenet)
    base_dataset = get_breast_imagenet(root_dir, fold=fold, split=split)
    
    # Wrap with MultiFeatureDataset to get 5 augmented versions
    dataset = MultiFeatureDataset(base_dataset, split=split)
    
    # Setup sampler - same as get_loader
    if split == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    # Create DataLoader - same as get_loader
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return loader


if __name__ == '__main__':
    # Test the dataloader
    root_path = '/content/drive/MyDrive/NoisyNN-Breast/combined-breast-3fold'
    train_loader = get_breast_loader(root_path, fold=1, split='train', batch_size=4)
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Number of images per sample: {len(images)}")
        print(f"  Image shapes: {[img.shape for img in images]}")
        print(f"  Labels: {labels}")
        if i >= 2:
            break

