from __future__ import print_function
from re import L
import re
from parameters import *
import torch.utils.data as data
import random
import os
import numpy as np
import torch
import albumentations as A
import cv2
import pdb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import RandAugment
from torch.utils.data import DataLoader

def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img




def get_imagenet(root, target_transform = None):
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            transforms.RandomRotation(10),        # Randomly rotate the image by up to 10 degrees
            transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandAugment() 
        ])

        transform_val = transforms.Compose([
            transforms.Resize((opt.res, opt.res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tra_root = os.path.join(root,'train')
        trainset = datasets.ImageFolder(root=tra_root,
                                transform=transform_train,
                                target_transform=target_transform)
        val_root = os.path.join(root,'val')
        valset = datasets.ImageFolder(root=val_root,
                                transform=transform_val,
                                target_transform=target_transform)
        return trainset,valset


class MultiModalDataset(data.Dataset):
    def __init__(self, roots, split='train', transform=None):
        """
        roots: dict, ör: {'bmode': '/path/bmode', 'enhanced': '/path/enhanced', 'improved': '/path/improved'}
        split: 'train' veya 'test'
        transform: torchvision transform
        """
        self.roots = roots
        self.split = split
        self.transform = transform
        self.modalities = list(roots.keys())
        # Her modalite için klasörleri bul
        self.samples = []
        # bmode klasörünü referans al
        base_root = os.path.join(roots['bmode'], split)
        for class_name in sorted(os.listdir(base_root)):
            class_dir = os.path.join(base_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
                    continue
                # Her modalite için dosya yollarını oluştur
                paths = {mod: os.path.join(roots[mod], split, class_name, fname) for mod in self.modalities}
                # Hepsi var mı kontrol et
                if all(os.path.exists(p) for p in paths.values()):
                    self.samples.append((paths, class_name))
        # Sınıf isimlerini ve indexlerini bul
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(base_root)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, class_name = self.samples[idx]
        # bmode ve improved grayscale, enhanced RGB
        bmode_img = cv2.imread(paths['bmode'], cv2.IMREAD_GRAYSCALE)
        improved_img = cv2.imread(paths['improvement'], cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(paths['enhancement'], cv2.IMREAD_COLOR)
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        # Transform uygula
        if self.transform:
            # Grayscale img'leri [H,W] -> [1,H,W] ve tensöre çevir
            bmode_img = self.transform(bmode_img)
            if bmode_img.ndim == 2:
                bmode_img = bmode_img.unsqueeze(0)
            improved_img = self.transform(improved_img)
            if improved_img.ndim == 2:
                improved_img = improved_img.unsqueeze(0)
            # enhanced zaten [3,H,W] olacak
            enhanced_img = self.transform(enhanced_img)
        # Birleştir
        x = torch.cat([bmode_img, improved_img, enhanced_img], dim=0)
        label = self.class_to_idx[class_name]
        return x, label


def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = DistributedSampler(trainset) 
    test_sampler = DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory = True,
                              #shuffle = True,  #ValueError: sampler option is mutually exclusive with shuffle
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.te_batch_size,
                             num_workers=4,
                             pin_memory = True,
                             #shuffle = True
                             ) if testset is not None else None
    return train_loader, test_loader


def get_multimodal_loader(roots, batch_size, split='train', transform=None, num_workers=4):
    dataset = MultiModalDataset(roots, split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)
    return loader, dataset


if __name__ == '__main__':

    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()



