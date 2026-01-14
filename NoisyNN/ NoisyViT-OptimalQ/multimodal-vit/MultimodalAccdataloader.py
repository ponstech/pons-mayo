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


def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

'''
def get_loader():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.ImageNet(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
    testset = datasets.ImageNet(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) 
    
    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.batch_size,
                             num_workers=4,
                             ) if testset is not None else None
    return train_loader, test_loader
'''

def get_imagenet(root, target_transform = None):
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        transform_train = transforms.Compose([
            #transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            #transforms.RandomRotation(10),        # Randomly rotate the image by up to 10 degrees
            RandAugment(),
            transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory = True,
                              #shuffle = True,  #ValueError: DistributedSampler option is mutually exclusive with shuffle
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.te_batch_size,
                             num_workers=4,
                             pin_memory = True,
                             #shuffle = True
                             ) if testset is not None else None
    return train_loader, test_loader


class MultiModalDataset(data.Dataset):
    def __init__(self, roots, split='train', transform=None):
        self.roots = roots
        self.split = split
        self.transform = transform
        self.modalities = list(roots.keys())
        self.samples = []
        
        # Debug: roots bilgilerini yazdır
        print(f"MultiModalDataset initialization:")
        print(f"Split: {split}")
        print(f"Modalities: {self.modalities}")
        for mod, path in roots.items():
            print(f"{mod}: {path}")
            split_path = os.path.join(path, split)
            print(f"  {mod} split path: {split_path}")
            print(f"  {mod} split exists: {os.path.exists(split_path)}")
            if os.path.exists(split_path):
                classes = os.listdir(split_path)
                print(f"  {mod} classes: {classes}")
                if classes:
                    sample_class = classes[0]
                    class_path = os.path.join(split_path, sample_class)
                    files = os.listdir(class_path)
                    print(f"  {mod} sample class {sample_class} files: {files[:5]}...")  # İlk 5 dosya
        
        base_root = os.path.join(roots['bmode'], split)
        if not os.path.exists(base_root):
            print(f"ERROR: Base root {base_root} does not exist!")
            return
            
        for class_name in sorted(os.listdir(base_root)):
            class_dir = os.path.join(base_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
                    continue
                
                # Dosya isimlerini modaliteye göre oluştur
                base_name = os.path.splitext(fname)[0]  # uzantı olmadan dosya adı
                ext = os.path.splitext(fname)[1]  # uzantı
                
                paths = {}
                for mod in self.modalities:
                    if mod == 'bmode':
                        paths[mod] = os.path.join(roots[mod], split, class_name, fname)
                    elif mod == 'enhanced':
                        enhanced_fname = f"{base_name}_enhancement{ext}"
                        paths[mod] = os.path.join(roots[mod], split, class_name, enhanced_fname)
                    elif mod == 'improved':
                        improved_fname = f"{base_name}_improvement{ext}"
                        paths[mod] = os.path.join(roots[mod], split, class_name, improved_fname)
                    else:
                        paths[mod] = os.path.join(roots[mod], split, class_name, fname)
                
                # Debug: Her dosya için path kontrolü
                all_exist = all(os.path.exists(p) for p in paths.values())
                if not all_exist:
                    print(f"Missing files for {fname}:")
                    for mod, p in paths.items():
                        if not os.path.exists(p):
                            print(f"  {mod}: {p} - NOT FOUND")
                if all_exist:
                    self.samples.append((paths, class_name))
        
        print(f"Total samples found: {len(self.samples)}")
        if self.samples:
            print(f"Sample paths example:")
            sample_paths, sample_class = self.samples[0]
            for mod, path in sample_paths.items():
                print(f"  {mod}: {path}")
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(base_root)))}
        print(f"Class to index mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, class_name = self.samples[idx]
        # bmode ve improved grayscale, enhanced RGB
        bmode_img = cv2.imread(paths['bmode'], cv2.IMREAD_GRAYSCALE)
        improved_img = cv2.imread(paths['improved'], cv2.IMREAD_GRAYSCALE)
        enhanced_img = cv2.imread(paths['enhanced'], cv2.IMREAD_COLOR)
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        # Dosya okuma kontrolü
        if bmode_img is None:
            print(f"ERROR: Could not read bmode image: {paths['bmode']}")
            # Hata durumunda basit bir görsel oluştur
            bmode_img = np.zeros((224, 224), dtype=np.uint8)
        if improved_img is None:
            print(f"ERROR: Could not read improved image: {paths['improved']}")
            improved_img = np.zeros((224, 224), dtype=np.uint8)
        if enhanced_img is None:
            print(f"ERROR: Could not read enhanced image: {paths['enhanced']}")
            enhanced_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        # Tüm görselleri 224x224 boyutuna resize et
        target_size = (224, 224)
        bmode_img = cv2.resize(bmode_img, target_size, interpolation=cv2.INTER_AREA)
        improved_img = cv2.resize(improved_img, target_size, interpolation=cv2.INTER_AREA)
        enhanced_img = cv2.resize(enhanced_img, target_size, interpolation=cv2.INTER_AREA)

        # Her modalite için uygun transform uygula
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
        else:
            # Transform yoksa manuel olarak tensöre çevir
            bmode_img = torch.from_numpy(bmode_img).float() / 255.0
            if bmode_img.ndim == 2:
                bmode_img = bmode_img.unsqueeze(0)
            improved_img = torch.from_numpy(improved_img).float() / 255.0
            if improved_img.ndim == 2:
                improved_img = improved_img.unsqueeze(0)
            enhanced_img = torch.from_numpy(enhanced_img).float() / 255.0
            enhanced_img = enhanced_img.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        
       # Birleştir: [1,H,W] + [1,H,W] + [3,H,W] = [5,H,W]
        x = torch.cat([bmode_img, improved_img, enhanced_img], dim=0)  # [5,H,W]
        
        # 5 kanalı 3 kanala basit dönüştürme (gradient gerektirmez)
        # Kanal 1: bmode + improved (grayscale)
        # Kanal 2: enhanced'ın ilk kanalı (R)
        # Kanal 3: enhanced'ın ikinci kanalı (G)
        x_3ch = torch.zeros(3, x.shape[1], x.shape[2])
        x_3ch[0] = (x[0] + x[1]) / 2.0  # bmode + improved ortalaması
        x_3ch[1] = x[2]  # enhanced R kanalı
        x_3ch[2] = x[3]  # enhanced G kanalı
        
        label = self.class_to_idx[class_name]
        return x_3ch, label

def get_multimodal_loaders(roots, batch_size, num_workers=4):
    """
    Hem train hem test için multimodal loader'ları döndür
    """
    train_dataset = MultiModalDataset(roots, split='train', transform=None)
    test_dataset = MultiModalDataset(roots, split='test', transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, train_dataset, test_dataset

# Eski fonksiyonu da tut (geriye uyumluluk için)
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



