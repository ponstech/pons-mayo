from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import RandAugment
import matplotlib.pyplot as plt
from parameters import *
import cv2
import pdb

def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

def get_imagenet(root, target_transform=None):
    transform_train = transforms.Compose([
        RandAugment(),
        transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tra_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')
    trainset = datasets.ImageFolder(root=tra_root,
                                    transform=transform_train,
                                    target_transform=target_transform)
    valset = datasets.ImageFolder(root=val_root,
                                  transform=transform_val,
                                  target_transform=target_transform)
    return trainset, valset

def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory=True)

    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.te_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

if __name__ == '__main__':
    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (imgs, labels) in enumerate(train_loader):
        print(i, labels)
        plt.imshow(np.transpose(imgs[0].numpy(), (1, 2, 0)))
        plt.show()


