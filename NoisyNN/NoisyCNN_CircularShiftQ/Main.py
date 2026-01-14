#Developer: Shawey
#Date: 03/16/2023
import CNNs 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os,random
import numpy as np
from torch.utils.data import DataLoader
import ssl
import tqdm
from dataloader import *
from parameters import *

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":

    seed_everything(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')

    
    noisy_cnn = CNNs.ResNet50(
        num_classes=args.class_num,
        Pretrain=args.pretrain,
        checkpoint=None,
        noise_type=args.noise_type,
        noise_strength=args.noise_str,
        noisy_layer=args.noise_layer,
        sub_noisy_layer=args.sub_noisy_layer
    )
    print('model parameters:', sum(param.numel() for param in noisy_cnn.parameters())/1e6)

    # === DATA ===
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    if args.datasets == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    elif args.datasets == 'cifar100':
        train_data = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
    elif args.datasets == 'TinyImageNet':
        train_loader, test_loader = get_loader(args.tinyImagenet_path)
    elif args.datasets == 'ImageNet':
        train_loader, test_loader = get_loader(args.Imagenet_path)

    if args.datasets in ['cifar10', 'cifar100']:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # === TRAIN ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(noisy_cnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-9)
    noisy_cnn.to(device)
    best_acc = 0

    for epoch in range(args.epoch):
        noisy_cnn.train()
        total, correct = 0, 0
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = noisy_cnn(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch+1} loss:{loss.item():.4f} Acc:{correct/total:.3f}')
        scheduler.step()

        # === TEST ===
        noisy_cnn.eval()
        correct, total = 0, 0
        for j, data in enumerate(tqdm.tqdm(test_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = noisy_cnn(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'准确率：{100.0 * correct/total:.4f}%')

        # === SAVE ===
        if correct/total > best_acc:
            best_acc = correct/total
            if not os.path.exists('./saved_models'):
                os.mkdir('./saved_models')
            # torch.save(...)

    print('Finish training!')

    # === FINAL EVAL ===
    print("\n=== FINAL EVALUATION ON TEST SET ===")
    all_preds, all_labels = [], []
    noisy_cnn.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = noisy_cnn(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.show()

#Developer: Shawey
#Date: 03/16/2023
import CNNs 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os,random
import numpy as np
from torch.utils.data import DataLoader
import ssl
import tqdm
from dataloader import *
from parameters import *

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":

    seed_everything(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')

    
    noisy_cnn = CNNs.ResNet50(
        num_classes=args.class_num,
        Pretrain=args.pretrain,
        checkpoint=None,
        noise_type=args.noise_type,
        noise_strength=args.noise_str,
        noisy_layer=args.noise_layer,
        sub_noisy_layer=args.sub_noisy_layer
    )
    print('model parameters:', sum(param.numel() for param in noisy_cnn.parameters())/1e6)

    # === DATA ===
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    if args.datasets == 'cifar10':
        train_data = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    elif args.datasets == 'cifar100':
        train_data = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
    elif args.datasets == 'TinyImageNet':
        train_loader, test_loader = get_loader(args.tinyImagenet_path)
    elif args.datasets == 'ImageNet':
        train_loader, test_loader = get_loader(args.Imagenet_path)

    if args.datasets in ['cifar10', 'cifar100']:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # === TRAIN ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(noisy_cnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-9)
    noisy_cnn.to(device)
    best_acc = 0

    for epoch in range(args.epoch):
        noisy_cnn.train()
        total, correct = 0, 0
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = noisy_cnn(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch {epoch+1} loss:{loss.item():.4f} Acc:{correct/total:.3f}')
        scheduler.step()

        # === TEST ===
        noisy_cnn.eval()
        correct, total = 0, 0
        for j, data in enumerate(tqdm.tqdm(test_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = noisy_cnn(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'准确率：{100.0 * correct/total:.4f}%')

        # === SAVE ===
        if correct/total > best_acc:
            best_acc = correct/total
            if not os.path.exists('./saved_models'):
                os.mkdir('./saved_models')
            # torch.save(...)

    print('Finish training!')

    # === FINAL EVAL ===
    print("\n=== FINAL EVALUATION ON TEST SET ===")
    all_preds, all_labels = [], []
    noisy_cnn.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = noisy_cnn(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout() 


    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    plt.savefig(f'./saved_models/confusion_matrix_{args.noise_type}_{args.noise_str}.png')

    with open(f'./saved_models/report_{args.noise_type}_{args.noise_str}.txt', 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))

    plt.show()  
