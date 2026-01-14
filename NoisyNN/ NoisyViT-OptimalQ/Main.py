from ViT import *
import numpy as np
import itertools
import time
from parameters import *
import torch
from Accdataloader import *
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter

if(opt.OptimalQ):
    writer = SummaryWriter('./output/optimal')
else:
    writer = SummaryWriter('./output/linear_'+str(int(10*opt.strength)))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Selected Device: {device}")


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

def train():
    if(opt.scale == 'tiny'):
        Layers = 12
        HiddenSize = 192
        Heads = 3
        MLPSize = 768
    elif(opt.scale == 'small'):
        Layers = 12
        HiddenSize = 384
        Heads = 12
        MLPSize = 1536
    elif(opt.scale == 'base'):
        Layers = 12
        HiddenSize = 768
        Heads = 12
        MLPSize = 3072
    elif(opt.scale == 'large'):
        Layers = 24
        HiddenSize = 1024
        Heads = 16
        MLPSize = 4096
    elif(opt.scale == 'huge'):
        Layers = 32
        HiddenSize = 1280
        Heads = 16
        MLPSize = 5120

    noise_vit = NoiseViT(patch_size=opt.patch_size, num_classes=opt.num_classes, embed_dim=HiddenSize, depth=Layers, num_heads=Heads)
    noise_vit.to(device)
    
    print('pretrained mode: '+'vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res))
    vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res), pretrained=True, num_classes=opt.num_classes)
    noise_vit.load_state_dict(vit_pretrain.state_dict())
    last_th = 0

    print('model parameters:', sum(param.numel() for param in noise_vit.parameters()) /1e6)

    tra_dataloader, te_dataloader = get_loader(opt.imagenet_path)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, noise_vit.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epochs, lr_min=1e-7, warmup_t=opt.warm_up)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    for epoch in range(opt.epochs):
        noise_vit.train()
        num_steps_per_epoch = len(tra_dataloader)
        num_updates = epoch * num_steps_per_epoch
        accs = AverageMeter()
        epoch_tra_loss = AverageMeter()
        total = correct = 0

        pbar = tqdm(enumerate(tra_dataloader), total=len(tra_dataloader))
        for i, (tra_imgs, tra_labels) in pbar:
            tra_imgs = tra_imgs.float().to(device)
            outputs = noise_vit(tra_imgs, noise_layer_index=opt.layer, strength=opt.strength, noise_type=opt.noise_type, train=opt.tra)
            cls_loss = loss_fn(outputs, tra_labels.to(device))
            _, predictions = torch.max(outputs, 1)
            total += tra_labels.size(0)
            correct += (predictions == tra_labels.to(device)).sum().item()

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            scheduler.step_update(num_updates=num_updates)

            epoch_tra_loss.update(cls_loss.detach())
            accs.update(correct/total)

            if i % 200 == 5:
                print(f"Train Epoch: [{epoch}][{i}/{len(tra_dataloader)}]\t"
                      f"Classification_Loss {epoch_tra_loss.val:.4f} ({epoch_tra_loss.avg:.4f})\t"
                      f"Accuracy {accs.val:.3f} ({accs.avg:.3f})")

        scheduler.step(epoch+1)
        writer.add_scalar('clssification_train_loss', epoch_tra_loss.avg, epoch)
        writer.add_scalar('train_acc', accs.avg, epoch)

      
        accs = AverageMeter()
        epoch_te_loss = AverageMeter()
        accs_top5 = AverageMeter()
        total = correct = correct_top5 = 0
        noise_vit.eval()

        for i, (te_imgs, te_labels) in enumerate(te_dataloader):
            te_imgs = te_imgs.float().to(device)
            with torch.no_grad():
                outputs = noise_vit(te_imgs, noise_layer_index=opt.layer, strength=opt.strength, noise_type=opt.noise_type, train=opt.inf)
                cls_loss = loss_fn(outputs, te_labels.to(device))
                _, predictions = torch.max(outputs, 1)
                total += te_labels.size(0)
                correct += (predictions == te_labels.to(device)).sum().item()
                epoch_te_loss.update(cls_loss.detach())
                accs.update(correct/total)
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += (top5_pred == te_labels.view(-1, 1).to(device)).sum().item()
                accs_top5.update(correct_top5/total)

            if i % 5 == 0:
                print(f"Test Epoch: [{epoch}][{i}/{len(te_dataloader)}]\t"
                      f"Classification_Loss {epoch_te_loss.val:.4f} ({epoch_te_loss.avg:.4f})\t"
                      f"Accuracy {accs.val:.3f} ({accs.avg:.3f})\t"
                      f"Top 5 Accuracy {accs_top5.val:.3f} ({accs_top5.avg:.3f})")

        writer.add_scalar('clssification_test_loss', epoch_te_loss.avg, epoch)
        writer.add_scalar('test_acc', accs.avg, epoch)
        writer.add_scalar('test_top5 acc', accs_top5.avg, epoch)

        if accs.avg > last_th:
            last_th = accs.avg
            torch.save(noise_vit.state_dict(), f"{opt.model_saved_path}/best_model_{last_th:.4f}.pth")

    writer.close()

    ###########################################
    # FINAL TEST REPORT
    ###########################################
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    print("\n=== FINAL EVALUATION ON TEST SET ===")
    all_preds = []
    all_labels = []
    noise_vit.eval()
    with torch.no_grad():
        for imgs, labels in te_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = noise_vit(imgs, noise_layer_index=opt.layer, strength=opt.strength, noise_type=opt.noise_type, train=opt.inf)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    report_text = classification_report(all_labels, all_preds, target_names=te_dataloader.dataset.classes, digits=4)
    print(report_text)

    # Save report to file
    os.makedirs('./saved_models', exist_ok=True)  # Klasör yoksa oluştur
    with open(f'./saved_models/report_{opt.noise_type}_{opt.strength}.txt', 'w') as f:
        f.write(report_text)


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=te_dataloader.dataset.classes,
                yticklabels=te_dataloader.dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.show()

if __name__ == '__main__':
    set_seed(42)
    train()
