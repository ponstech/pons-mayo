from ViT import *
import numpy as np
import itertools
import time
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from breast_dataloader import get_breast_loader
from dataloader import get_loader
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from parameters import *  # Import after setting up paths

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

# Transforms for CIFAR datasets (if needed)
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

# NoisyNN noise injection functions (for ViT - operates on sequence tokens)
def OptimalQ_ViT(x):
    """OptimalQ for ViT - operates on sequence dimension"""
    s1, s2, s3 = x.shape  # [batch, seq_len, embed_dim]
    hidden_states_copy = x.clone()
    noise = 0 
    for i in range(1, s1):  
        sub_noise = torch.cat((hidden_states_copy[i:,1:,:], hidden_states_copy[0:i,1:,:]), dim=0)
        noise = noise + sub_noise 
    x[:,1:,:] = x[:,1:,:] - x[:,1:,:].clone() * (s1-1) / (s1+1) + noise / (s1+1)   
    return x

def Gaussian_ViT(x):
    """Gaussian noise for ViT"""
    s1, s2, s3 = x.shape
    means = opt.gau_mean * torch.ones(s1, s2-1, s3) 
    stds = opt.gau_var * torch.ones(s1, s2-1, s3)
    gaussian_noise = (torch.normal(means, stds)).to(x.device)
    x[:,1:,:] = opt.strength * gaussian_noise + x[:,1:,:]
    return x

def Impulse_ViT(x, prob):
    """Impulse noise for ViT"""
    s1, s2, s3 = x.shape
    noise_tensor = torch.rand(s1, s2-1, s3).to(x.device)
    x_clone = x.clone()
    salt = (torch.max(x_clone[:,1:,:])).detach()
    pepper = (torch.min(x_clone[:,1:,:])).detach()
    x_clone[:,1:,:][noise_tensor < prob/2] = salt
    x_clone[:,1:,:][noise_tensor > 1-prob/2] = pepper
    return x_clone

def apply_noise_vit(x, layer_idx, noise_type, strength):
    """Apply NoisyNN noise injection for ViT"""
    if layer_idx == opt.layer:
        if noise_type == 'linear' or noise_type == 'liner':  # typo in original
            if opt.OptimalQ:
                x = OptimalQ_ViT(x)
            else:
                hidden_states_copy = x.detach()
                hidden_states_copy = torch.cat((hidden_states_copy[1:,:,:], hidden_states_copy[0,:,:].unsqueeze(0)), dim=0)
                x = x + strength * (hidden_states_copy - x.detach())
        elif noise_type == 'gaussian':
            x = Gaussian_ViT(x)
        elif noise_type == 'impulse':
            x = Impulse_ViT(x, strength)
    return x

def train():
    # Initialize TensorBoard writer
    if(opt.OptimalQ):
        writer = SummaryWriter('./output/optimal')
    else:
        writer = SummaryWriter('./output/linear_'+str(int(10*opt.strength)))
    
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

    # Create base ViT model
    noise_vit = NoiseViT(patch_size=opt.patch_size, num_classes=opt.num_classes, 
                         embed_dim=HiddenSize, depth=Layers, num_heads=Heads)
    noise_vit.to(device)
    
    # Load pretrained weights if needed
    if(1):
        print('pretrained mode: '+'vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res))
        if(opt.scale == 'huge'):
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res)+'_in21k', 
                                            pretrained=True, num_classes=opt.num_classes)
        else:
            vit_pretrain = timm.create_model('vit_'+opt.scale+'_patch'+str(opt.patch_size)+'_'+str(opt.res), 
                                            pretrained=True, num_classes=opt.num_classes)
        vit_pretrain_weight = vit_pretrain.state_dict()
        noise_vit.load_state_dict(vit_pretrain_weight, strict=False)  # strict=False for different num_classes
        # Update head for 2 classes
        noise_vit.head = torch.nn.Linear(HiddenSize, opt.num_classes)
        noise_vit.head.to(device)
        last_th = 0

    print('model parameters:', sum(param.numel() for param in noise_vit.parameters()) / 1e6)

    # Override forward method for fusion
    original_forward = noise_vit.forward
    
    if opt.fusion_type == 'mid':
        def fusion_forward(x0, x1, x2, x3, x4, strength=0.0, noise_layer_index=11, noise_type='linear', train=True):
            """
            Mid Fusion for ViT: 5 images -> patch embedding -> concat -> transformer blocks
            """
            # Patch embedding for each image
            x0 = noise_vit.patch_embed(x0)  # [B, N, D]
            x1 = noise_vit.patch_embed(x1)
            x2 = noise_vit.patch_embed(x2)
            x3 = noise_vit.patch_embed(x3)
            x4 = noise_vit.patch_embed(x4)
            
            # Add positional embedding to each separately before concat
            B = x0.shape[0]
            num_patches = x0.shape[1]
            
            # Get pos_embed for patches (excluding cls token)
            # pos_embed shape: [1, num_patches+1, D] where first is cls token
            patch_pos_embed = noise_vit.pos_embed[:, 1:, :]  # [1, num_patches, D]
            
            # Add pos embedding to each
            x0 = x0 + patch_pos_embed
            x1 = x1 + patch_pos_embed
            x2 = x2 + patch_pos_embed
            x3 = x3 + patch_pos_embed
            x4 = x4 + patch_pos_embed
            
            # Mid fusion: concatenate patch embeddings
            # Concatenate along sequence dimension: [B, N*5, D]
            xx = torch.cat([x0, x1, x2, x3, x4], dim=1)
            
            # Add cls token
            cls_token = noise_vit.cls_token.expand(B, -1, -1)
            cls_pos_embed = noise_vit.pos_embed[:, 0:1, :]  # [1, 1, D]
            cls_token = cls_token + cls_pos_embed
            xx = torch.cat((cls_token, xx), dim=1)
            xx = noise_vit.pos_drop(xx)
            
            # Apply noise at embedding if specified
            if noise_layer_index == -1:
                xx = apply_noise_vit(xx, -1, noise_type, strength)
            
            # Pass through transformer blocks
            for i in range(noise_vit.layer):
                if i == noise_layer_index:
                    # Apply noise in this layer
                    xx = apply_noise_vit(xx, i, noise_type, strength)
                    xx = noise_vit.blocks[i](xx, False, strength, noise_type, train)
                else:
                    xx = noise_vit.blocks[i](xx, False, strength, noise_type, train)
            
            xx = noise_vit.norm(xx)
            xx = noise_vit.fc_norm(xx)
            return noise_vit.head(xx[:, 0])
        
        noise_vit.forward = fusion_forward
        
    elif opt.fusion_type == 'late':
        def fusion_forward(x0, x1, x2, x3, x4, strength=0.0, noise_layer_index=11, noise_type='linear', train=True):
            """
            Late Fusion for ViT: 5 images -> each through transformer -> concat features -> head
            """
            # Process each image separately through full ViT
            def process_single_image(x):
                x = noise_vit.patch_embed(x)
                B = x.shape[0]
                cls_token = noise_vit.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                x = noise_vit.pos_drop(x + noise_vit.pos_embed)
                
                if noise_layer_index == -1:
                    x = apply_noise_vit(x, -1, noise_type, strength)
                
                for i in range(noise_vit.layer):
                    if i == noise_layer_index:
                        x = apply_noise_vit(x, i, noise_type, strength)
                        x = noise_vit.blocks[i](x, False, strength, noise_type, train)
                    else:
                        x = noise_vit.blocks[i](x, False, strength, noise_type, train)
                
                x = noise_vit.norm(x)
                x = noise_vit.fc_norm(x)
                return x[:, 0]  # Return cls token feature
            
            # Process all 5 images
            feat0 = process_single_image(x0)
            feat1 = process_single_image(x1)
            feat2 = process_single_image(x2)
            feat3 = process_single_image(x3)
            feat4 = process_single_image(x4)
            
            # Late fusion: concatenate features
            fused_feat = torch.cat([feat0, feat1, feat2, feat3, feat4], dim=1)
            
            # Final classification head (need to adjust head size)
            if not hasattr(noise_vit, 'fusion_head'):
                noise_vit.fusion_head = torch.nn.Linear(HiddenSize * 5, opt.num_classes).to(device)
            
            return noise_vit.fusion_head(fused_feat)
        
        noise_vit.forward = fusion_forward
        
    elif opt.fusion_type == 'early':
        def fusion_forward(x0, x1, x2, x3, x4, strength=0.0, noise_layer_index=11, noise_type='linear', train=True):
            """
            Early Fusion for ViT: 5 images -> concat at channel dimension -> patch embedding -> transformer
            """
            # Early fusion: concatenate 5 images at channel dimension
            # [B, 3, H, W] * 5 -> [B, 15, H, W]
            x = torch.cat([x0, x1, x2, x3, x4], dim=1)
            
            # Need to adjust patch_embed for 15 channels instead of 3
            # Create a new patch embedding layer if not exists
            if not hasattr(noise_vit, 'early_fusion_patch_embed'):
                from timm.models.vision_transformer import PatchEmbed
                noise_vit.early_fusion_patch_embed = PatchEmbed(
                    img_size=opt.res,
                    patch_size=opt.patch_size,
                    in_chans=15,  # 3 channels * 5 images
                    embed_dim=HiddenSize
                ).to(device)
                # Initialize with pretrained weights (repeat 3-channel weights 5 times and average)
                pretrained_weight = noise_vit.patch_embed.proj.weight.data  # [D, 3, P, P]
                # Repeat and average: [D, 15, P, P] = repeat [D, 3, P, P] 5 times
                new_weight = pretrained_weight.repeat(1, 5, 1, 1) / 5.0
                noise_vit.early_fusion_patch_embed.proj.weight.data = new_weight
            
            # Patch embedding with 15-channel input
            x = noise_vit.early_fusion_patch_embed(x)  # [B, N, D]
            
            B = x.shape[0]
            num_patches = x.shape[1]
            
            # Add cls token
            cls_token = noise_vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = noise_vit.pos_drop(x + noise_vit.pos_embed)
            
            # Apply noise at embedding if specified
            if noise_layer_index == -1:
                x = apply_noise_vit(x, -1, noise_type, strength)
            
            # Pass through transformer blocks
            for i in range(noise_vit.layer):
                if i == noise_layer_index:
                    x = apply_noise_vit(x, i, noise_type, strength)
                    x = noise_vit.blocks[i](x, False, strength, noise_type, train)
                else:
                    x = noise_vit.blocks[i](x, False, strength, noise_type, train)
            
            x = noise_vit.norm(x)
            x = noise_vit.fc_norm(x)
            return noise_vit.head(x[:, 0])
        
        noise_vit.forward = fusion_forward
    else:
        raise ValueError(f"Unknown fusion type: {opt.fusion_type}. Choose from: early, mid, late")

    # Dataset loading
    if opt.datasets == 'BreastCancer':
        tra_dataloader = get_breast_loader(
            root_dir=opt.breast_dataset_path,
            fold=opt.fold,
            split='train',
            batch_size=opt.batch_size,
            num_workers=4
        )
        te_dataloader = get_breast_loader(
            root_dir=opt.breast_dataset_path,
            fold=opt.fold,
            split='val',
            batch_size=opt.te_batch_size,
            num_workers=4
        )
        test_dataloader = get_breast_loader(
            root_dir=opt.breast_dataset_path,
            fold=opt.fold,
            split='test',
            batch_size=opt.te_batch_size,
            num_workers=4
        )
    elif(opt.datasets == 'ImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.imagenet_path)
    elif(opt.datasets == 'TinyImageNet'):
        tra_dataloader, te_dataloader = get_loader(opt.tinyImagenet_path)
    elif(opt.datasets == 'Cifar100'):
        train_data = datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
        tra_dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
        te_dataloader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False)
    elif(opt.datasets == 'Cifar10'):
        train_data = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        tra_dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
        te_dataloader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False)
    
    start_t = time.time()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, noise_vit.parameters()), 
                                  lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epochs, lr_min=1e-7, warmup_t=opt.warm_up)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    # Store results for final table
    results = []

    for epoch in range(opt.epochs):
        noise_vit.train()

        num_steps_per_epoch = len(tra_dataloader)
        num_updates = epoch * num_steps_per_epoch

        batch_time = AverageMeter()
        accs = AverageMeter()
        epoch_tra_loss = AverageMeter()
        
        total = 0
        correct = 0
        
        pbar = tqdm(enumerate(tra_dataloader), total=len(tra_dataloader))
        for i, (images_tuple, tra_labels) in pbar:
            batchSize = tra_labels.shape[0]
            
            # Unpack multi-feature data: (x0, x1, x2, x3, x4), labels
            images_list = [img.float().to(device) for img in images_tuple]
            tra_labels = tra_labels.to(device)
            
            # Forward pass with fusion
            outputs = noise_vit(*images_list, 
                               strength=opt.strength, 
                               noise_layer_index=opt.layer,
                               noise_type=opt.noise_type, 
                               train=opt.tra)
            
            cls_loss = loss_fn(outputs, tra_labels)

            # Accuracy
            _, predictions = torch.max(outputs, 1)
            total += tra_labels.size(0)
            correct += (predictions == tra_labels).sum().item()

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step() 
            scheduler.step_update(num_updates=num_updates)      

            epoch_tra_loss.update(cls_loss.detach())
            accs.update(correct/total)
            batch_time.update(time.time() - start_t)
            
            if i % 200 == 5:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(tra_dataloader),
                                                                        batch_time=batch_time,
                                                                        loss=epoch_tra_loss,
                                                                        top1=accs))

        scheduler.step(epoch+1)
        train_acc_final = correct / total if total > 0 else 0.0
        writer.add_scalar('clssification_train_loss', epoch_tra_loss.avg, epoch)
        writer.add_scalar('train_acc', train_acc_final, epoch)

        # Validation
        accs = AverageMeter()
        epoch_te_loss = AverageMeter()
        accs_top5 = AverageMeter()  

        total = 0
        correct = 0
        correct_top5 = 0
        noise_vit.eval()
        
        for i, (images_tuple, te_labels) in enumerate(te_dataloader):
            images_list = [img.float().to(device) for img in images_tuple]
            te_labels = te_labels.to(device)

            with torch.no_grad():
                outputs = noise_vit(*images_list,
                                   strength=opt.strength,
                                   noise_layer_index=opt.layer,
                                   noise_type=opt.noise_type,
                                   train=opt.inf)
                cls_loss = loss_fn(outputs, te_labels)
       
                # Accuracy
                _, predictions = torch.max(outputs, 1)
                total += te_labels.size(0)
                correct += (predictions == te_labels).sum().item()
                
                epoch_te_loss.update(cls_loss.detach())
                accs.update(correct/total)

                # Top 5 accuracy (only if num_classes > 5, otherwise use min(num_classes, 5))
                topk = min(5, opt.num_classes)
                if topk > 1:
                    _, topk_pred = outputs.topk(topk, dim=1, largest=True, sorted=True)
                    correct_top5 += (topk_pred == te_labels.view(-1, 1)).sum().item()
                    accs_top5.update(correct_top5/total)
                else:
                    # For binary classification, top-1 is the same as top-5
                    accs_top5.update(correct/total)
        
            if i % 5 == 0:
                topk = min(5, opt.num_classes)
                topk_str = f'Top {topk}' if topk > 1 else 'Top 1'
                print('Test Epoch: [{0}][{1}/{2}]\t'                 
                    'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    '{topk_str} Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(te_dataloader),
                                                                            loss=epoch_te_loss,
                                                                            top1=accs,
                                                                            topk_str=topk_str,
                                                                            top5=accs_top5))
        
        writer.add_scalar('clssification_test_loss', epoch_te_loss.avg, epoch)
        writer.add_scalar('test_acc', accs.avg, epoch)
        writer.add_scalar('test_top5 acc', accs_top5.avg, epoch)
        
        # Evaluate on test set (if available)
        test_acc = 0.0
        if opt.datasets == 'BreastCancer' and 'test_dataloader' in locals():
            noise_vit.eval()
            test_total = 0
            test_correct = 0
            with torch.no_grad():
                for i, (images_tuple, test_labels) in enumerate(test_dataloader):
                    images_list = [img.float().to(device) for img in images_tuple]
                    test_labels = test_labels.to(device)
                    outputs = noise_vit(*images_list,
                                     strength=opt.strength,
                                     noise_layer_index=opt.layer,
                                     noise_type=opt.noise_type,
                                     train=opt.inf)
                    _, predictions = torch.max(outputs, 1)
                    test_total += test_labels.size(0)
                    test_correct += (predictions == test_labels).sum().item()
            test_acc = test_correct / test_total if test_total > 0 else 0.0
            print(f'Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})')
        
        # Store results for this epoch
        # Recalculate train accuracy (need to reset counters)
        train_total = 0
        train_correct = 0
        noise_vit.eval()
        with torch.no_grad():
            for i, (images_tuple, tra_labels) in enumerate(tra_dataloader):
                images_list = [img.float().to(device) for img in images_tuple]
                tra_labels = tra_labels.to(device)
                outputs = noise_vit(*images_list,
                                 strength=opt.strength,
                                 noise_layer_index=opt.layer,
                                 noise_type=opt.noise_type,
                                 train=opt.inf)
                _, predictions = torch.max(outputs, 1)
                train_total += tra_labels.size(0)
                train_correct += (predictions == tra_labels).sum().item()
        train_acc_final = train_correct / train_total if train_total > 0 else 0.0
        val_acc_final = accs.avg if hasattr(accs, 'avg') else 0.0
        results.append({
            'epoch': epoch + 1,
            'train_acc': train_acc_final,
            'val_acc': val_acc_final,
            'test_acc': test_acc,
            'train_loss': epoch_tra_loss.avg if hasattr(epoch_tra_loss, 'avg') else 0.0,
            'val_loss': epoch_te_loss.avg if hasattr(epoch_te_loss, 'avg') else 0.0
        })
        
        if accs.avg > last_th:
            last_th = accs.avg
            if(opt.noise_type == 'gaussian'):
                torch.save(noise_vit.state_dict(), opt.model_saved_path + '/'+'acc_'+str(last_th)+ 
                          '_lr_'+str(opt.lr)+'_strth_'+str(opt.strength)+'_layer_'+str(opt.layer)+
                          '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+
                          '_'+opt.datasets+'_'+str(opt.gau_mean)+'_'+str(opt.gau_var)+'_NV.pkl') 
            else:
                if(opt.OptimalQ == 1):
                    torch.save(noise_vit.state_dict(), opt.model_saved_path + '/'+'acc_'+str(last_th)+ 
                              '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+
                              '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+
                              '_'+opt.datasets+'_'+opt.fusion_type+'_NoisyViT.pkl') 
                else:
                    torch.save(noise_vit.state_dict(), opt.model_saved_path + '/'+'acc_'+str(last_th)+ 
                              '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+
                              '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+
                              '_'+opt.datasets+'_'+opt.fusion_type+'_OrdinaryViT.pkl') 

    writer.close()
    
    # Final evaluation with best model on test set
    if opt.datasets == 'BreastCancer' and 'test_dataloader' in locals():
        print("\n" + "="*60)
        print("Final Evaluation with Best Model on Test Set")
        print("="*60)
        
        # Load best model (last saved model)
        best_model_path = None
        if opt.noise_type == 'gaussian':
            best_model_path = opt.model_saved_path + '/' + 'acc_' + str(last_th) + \
                            '_lr_'+str(opt.lr)+'_strth_'+str(opt.strength)+'_layer_'+str(opt.layer)+\
                            '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                            '_'+opt.datasets+'_'+str(opt.gau_mean)+'_'+str(opt.gau_var)+'_NV.pkl'
        else:
            if opt.OptimalQ == 1:
                best_model_path = opt.model_saved_path + '/' + 'acc_' + str(last_th) + \
                                '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+\
                                '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                                '_'+opt.datasets+'_'+opt.fusion_type+'_NoisyViT.pkl'
            else:
                best_model_path = opt.model_saved_path + '/' + 'acc_' + str(last_th) + \
                                '_lr_'+str(opt.lr)+'_bs_'+str(opt.batch_size)+'_layer_'+str(opt.layer)+\
                                '_'+opt.scale+'_'+str(opt.res)+'_'+str(opt.patch_size)+'_'+opt.noise_type+\
                                '_'+opt.datasets+'_'+opt.fusion_type+'_OrdinaryViT.pkl'
        
        if best_model_path and os.path.exists(best_model_path):
            noise_vit.load_state_dict(torch.load(best_model_path))
        
        noise_vit.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i, (images_tuple, test_labels) in enumerate(tqdm(test_dataloader, desc='Final Testing')):
                images_list = [img.float().to(device) for img in images_tuple]
                test_labels = test_labels.to(device)
                outputs = noise_vit(*images_list,
                                 strength=opt.strength,
                                 noise_layer_index=opt.layer,
                                 noise_type=opt.noise_type,
                                 train=opt.inf)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import numpy as np
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # For binary classification (2 classes: 0=benign, 1=malignant)
        if opt.num_classes == 2:
            TN = cm[0, 0]  # True Negatives (benign predicted as benign)
            FP = cm[0, 1]  # False Positives (benign predicted as malignant)
            FN = cm[1, 0]  # False Negatives (malignant predicted as benign)
            TP = cm[1, 1]  # True Positives (malignant predicted as malignant)
            
            # Calculate metrics
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Print results table
            print("\n" + "="*80)
            print("FINAL TEST RESULTS")
            print("="*80)
            print(f"{'Metric':<20} {'Value':<15} {'Description'}")
            print("-"*80)
            print(f"{'Accuracy':<20} {accuracy:<15.4f} Overall correctness")
            print(f"{'Precision':<20} {precision:<15.4f} TP / (TP + FP)")
            print(f"{'Recall (Sensitivity)':<20} {recall:<15.4f} TP / (TP + FN)")
            print(f"{'Specificity':<20} {specificity:<15.4f} TN / (TN + FP)")
            print(f"{'F1-Score':<20} {f1_score:<15.4f} Harmonic mean of precision and recall")
            print("="*80)
            
            print("\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Benign  Malignant")
            print(f"Actual Benign    {TN:4d}     {FP:4d}")
            print(f"      Malignant  {FN:4d}     {TP:4d}")
            
            # Save confusion matrix as PNG
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malignant'],
                        yticklabels=['Benign', 'Malignant'],
                        cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            cm_filename = opt.model_saved_path + f'/confusion_matrix_breast_{opt.fusion_type}_fold{opt.fold}_{opt.scale}_noise_{opt.noise_type}_str_{opt.strength}_layer_{opt.layer}.png'
            plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to: {cm_filename}")
            
            # Save results to CSV
            import csv
            csv_filename = opt.model_saved_path + f'/final_results_breast_{opt.fusion_type}_fold{opt.fold}_{opt.scale}_noise_{opt.noise_type}_str_{opt.strength}_layer_{opt.layer}.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['metric', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'metric': 'accuracy', 'value': accuracy})
                writer.writerow({'metric': 'precision', 'value': precision})
                writer.writerow({'metric': 'recall', 'value': recall})
                writer.writerow({'metric': 'specificity', 'value': specificity})
                writer.writerow({'metric': 'f1_score', 'value': f1_score})
                writer.writerow({'metric': 'TP', 'value': TP})
                writer.writerow({'metric': 'TN', 'value': TN})
                writer.writerow({'metric': 'FP', 'value': FP})
                writer.writerow({'metric': 'FN', 'value': FN})
            print(f"\nResults saved to: {csv_filename}")
        else:
            # Multi-class classification
            accuracy = np.sum(all_predictions == all_labels) / len(all_labels)
            report = classification_report(all_labels, all_predictions, output_dict=True)
            
            print("\n" + "="*80)
            print("FINAL TEST RESULTS")
            print("="*80)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_labels, all_predictions))
            print("="*80)
        
        print(f'\nBest Validation Accuracy: {last_th:.4f}')

if __name__ == '__main__':
    set_seed(42)
    train()

