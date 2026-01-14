#Developer: Shawey
#Date: 03/16/2023
#Modified for Breast Cancer Multi-Feature Fusion
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from breast_dataloader import get_breast_loader
from fusion_models.mid_fusion import MSResNet as MidFusionMSResNet
from fusion_models.late_fusion import MSResNet as LateFusionMSResNet
from fusion_models.early_fusion import MSResNet as EarlyFusionMSResNet
from parameters import *

seed_everything(42)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

# NoisyNN noise injection functions
def Gaussian(x):
    s1, s2, s3, s4 = x.shape
    means = args.gau_mean * torch.ones(s1, s2, s3, s4)
    stds = args.gau_var * torch.ones(s1, s2, s3, s4)
    gaussian_noise = (torch.normal(means, stds)).to(x.device)
    return args.noise_str * gaussian_noise + x

def Impulse(x, prob):
    noise_tensor = torch.rand(x.size()).to(x.device)
    salt = (torch.max(x.clone())).detach()
    pepper = (torch.min(x.clone())).detach()
    x_clone = x.clone()
    x_clone[noise_tensor < prob/2] = salt
    x_clone[noise_tensor > 1-prob/2] = pepper
    return x_clone

def apply_noise(x, layer_idx):
    """Apply NoisyNN noise injection"""
    if layer_idx == args.noisy_layer:
        if args.noise_type == 'linear':
            x_copy = x.detach()
            x_copy = torch.cat((x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)), dim=0)
            noise = (x_copy - x.detach())
            x = x + args.noise_str * noise
        elif args.noise_type == 'gaussian':
            x = Gaussian(x)
        elif args.noise_type == 'impulse':
            x = Impulse(x, args.noise_str)
    return x

# Model loading - Fusion models from fusion_models directory
print(f"Loading {args.fusion_type} fusion model...")
if args.fusion_type == 'mid':
    model = MidFusionMSResNet(
        input_channel=3,
        layers=[1, 1, 1],
        num_classes=args.class_num
    )
    # Override forward to add noise injection
    original_forward = model.forward
    def noisy_forward(x0, x1, x2, x3, x4):
        # Process each input with noise injection
        for i, x in enumerate([x0, x1, x2, x3, x4]):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = apply_noise(x, 1)
            if i == 0:
                x0 = x
            elif i == 1:
                x1 = x
            elif i == 2:
                x2 = x
            elif i == 3:
                x3 = x
            else:
                x4 = x
        
        xx = torch.cat([x0, x1, x2, x3, x4], dim=1)
        
        # 3x3 branch with noise
        x = model.layer3x3_1(xx)
        x = apply_noise(x, 2)
        x = model.layer3x3_2(x)
        x = apply_noise(x, 3)
        x = model.layer3x3_3(x)
        x = apply_noise(x, 4)
        # Use adaptive pooling if feature map is too small
        if x.shape[2] < 16 or x.shape[3] < 16:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = model.maxpool3(x)
        
        # 5x5 branch with noise
        y = model.layer5x5_1(xx)
        y = apply_noise(y, 2)
        y = model.layer5x5_2(y)
        y = apply_noise(y, 3)
        y = model.layer5x5_3(y)
        y = apply_noise(y, 4)
        # Use adaptive pooling if feature map is too small
        if y.shape[2] < 11 or y.shape[3] < 11:
            y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1))
        else:
            y = model.maxpool5(y)
        
        # 7x7 branch with noise
        z = model.layer7x7_1(xx)
        z = apply_noise(z, 2)
        z = model.layer7x7_2(z)
        z = apply_noise(z, 3)
        z = model.layer7x7_3(z)
        z = apply_noise(z, 4)
        # Use adaptive pooling if feature map is too small
        if z.shape[2] < 6 or z.shape[3] < 6:
            z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
        else:
            z = model.maxpool7(z)
        
        out = torch.cat([x, y, z], dim=1)
        out = out.squeeze()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return model.fc(out)
    
    model.forward = noisy_forward
    
elif args.fusion_type == 'late':
    model = LateFusionMSResNet(
        input_channel=3,
        layers=[1, 1, 1],
        num_classes=args.class_num
    )
    # Override forward to add noise injection
    original_forward = model.forward
    def process_single_input(a):
        a = model.conv1(a)
        a = model.bn1(a)
        a = model.relu(a)
        a = model.maxpool(a)
        a = apply_noise(a, 1)
        
        x = model.layer3x3_1(a)
        x = apply_noise(x, 2)
        x = model.layer3x3_2(x)
        x = apply_noise(x, 3)
        x = model.layer3x3_3(x)
        x = apply_noise(x, 4)
        # Use adaptive pooling if feature map is too small
        if x.shape[2] < 16 or x.shape[3] < 16:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = model.maxpool3(x)
        
        y = model.layer5x5_1(a)
        y = apply_noise(y, 2)
        y = model.layer5x5_2(y)
        y = apply_noise(y, 3)
        y = model.layer5x5_3(y)
        y = apply_noise(y, 4)
        # Use adaptive pooling if feature map is too small
        if y.shape[2] < 11 or y.shape[3] < 11:
            y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1))
        else:
            y = model.maxpool5(y)
        
        z = model.layer7x7_1(a)
        z = apply_noise(z, 2)
        z = model.layer7x7_2(z)
        z = apply_noise(z, 3)
        z = model.layer7x7_3(z)
        z = apply_noise(z, 4)
        # Use adaptive pooling if feature map is too small
        if z.shape[2] < 6 or z.shape[3] < 6:
            z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
        else:
            z = model.maxpool7(z)
        
        return x, y, z
    
    def noisy_forward(a0, a1, a2, a3, a4):
        x0, y0, z0 = process_single_input(a0)
        x1, y1, z1 = process_single_input(a1)
        x2, y2, z2 = process_single_input(a2)
        x3, y3, z3 = process_single_input(a3)
        x4, y4, z4 = process_single_input(a4)
        
        out = torch.cat([x0, x1, x2, x3, x4,
                         y0, y1, y2, y3, y4,
                         z0, z1, z2, z3, z4], dim=1)
        out = out.squeeze()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return model.fc(out)
    
    model.forward = noisy_forward
    
elif args.fusion_type == 'early':
    # Early fusion: 5 images concatenated at input level (channel dimension)
    model = EarlyFusionMSResNet(
        input_channel=15,  # 3 channels * 5 images = 15
        layers=[1, 1, 1],
        num_classes=args.class_num
    )
    # Override forward to add noise injection and handle 5 inputs
    original_forward = model.forward
    def noisy_forward(x0, x1, x2, x3, x4):
        # Early fusion: concatenate 5 images at channel dimension
        x = torch.cat([x0, x1, x2, x3, x4], dim=1)  # [B, 15, H, W]
        
        x = model.conv1(x)
        x = apply_noise(x, 1)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        
        # 3x3 branch with noise
        x_branch = model.layer3x3_1(x)
        x_branch = apply_noise(x_branch, 2)
        x_branch = model.layer3x3_2(x_branch)
        x_branch = apply_noise(x_branch, 3)
        x_branch = model.layer3x3_3(x_branch)
        x_branch = apply_noise(x_branch, 4)
        # Use adaptive pooling if feature map is too small
        if x_branch.shape[2] < 16 or x_branch.shape[3] < 16:
            x_branch = torch.nn.functional.adaptive_avg_pool2d(x_branch, (1, 1))
        else:
            x_branch = model.maxpool3(x_branch)
        
        # 5x5 branch with noise
        y_branch = model.layer5x5_1(x)
        y_branch = apply_noise(y_branch, 2)
        y_branch = model.layer5x5_2(y_branch)
        y_branch = apply_noise(y_branch, 3)
        y_branch = model.layer5x5_3(y_branch)
        y_branch = apply_noise(y_branch, 4)
        # Use adaptive pooling if feature map is too small
        if y_branch.shape[2] < 11 or y_branch.shape[3] < 11:
            y_branch = torch.nn.functional.adaptive_avg_pool2d(y_branch, (1, 1))
        else:
            y_branch = model.maxpool5(y_branch)
        
        # 7x7 branch with noise
        z_branch = model.layer7x7_1(x)
        z_branch = apply_noise(z_branch, 2)
        z_branch = model.layer7x7_2(z_branch)
        z_branch = apply_noise(z_branch, 3)
        z_branch = model.layer7x7_3(z_branch)
        z_branch = apply_noise(z_branch, 4)
        # Use adaptive pooling if feature map is too small
        if z_branch.shape[2] < 6 or z_branch.shape[3] < 6:
            z_branch = torch.nn.functional.adaptive_avg_pool2d(z_branch, (1, 1))
        else:
            z_branch = model.maxpool7(z_branch)
        
        out = torch.cat([x_branch, y_branch, z_branch], dim=1)
        out = out.squeeze()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return model.fc(out)
    
    model.forward = noisy_forward
else:
    raise ValueError(f"Unknown fusion type: {args.fusion_type}. Choose from: early, mid, late")

print('Model parameters:', sum(param.numel() for param in model.parameters())/1e6, 'M')

# Dataset loading
if args.datasets == 'BreastCancer':
    train_loader = get_breast_loader(
        root_dir=args.breast_dataset_path,
        fold=args.fold,
        split='train',
        batch_size=args.batch_size,
        num_workers=4
    )
    val_loader = get_breast_loader(
        root_dir=args.breast_dataset_path,
        fold=args.fold,
        split='val',
        batch_size=args.batch_size,
        num_workers=4
    )
    test_loader = get_breast_loader(
        root_dir=args.breast_dataset_path,
        fold=args.fold,
        split='test',
        batch_size=args.batch_size,
        num_workers=4
    )
else:
    raise ValueError(f"Dataset {args.datasets} not supported for fusion models")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epoch, eta_min=1e-9, last_epoch=-1
)

# Move model to device
model.to(device)
best_val_acc = 0
best_test_acc = 0
best_model_state = None

# Training loop
print("Starting training...")
# Store results for final table
results = []

for epoch in range(args.epoch):
    model.train()
    total = 0
    correct = 0
    train_loss = 0.0
    
    for i, data in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch}')):
        # Unpack multi-feature data: (x0, x1, x2, x3, x4), labels
        images_tuple, labels = data
        
        # Move all images and labels to device
        images_list = [img.to(device) for img in images_tuple]
        labels = labels.to(device)
        
        # Forward pass
        if args.fusion_type in ['early', 'mid', 'late']:
            outputs = model(*images_list)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Calculate accuracy
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct / total
    print(f'Epoch {epoch+1}/{args.epoch} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
    scheduler.step()
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for j, data in enumerate(tqdm.tqdm(val_loader, desc='Validation')):
            images_tuple, labels = data
            images_list = [img.to(device) for img in images_tuple]
            labels = labels.to(device)
            
            outputs = model(*images_list)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item()
    
    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}')
    
    # Evaluate on test set each epoch
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            images_tuple, labels = data
            images_list = [img.to(device) for img in images_tuple]
            labels = labels.to(device)
            
            outputs = model(*images_list)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})')
    
    # Store results for this epoch
    results.append({
        'epoch': epoch + 1,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    })
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_model_state = model.state_dict().copy()
        
        model_name = f'val_{val_acc:.4f}_test_{test_acc:.4f}_breast_{args.fusion_type}_fold{args.fold}_{args.resnet}_noise_{args.noise_type}_str_{args.noise_str}_layer_{args.noisy_layer}.pth'
        torch.save(model.state_dict(), f'./saved_models/{model_name}')
        print(f'Model saved: {model_name} (Val: {val_acc:.4f}, Test: {test_acc:.4f})')

# Final evaluation with best model
print("\n" + "="*60)
print("Final Evaluation with Best Model (based on validation)")
print("="*60)
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for data in tqdm.tqdm(test_loader, desc='Final Testing'):
        images_tuple, labels = data
        images_list = [img.to(device) for img in images_tuple]
        labels = labels.to(device)
        
        outputs = model(*images_list)
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# For binary classification (2 classes: 0=benign, 1=malignant)
if args.class_num == 2:
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
    
    cm_filename = f'./saved_models/confusion_matrix_breast_{args.fusion_type}_fold{args.fold}_noise_{args.noise_type}_str_{args.noise_str}_layer_{args.noisy_layer}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_filename}")
    
    # Save results to CSV
    import csv
    csv_filename = f'./saved_models/final_results_breast_{args.fusion_type}_fold{args.fold}_noise_{args.noise_type}_str_{args.noise_str}_layer_{args.noisy_layer}.csv'
    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')
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

print(f'\nBest Validation Accuracy: {best_val_acc:.4f}')
print('\nTraining completed!')

