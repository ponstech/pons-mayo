#Developer: Shawey
#Date: 03/16/2023
import CNNs 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import os
import random
import numpy as np
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tqdm
from MultimodalDataloader import MultimodalDataLoader
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_multimodal_cnn():
    """Main training function for multimodal CNN using existing CNN models"""
    
    # Set random seed
    seed_everything(42)
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Model loading - using existing CNN models
    print("Loading existing CNN model...")
    if args.resnet == 'resnet18':
        noisy_cnn = CNNs.ResNet18(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
    elif args.resnet == 'resnet34':
        noisy_cnn = CNNs.ResNet34(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
    elif args.resnet == 'resnet50':
        noisy_cnn = CNNs.ResNet50(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
    elif args.resnet == 'resnet101':
        noisy_cnn = CNNs.ResNet101(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
    elif args.resnet == 'resnet152':
        noisy_cnn = CNNs.ResNet152(num_classes=args.class_num, Pretrain=True, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
    
    print('Model parameters:', sum(param.numel() for param in noisy_cnn.parameters())/1e6, 'M')
    
    # Create multimodal dataloaders
    print("Creating multimodal dataloaders...")
    if hasattr(args, 'multimodal_data_path'):
        data_root = args.multimodal_data_path
    else:
        data_root = "/content/drive/MyDrive/dataset"  # Default path
    
    try:
        dataloader = MultimodalDataLoader(
            data_root=data_root,
            batch_size=args.batch_size,
            input_size=224,  # ResNet expects 224x224
            num_workers=2,
            split_type='train_test'  # Use train/test splits from your dataset structure
        )
        train_loader, test_loader = dataloader.get_loaders()
        classes = dataloader.get_classes()
        num_classes = dataloader.get_num_classes()
        print(f"Loaded {num_classes} classes: {classes}")
        print("Note: Input shape is (batch_size, 3, 224, 224) - 3 modalities as channels")
        print(f"Dataset structure: {data_root}/bmode/enhancement/improvement with train/test splits")

    except Exception as e:
        print(f"Error loading multimodal data: {e}")
        print("Please ensure the data directory structure is correct:")
        print(f"Expected: {data_root}/")
        print("├── bmode/")
        print("│   ├── train/")
        print("│   ├── test/")
        print("│   └── val/")
        print("├── enhancement/")
        print("│   ├── train/")
        print("│   ├── test/")
        print("│   └── val/")
        print("└── improvement/")
        print("    ├── train/")
        print("    ├── test/")
        print("    └── val/")
        return
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(noisy_cnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-9, last_epoch=-1)
    
    # Move model to device
    noisy_cnn.to(device)
    
    # Training variables
    best_acc = 0
    start_epoch = 0

    # Resume support
    def try_auto_resume():
        latest_path = None
        if not os.path.isdir(args.auto_resume_dir):
            return None
        candidates = [os.path.join(args.auto_resume_dir, f) for f in os.listdir(args.auto_resume_dir) if f.endswith('.pt') or f.endswith('.pth')]
        if not candidates:
            return None
        latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
        return latest_path

    resume_path = args.resume if hasattr(args, 'resume') and args.resume else ''
    if resume_path == 'auto':
        picked = try_auto_resume()
        if picked:
            resume_path = picked
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if 'model' in ckpt:
            noisy_cnn.load_state_dict(ckpt['model'])
        else:
            noisy_cnn.load_state_dict(ckpt)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'best_acc' in ckpt:
            best_acc = ckpt['best_acc']
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f"Resumed: start_epoch={start_epoch}, best_acc={best_acc:.4f}")
    train_losses = []
    train_accs = []
    test_accs = []
    
    print("Starting training...")
    train_losses = []
    train_accs = []
    test_accs = []
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(start_epoch, args.epoch):
        # Training phase
        noisy_cnn.train()
        total = 0
        correct = 0
        epoch_loss = 0.0
        
        for i, data in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch}')):
            # Unpack multimodal data (now single tensor with 3 channels)
            multimodal_input, labels = data
            
            # Move data to device
            multimodal_input = multimodal_input.to(device)
            labels = labels.to(device)
            
            # Forward pass - using existing CNN model
            outputs = noisy_cnn(multimodal_input)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        
        # Testing phase
        noisy_cnn.eval()
        test_correct = 0
        test_total = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for j, data in enumerate(tqdm.tqdm(test_loader, desc='Testing')):
                multimodal_input, labels = data
                
                multimodal_input = multimodal_input.to(device)
                labels = labels.to(device)
                
                # Forward pass - using existing CNN model
                outputs = noisy_cnn(multimodal_input)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Store predictions for metrics
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        test_acc = test_correct / test_total
        test_accs.append(test_acc)
        
        print(f'Test Accuracy: {test_acc:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(noisy_cnn, test_acc, epoch, args)
            print(f'New best model saved with accuracy: {test_acc:.4f}')

        # Always save a checkpoint for resume
        save_checkpoint(noisy_cnn, optimizer, scheduler, best_acc, epoch, args)

        # Calculate and display metrics
        if epoch % 5 == 0 or epoch == args.epoch - 1:  # Every 5 epochs or last epoch
            calculate_and_save_metrics(y_true, y_pred, classes, test_acc, epoch, args)
    
    print('Training finished!')
    print(f'Best test accuracy: {best_acc:.4f}')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, test_accs, args)

def save_model(model, accuracy, epoch, args):
    """Save the best model"""
    save_dir = '/content/drive/MyDrive/NoisyCNN_CircularShiftQ/saved_models/multimodal'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create filename based on model configuration
    if args.noise_str == 0:
        filename = f'{accuracy:.4f}_multimodal_{args.resnet}_vanilla_epoch_{epoch}.pth'
    else:
        filename = f'{accuracy:.4f}_multimodal_{args.resnet}_noise_{args.noise_type}_str_{args.noise_str}_epoch_{epoch}.pth'
    
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to: {save_path}')

def save_checkpoint(model, optimizer, scheduler, best_acc, epoch, args):
    save_dir = '/content/drive/MyDrive/NoisyCNN_CircularShiftQ/saved_models/multimodal'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'epoch': epoch,
        'args': vars(args) if hasattr(args, '__dict__') else {}
    }
    fname = f'checkpoint_{args.resnet}_epoch_{epoch}.pt'
    path = os.path.join(save_dir, fname)
    torch.save(ckpt, path)
    print(f'Checkpoint saved to: {path}')


def calculate_and_save_metrics(y_true, y_pred, classes, accuracy, epoch, args):
    """Calculate and save evaluation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Use class names if available, otherwise use generic names
    if classes and len(classes) == len(cm):
        class_names = classes
    else:
        class_names = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'][:len(cm)]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Multimodal {args.resnet} Confusion Matrix - Epoch {epoch+1}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    
    # Save confusion matrix
    save_dir = '/content/drive/MyDrive/NoisyCNN_CircularShiftQ/ResNet34/multimodal'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cm_filename = f'multimodal_{args.resnet}_confusion_matrix_epoch_{epoch+1}.png'
    plt.savefig(os.path.join(save_dir, cm_filename))
    plt.close()
    
    # Print metrics
    print(f"\nEpoch {epoch+1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    
    # Print specificity per class (normal Main.py'deki gibi)
    for i, spec in enumerate(specificity_per_class):
        print(f"Specificity (Class {i}): {spec:.4f}")
    
    # Save metrics to CSV
    metrics_dict = {
        'epoch': epoch + 1,
        'model': args.resnet,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'noise_type': args.noise_type,
        'noise_strength': args.noise_str
    }
    
    # Add specificity to CSV (normal Main.py'deki gibi)
    for idx, spec in enumerate(specificity_per_class):
        metrics_dict[f'specificity_class_{idx}'] = spec
    
    csv_path = os.path.join(save_dir, 'multimodal_resnet34_cnn_metrics.csv')
    df = pd.DataFrame([metrics_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def plot_training_curves(train_losses, train_accs, test_accs, args):
    """Plot training curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot train accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 3, 3)
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    save_dir = '/content/drive/MyDrive/NoisyCNN_CircularShiftQ/saved_models/multimodal'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plot_filename = f'multimodal_{args.resnet}_training_curves.png'
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close()
    
    print(f'Training curves saved to: {os.path.join(save_dir, plot_filename)}')

if __name__ == "__main__":
    # Add multimodal data path to args if not present
    if not hasattr(args, 'multimodal_data_path'):
        args.multimodal_data_path = "/content/drive/MyDrive/dataset"
    
    print("Starting Multimodal CNN Training with Existing CNN Models")
    print(f"Model: {args.resnet}")
    print(f"Data path: {args.multimodal_data_path}")
    print(f"Number of classes: {args.class_num}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epoch}")
    print(f"Noise type: {args.noise_type}")
    print(f"Noise strength: {args.noise_str}")
    print("Note: Using 3 modalities (B-mode, Enhanced, Improved) as 3 channels")
    
    train_multimodal_cnn()
