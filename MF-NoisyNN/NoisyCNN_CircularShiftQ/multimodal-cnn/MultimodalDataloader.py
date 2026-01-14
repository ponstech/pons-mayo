#Developer: Shawey
#Date: 03/16/2023
import torch
import torch.utils.data as data
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from parameters import *

class MultimodalDataset(data.Dataset):
    """
    Dataset for multimodal ultrasound images (B-mode, Enhanced, Improved)
    This dataloader works with existing CNN models by concatenating the 3 modalities
    """
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Define the three modality directories (your dataset structure)
        self.bmode_dir = os.path.join(root_dir, 'bmode')
        self.enhanced_dir = os.path.join(root_dir, 'enhancement')  
        self.improved_dir = os.path.join(root_dir, 'improvement')  
        
        # Get all image files
        self.image_files = []
        self.labels = []
        
        # Load class information - classes should be inside the split directory
        split_dir = os.path.join(self.bmode_dir, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory '{self.split}' not found in {self.bmode_dir}")
        
        self.classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found classes in split '{self.split}': {self.classes}")
        print(f"Classes directory: {split_dir}")
        
        # Collect all valid image pairs for the specified split
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            bmode_class_dir = os.path.join(self.bmode_dir, self.split, class_name)
            enhanced_class_dir = os.path.join(self.enhanced_dir, self.split, class_name)
            improved_class_dir = os.path.join(self.improved_dir, self.split, class_name)
            
            # Debug: Check split directories
            print(f"Checking class {class_name} for split '{self.split}':")
            print(f"  B-mode: {bmode_class_dir} - Exists: {os.path.exists(bmode_class_dir)}")
            print(f"  Enhancement: {enhanced_class_dir} - Exists: {os.path.exists(enhanced_class_dir)}")
            print(f"  Improvement: {improved_class_dir} - Exists: {os.path.exists(improved_class_dir)}")
            
            if not all(os.path.exists(d) for d in [bmode_class_dir, enhanced_class_dir, improved_class_dir]):
                print(f"Warning: Missing modality directory for class {class_name} in split '{self.split}'")
                continue
            
            # Get all image files in this class for the specified split
            bmode_files = sorted([f for f in os.listdir(bmode_class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff', '.bmp'))])
            
            print(f"  Found {len(bmode_files)} images in class {class_name} for split '{self.split}'")
            
            for img_file in bmode_files:
                bmode_path = os.path.join(bmode_class_dir, img_file)

                # Create enhanced and improved file paths with modalite suffixes
                # Remove file extension to add modalite suffix
                base_name = os.path.splitext(img_file)[0]  # Remove .png, .jpg etc.
                file_ext = os.path.splitext(img_file)[1]   # Get .png, .jpg etc.
                
                enhanced_filename = f"{base_name}_enhancement{file_ext}"
                improved_filename = f"{base_name}_improvement{file_ext}"
                
                enhanced_path = os.path.join(enhanced_class_dir, enhanced_filename)
                improved_path = os.path.join(improved_class_dir, improved_filename)
                
                # Check if all three modalities exist
                if all(os.path.exists(p) for p in [bmode_path, enhanced_path, improved_path]):
                    self.image_files.append({
                        'bmode': bmode_path,
                        'enhancement': enhanced_path,
                        'improvement': improved_path
                    })
                    self.labels.append(class_idx)
                else:
                    print(f"  Warning: Missing modality for {img_file} in class {class_name}")
                    if not os.path.exists(bmode_path):
                        print(f"    Missing B-mode: {bmode_path}")
                    if not os.path.exists(enhanced_path):
                        print(f"    Missing Enhancement: {enhanced_path}")
                    if not os.path.exists(improved_path):
                        print(f"    Missing Improvement: {improved_path}")
        
        print(f"Loaded {len(self.image_files)} multimodal image pairs for {split} split")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load all three modalities
        bmode_path = self.image_files[idx]['bmode']
        enhanced_path = self.image_files[idx]['enhancement']
        improved_path = self.image_files[idx]['improvement']
        
        # Load images
        bmode_img = Image.open(bmode_path).convert('L')  # Convert to grayscale
        enhanced_img = Image.open(enhanced_path).convert('L')
        improved_img = Image.open(improved_path).convert('L')
        
        # Apply transforms (without normalization)
        if self.transform:
            bmode_img = self.transform(bmode_img)
            enhanced_img = self.transform(enhanced_img)
            improved_img = self.transform(improved_img)
        
        # Concatenate the three modalities to create a 3-channel input
        # This maintains compatibility with existing CNN models
        multimodal_input = torch.cat([bmode_img, enhanced_img, improved_img], dim=0)
        
        # Normalize the 3-channel input
        multimodal_input = (multimodal_input - 0.5) / 0.5  # Normalize to [-1, 1] range
        
        label = self.labels[idx]
        
        return multimodal_input, label

def get_multimodal_transforms(input_size=224):
    """
    Get transforms for multimodal training and testing
    Compatible with existing CNN models (ResNet expects 224x224)
    """
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Don't normalize here - we'll normalize after concatenation
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # Don't normalize here - we'll normalize after concatenation
    ])
    
    return transform_train, transform_test

def create_multimodal_dataloaders(data_root, batch_size=32, input_size=224, num_workers=4, split_type='train_test'):
    """
    Create multimodal dataloaders for training and testing
    Compatible with existing CNN models
    Supports your dataset structure: bmode/enhancement/improvement with test/train/val splits
    """
    transform_train, transform_test = get_multimodal_transforms(input_size)
    
    print(f"Creating dataloaders for split_type: {split_type}")
    print(f"Data root: {data_root}")
    
    if split_type == 'train_test':
        # Use train and test splits
        print("Creating train dataset...")
        train_dataset = MultimodalDataset(
            root_dir=data_root,  # Root directory containing bmode/enhancement/improvement
            transform=transform_train,
            split='train'
        )
        
        print("Creating test dataset...")
        test_dataset = MultimodalDataset(
            root_dir=data_root,  # Root directory containing bmode/enhancement/improvement
            transform=transform_test,
            split='test'
        )
    elif split_type == 'train_val':
        # Use train and validation splits
        print("Creating train dataset...")
        train_dataset = MultimodalDataset(
            root_dir=data_root,
            transform=transform_train,
            split='train'
        )
        
        print("Creating validation dataset...")
        test_dataset = MultimodalDataset(
            root_dir=data_root,
            transform=transform_test,
            split='val'
        )
    else:
        raise ValueError(f"Invalid split_type: {split_type}. Use 'train_test' or 'train_val'")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes

class MultimodalDataLoader:
    """
    Wrapper class for multimodal dataloader with additional functionality
    Compatible with existing CNN models
    Supports your dataset structure: bmode/enhancement/improvement with test/train/val splits
    """
    def __init__(self, data_root, batch_size=32, input_size=224, num_workers=4, split_type='train_test'):
        self.data_root = data_root
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        self.split_type = split_type
        
        self.train_loader, self.test_loader, self.classes = create_multimodal_dataloaders(
            data_root, batch_size, input_size, num_workers, split_type
        )
        
        self.num_classes = len(self.classes)
        print(f"Created multimodal dataloaders with {self.num_classes} classes")
        print(f"Input shape: (batch_size, 3, {input_size}, {input_size}) - 3 modalities as channels")
    
    def get_loaders(self):
        return self.train_loader, self.test_loader
    
    def get_classes(self):
        return self.classes
    
    def get_num_classes(self):
        return self.num_classes

# Example usage and testing
if __name__ == "__main__":
    # Test the dataloader
    data_root = "/content/drive/MyDrive/dataset"
    
    try:
        dataloader = MultimodalDataLoader(data_root, batch_size=4, input_size=224)
        train_loader, test_loader = dataloader.get_loaders()
        
        # Test a batch
        for multimodal_input, labels in train_loader:
            print(f"Batch shapes:")
            print(f"Multimodal input: {multimodal_input.shape}")  # Should be (batch_size, 3, 224, 224)
            print(f"Labels: {labels.shape}")
            print(f"Labels: {labels}")
            break
            
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        print("Make sure the data directory structure is correct:")
        print("data_root/")
        print("├── train/")
        print("│   ├── bmode/")
        print("│   │   ├── class1/")
        print("│   │   └── class2/")
        print("│   ├── enhanced/")
        print("│   │   ├── class1/")
        print("│   │   └── class2/")
        print("│   └── improved/")
        print("│       ├── class1/")
        print("│       └── class2/")
        print("└── test/")
        print("    ├── bmode/")
        print("    ├── enhanced/")
        print("    └── improved/")
