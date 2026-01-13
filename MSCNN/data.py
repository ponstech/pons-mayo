import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable

file_name = 'C:/Users/Monster/Desktop/mscnn/mscnn/dataset/'
# file_name = '../datasets/endo_xiao/bmode/'
b_size = 16
# fold = 4

# from PIL import Image
# import pandas as pd
# import os

# class FusionDataset(object):
#     def __init__(self, csv_file, dataset_1, transform1 = None):
#         super(FusionDataset, self).__init__()
        
#         self.frame = pd.read_csv(csv_file, header = None)
#         self.transform1 = transform1
#         self.dataset_1 = dataset_1
                  
#     def __len__(self):
#         return len(self.frame)
    
#     def __getitem__(self, idx):        
#         cat = self.frame.loc[idx][0].split(' ')
        
#         img_path_1 = os.path.join('..', cat[1], self.dataset_1, cat[2])
#         label = int(cat[3])
#         img_1 = Image.open(img_path_1).convert('RGB')
        
#         # if self.transform is not None:
#         img_1 = self.transform1(img_1)          
#         return img_1, label


def data_load():
    # load data
    data_transform = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.Grayscale(1),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST)
                
        ])


    # B-mode data loader
    train_data_Bmode = torchvision.datasets.ImageFolder(root=file_name+'train', transform=data_transform)
    print(train_data_Bmode.class_to_idx)
    target_names = list(train_data_Bmode.class_to_idx)
    print(torch.unique(torch.as_tensor(train_data_Bmode.targets), return_counts=True))
    
    train_data_Bmode_loader = torch.utils.data.DataLoader(train_data_Bmode, batch_size=b_size,  shuffle=True, num_workers=2)
    num_train_instances = len(train_data_Bmode)

    test_data_Bmode = torchvision.datasets.ImageFolder(root=file_name+'val', transform=data_transform)
    test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=b_size, shuffle=True, num_workers=2)
    print(torch.unique(torch.as_tensor(test_data_Bmode.targets), return_counts=True))
    num_test_instances = len(test_data_Bmode)
    
    
#     file_train = os.path.join('../../Multi-Scale-Feature-Fusion/Fus-CNNs_COVID-19_US', 'train_ds_'+str(fold)+'.txt')
#     train_data_Bmode_loader =  torch.utils.data.DataLoader(
#                                 FusionDataset(file_train, 'org', data_transform), 
#                                 batch_size=b_size, 
#                                 shuffle=True, 
#                                 num_workers=2) 
                    
#     num_train_instances = len(train_data_Bmode_loader.dataset)

#     file_test = os.path.join('../../Multi-Scale-Feature-Fusion/Fus-CNNs_COVID-19_US', 'val_ds_'+str(fold)+'.txt')
#     test_data_Bmode_loader =  torch.utils.data.DataLoader(
#                                 FusionDataset(file_test, 'org', data_transform), 
#                                 batch_size=b_size, 
#                                 shuffle=True, 
#                                 num_workers=2)   
                    
#     num_test_instances = len(test_data_Bmode_loader.dataset)
    
    
#     # R1 data loader
#     train_data_R1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R1/train', transform=data_transform)
#     train_data_R1_loader = torch.utils.data.DataLoader(train_data_R1, batch_size=16, shuffle=False, num_workers=0)

#     test_data_R1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R1/test', transform=data_transform)
#     test_data_R1_loader = torch.utils.data.DataLoader(test_data_R1, batch_size=16, shuffle=False, num_workers=0)


#     # R4 data loader
#     train_data_R4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R4/train', transform=data_transform)
#     train_data_R4_loader = torch.utils.data.DataLoader(train_data_R4,  batch_size=16, shuffle=False, num_workers=0)

#     test_data_R4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R4/test', transform=data_transform)
#     test_data_R4_loader = torch.utils.data.DataLoader(test_data_R4, batch_size=16, shuffle=False, num_workers=0)


#     # S1 data loader
#     train_data_S1= torchvision.datasets.ImageFolder(root='.../Dataset/1/S1/train', transform=data_transform)
#     train_data_S1_loader = torch.utils.data.DataLoader(train_data_S1, batch_size=16, shuffle=False, num_workers=0)

#     test_data_S1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/S1/test', transform=data_transform)
#     test_data_S1_loader = torch.utils.data.DataLoader(test_data_S1, batch_size=16, shuffle=False, num_workers=0)


#     # S4 data loader
#     train_data_S4= torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/train', transform=data_transform)
#     train_data_S4_loader = torch.utils.data.DataLoader(train_data_S4, batch_size=16, shuffle=False, num_workers=0)

#     test_data_S4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/test', transform=data_transform)
#     test_data_S4_loader = torch.utils.data.DataLoader(test_data_S4, batch_size=16, shuffle=False, num_workers=0)


    # do concatenation for training data
#     samples_train = []
#     labels_train = []
#     train_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1), (samples_S4, labels_S4) \
#             in zip(train_data_Bmode_loader, train_data_R1_loader,  train_data_R4_loader, train_data_S1_loader, train_data_S4_loader):
#     for (samples_Bmode, labels_Bmode) in train_data_Bmode_loader:
#         train_flag += 1
#         samples1 = Variable(samples_Bmode)
# #         samples2 = Variable(samples_R1)
# #         samples3 = Variable(samples_R4)
# #         samples4 = Variable(samples_S1)
# #         samples5 = Variable(samples_S4)
        
# #         std, mean = torch.std_mean(samples1, unbiased=True)
# #         sample_train = (samples1-mean)/std
#         sample_train = samples1

#         labels = labels_Bmode.squeeze()
#         label_train = Variable(labels)
#         if train_flag == 1:
#             samples_train = sample_train
#             labels_train = label_train
#         if train_flag > 1:
#             samples_train = torch.cat([samples_train, sample_train], dim=0)
#             labels_train = torch.cat([labels_train, label_train], dim=0) 


    # do concatenation for test data
#     samples_test = []
#     labels_test = []
#     test_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1), (samples_S4, labels_S4) \
#             in zip(test_data_Bmode_loader, test_data_R1_loader, test_data_R4_loader, test_data_S1_loader, test_data_S4_loader):
#     for (samples_Bmode, labels_Bmode) in test_data_Bmode_loader:
#         test_flag += 1
#         samples1 = Variable(samples_Bmode)
# #         samples2 = Variable(samples_R1)
# #         samples3 = Variable(samples_R4)
# #         samples4 = Variable(samples_S1)
# #         samples5 = Variable(samples_S4)
# #         sample_test = torch.cat([samples1, samples2, samples3, samples4, samples5], dim=1)

# #         std, mean = torch.std_mean(samples1, unbiased=True)
# #         sample_test = (samples1-mean)/std
#         sample_test = samples1

#         labels = labels_Bmode.squeeze()
#         label_test = Variable(labels)
#         if test_flag == 1:
#             samples_test = sample_test
#             labels_test = label_test
#         if test_flag > 1:
#             samples_test = torch.cat([samples_test, sample_test], dim=0)
#             labels_test = torch.cat([labels_test, label_test], dim=0)

    return target_names, train_data_Bmode_loader, test_data_Bmode_loader, num_train_instances, num_test_instances

