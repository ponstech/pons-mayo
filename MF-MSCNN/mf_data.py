import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        xC = self.datasetC[index]
        return xA, xB, xC
    
    def __len__(self):
        return len(self.datasetA)


file_name = "./" # data path
b_size = 12
def data_load():
    # load data
#     data_transform = transforms.Compose([
#                 transforms.Grayscale(1),
#                 transforms.Resize((256,256)),
#                 transforms.ToTensor(),   
#         ])
    
    data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(1),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    data_enhanced_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    # ORG data loader
    train_data_Bmode = torchvision.datasets.ImageFolder(root=file_name+'bmode/train', transform=data_transform)
    print(train_data_Bmode.class_to_idx)
    target_names = list(train_data_Bmode.class_to_idx)
#     train_data_Bmode_loader = torch.utils.data.DataLoader(train_data_Bmode, batch_size=b_size,  shuffle=False, num_workers=0)
    num_train_instances = len(train_data_Bmode)

    test_data_Bmode = torchvision.datasets.ImageFolder(root=file_name+'bmode/val', transform=data_transform)
#     test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=b_size, shuffle=False, num_workers=0)
    num_test_instances = len(test_data_Bmode)


    # phase1 data loader
    train_data_enhanced = torchvision.datasets.ImageFolder(root=file_name+'enhanced/train', transform= data_enhanced_transform)
#     train_data_R1_loader = torch.utils.data.DataLoader(train_data_R1, batch_size=b_size, shuffle=False, num_workers=0)

    test_data_enhanced = torchvision.datasets.ImageFolder(root=file_name+'enhanced/val', transform= data_enhanced_transform)
#     test_data_R1_loader = torch.utils.data.DataLoader(test_data_R1, batch_size=b_size, shuffle=False, num_workers=0)


    # phase2 data loader
    train_data_improved = torchvision.datasets.ImageFolder(root=file_name+'imp/train', transform=data_transform)
#     train_data_R4_loader = torch.utils.data.DataLoader(train_data_R4,  batch_size=b_size, shuffle=False, num_workers=0)

    test_data_improved = torchvision.datasets.ImageFolder(root=file_name+'imp/val', transform=data_transform)
#     test_data_R4_loader = torch.utils.data.DataLoader(test_data_R4, batch_size=b_size, shuffle=False, num_workers=0)


    
    dataset_tr = MyDataset(train_data_Bmode, train_data_enhanced, train_data_improved)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=b_size, shuffle=True, pin_memory=True, num_workers=4)
    
    dataset_val = MyDataset(test_data_Bmode, test_data_enhanced, test_data_improved)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=b_size, shuffle=False,pin_memory=True, num_workers=4)
    
    
    
#     # S4 data loader
#     train_data_S4= torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/train', transform=data_transform)
#     train_data_S4_loader = torch.utils.data.DataLoader(train_data_S4, batch_size=16, shuffle=False, num_workers=0)

#     test_data_S4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/test', transform=data_transform)
#     test_data_S4_loader = torch.utils.data.DataLoader(test_data_S4, batch_size=16, shuffle=False, num_workers=0)


#     # do concatenation for training data
#     samples_train = []
#     labels_train = []
#     train_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1) \
#             in zip(train_data_Bmode_loader, train_data_R1_loader,  train_data_R4_loader, train_data_S1_loader):
# #     for (samples_Bmode, labels_Bmode) in train_data_Bmode_loader:
#         train_flag += 1
#         samples1 = Variable(samples_Bmode)
#         samples2 = Variable(samples_R1)
#         samples3 = Variable(samples_R4)
#         samples4 = Variable(samples_S1)
# #         samples5 = Variable(samples_S4)
#         sample_train = torch.cat([samples1, samples2, samples3, samples4], dim=1)
        
# #         std, mean = torch.std_mean(samples1, unbiased=True)
# #         sample_train = (samples1-mean)/std
# #         sample_train = samples1

#         labels = labels_Bmode.squeeze()
#         label_train = Variable(labels)
#         if train_flag == 1:
#             samples_train = sample_train
#             labels_train = label_train
#         if train_flag > 1:
#             samples_train = torch.cat([samples_train, sample_train], dim=0)
#             labels_train = torch.cat([labels_train, label_train], dim=0) 


#     # do concatenation for test data
#     samples_test = []
#     labels_test = []
#     test_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1) \
#             in zip(test_data_Bmode_loader, test_data_R1_loader, test_data_R4_loader, test_data_S1_loader):
# #     for (samples_Bmode, labels_Bmode) in test_data_Bmode_loader:
#         test_flag += 1
#         samples1 = Variable(samples_Bmode)
#         samples2 = Variable(samples_R1)
#         samples3 = Variable(samples_R4)
#         samples4 = Variable(samples_S1)
# #         samples5 = Variable(samples_S4)
#         sample_test = torch.cat([samples1, samples2, samples3, samples4], dim=1)

# #         std, mean = torch.std_mean(samples1, unbiased=True)
# #         sample_test = (samples1-mean)/std
# #         sample_test = samples1

#         labels = labels_Bmode.squeeze()
#         label_test = Variable(labels)
#         if test_flag == 1:
#             samples_test = sample_test
#             labels_test = label_test
#         if test_flag > 1:
#             samples_test = torch.cat([samples_test, sample_test], dim=0)
#             labels_test = torch.cat([labels_test, label_test], dim=0)

#     return samples_train, labels_train, samples_test, labels_test, num_train_instances, num_test_instances
#     return train_data_Bmode_loader, test_data_Bmode_loader, train_data_R1_loader, test_data_R1_loader, \
#             train_data_R4_loader, test_data_R4_loader, train_data_S1_loader, test_data_S1_loader, \
#             num_train_instances, num_test_instances

    return target_names, loader_tr, loader_val, num_train_instances, num_test_instances

