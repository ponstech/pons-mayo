import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
from fusion_models.early_fusion import *
# from fusion_models.early_fusion_attention import *
import random
import glob, os
from torchvision import transforms
import torchvision
from sklearn.metrics import classification_report
# from MobileNetV2 import *
import torchvision.transforms.functional as F
from PIL import Image

torch.manual_seed(2021)  # cpu
torch.cuda.manual_seed(2021)  # gpu
np.random.seed(2021)  # numpy
random.seed(2021)  # random and transforms
torch.backends.cudnn.deterministic = True  


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
cuda = torch.device('cuda:0')

file_name = '../datasets/anatomy/bmode/test/'
# file_name = '../datasets/endo_xiao/test/'

txt_name = 'scores.txt'
b_size = 16
num_cls = 6
input_ch = 3

# fold = 4

model_name = 'msc_anatomy.pt'
# model_name = 'msc_liver_mixed_k' + str(fold) + '_best.pt'
    
from PIL import Image
import pandas as pd

class FusionDataset(object):
    def __init__(self, csv_file, dataset_1, transform1 = None):
        super(FusionDataset, self).__init__()
        
        self.frame = pd.read_csv(csv_file, header = None)
        self.transform1 = transform1
        self.dataset_1 = dataset_1
                  
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):        
        cat = self.frame.loc[idx][0].split(' ')
        
        img_path_1 = os.path.join('..', cat[1], self.dataset_1, cat[2])
        label = int(cat[3])
        img_1 = Image.open(img_path_1).convert('RGB')
        
        # if self.transform is not None:
        img_1 = self.transform1(img_1)          
        return img_1, label

    
def data_load():
    # load data
    data_transform = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.Grayscale(1),
                transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST)
                
        ])

    test_data_Bmode = torchvision.datasets.ImageFolder(root= file_name, transform=data_transform)
    print(test_data_Bmode.class_to_idx, '\n')
    target_names = list(test_data_Bmode.class_to_idx)
    test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=b_size, shuffle=False, num_workers=0)
    num_test_instances = len(test_data_Bmode)
    
    
#     file_test = os.path.join('../../Multi-Scale-Feature-Fusion/Fus-CNNs_COVID-19_US', 'test_ds_'+str(fold)+'.txt')
#     test_data_Bmode_loader =  torch.utils.data.DataLoader(
#                                 FusionDataset(file_test, 'org', data_transform), 
#                                 batch_size=b_size, 
#                                 shuffle=True, 
#                                 num_workers=2)   
                    
#     num_test_instances = len(test_data_Bmode_loader.dataset)

#     # do concatenation for training data
#     samples_test = []
#     labels_test = []
#     test_flag = 0

#     for (samples_Bmode, labels_Bmode) in test_data_Bmode_loader:
# #         print(samples_Bmode[:,:,110,110]*255)
        
#         test_flag += 1
#         samples1 = Variable(samples_Bmode)
        
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

#     return samples_test, labels_test, test_data_Bmode, num_test_instances
    return target_names, test_data_Bmode_loader, num_test_instances, test_data_Bmode

def worker_init_fn(worker_id):
    np.random.seed(2021 + worker_id)

# samples_test, labels_test, test_data_Bmode, num_test_instances = data_load()
target_names, dataset_test_loader, num_test_instances, test_data_Bmode = data_load()

# # data for testing
# dataset_test = Data.TensorDataset(samples_test, labels_test)
# # identify data loader
# dataset_test_loader = Data.DataLoader(dataset=dataset_test, batch_size=b_size, shuffle=False, num_workers=0,
#                                       pin_memory=True, worker_init_fn=worker_init_fn)

if __name__ == '__main__':
    
    msresnet = MSResNet(input_channel=input_ch, layers=[1, 1, 1], num_classes=num_cls)
#     print(msresnet)
#     msresnet = mobilenet_v2()
    msresnet = msresnet.cuda()
    
    msresnet.load_state_dict(torch.load('models/' + model_name))
    msresnet.eval()   # Set model to evaluate mode
    
    y_true = []
    y_pre = []
    correct_test = 0
    
#     f = open(txt_name, 'w')
    for i, (samples, labels) in enumerate(dataset_test_loader):
        samplesV = Variable(samples.cuda())
#         samplesX0 = samplesV[:, 0:1, :, :]
        samplesX0 = samplesV
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        predict_label = msresnet(samplesX0)
#         print(predict_label, predict_label.data.max(1))
        prediction = predict_label.data.max(1)[1]
        correct_test += prediction.eq(labelsV.data.long()).sum()
        prediction = prediction.data.cpu()
        y_true.extend(labels.cpu().numpy())
        y_pre.extend(prediction.cpu().numpy())
        
#         scores = predict_label.data.cpu()
#         prediction = prediction.numpy()
        
#         for j in range (len(prediction)):
#             if prediction[j] == 0:
#                 f.write (f'{test_data_Bmode.samples[j+16*i][0][2:]}: CORRECT\n\n')
#             else:
#                 f.write (f'{test_data_Bmode.samples[j+16*i][0][2:]}: WRONG\n\n')
#             if 'kidney' in test_data_Bmode.samples[j+b_size*i][0][2:]:    
#                 f.write (f'{test_data_Bmode.samples[j+b_size*i][0][2:]}: {prediction[j]}\n')
#             f.write (f'{test_data_Bmode.samples[j+b_size*i][0][2:]}: {scores[j]}\n')
            
#     f.close()
    
#     print("Test accuracy:", (100 * float(correct_test) / num_test_instances))
    print (f'Model name: {model_name}\n')
#     print (f'Accuracy: {sum(1 for x,y in zip(a,b) if x == y) / len(a)}\n')
    print(classification_report(y_true, y_pre, digits=4, target_names=target_names, zero_division=0), '\n')
    print('********************************************************\n')
#     y_true = np.array(y_true)
#     y_pre = np.array(y_pre)
#     TN0 = np.sum((y_true==1) & (y_pre==1))
#     FN0 = np.sum((y_true==0) & (y_pre==1))
#     npv0 = round(TN0/(TN0+FN0), 4)
#     print(f'Negative Predictive Value_cls0: {npv0}\n') 
    
#     TN1 = np.sum((y_true==0) & (y_pre==0))
#     FN1 = np.sum((y_true==1) & (y_pre==0))
#     npv1 = round(TN1/(TN1+FN1), 4)
#     print(f'Negative Predictive Value_cls1: {npv1}\n') 

