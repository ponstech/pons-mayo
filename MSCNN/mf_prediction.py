import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
from fusion_models.late_fusion import *
import random
import glob
from torchvision import transforms
import torchvision
from sklearn.metrics import classification_report
# import torchvision.transforms.functional as F
from PIL import Image

import logging
import pickle as pkl
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
import time
import timm
from data import MyDataset
import torch.nn.functional as F
import pandas as pd

torch.manual_seed(2021)  # cpu
torch.cuda.manual_seed(2021)  # gpu
np.random.seed(2021)  # numpy
random.seed(2021)  # random and transforms
torch.backends.cudnn.deterministic = True  

import os 
cuda = torch.device('cuda:0')

    
txt_name = 'results.txt'
b_size = 16
num_cls = 2
input_chnl = 1
file_name = "./"  # test data path

model_name = './' # .pt model path 

def data_load():    
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
    test_data_Bmode = torchvision.datasets.ImageFolder(root=file_name+'bmode/test', transform=data_transform)
    print(test_data_Bmode.class_to_idx, '\n')
    target_names = list(test_data_Bmode.class_to_idx)
    num_test_instances = len(test_data_Bmode)

    # phase1 data loader
    test_data_enhanced = torchvision.datasets.ImageFolder(root=file_name+'enhanced/test', transform=data_enhanced_transform)

    # phase2 data loader
    test_data_improved = torchvision.datasets.ImageFolder(root=file_name+'imp/test', transform=data_transform)

   
    
    dataset_ts = MyDataset(test_data_Bmode, test_data_enhanced, test_data_improved)
    loader_ts = torch.utils.data.DataLoader(dataset_ts, batch_size=b_size, shuffle=False)

#     # do concatenation for training data
#     samples_test = []
#     labels_test = []
#     test_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1) \
#             in zip(test_data_Bmode_loader, test_data_R1_loader, test_data_R4_loader, test_data_S1_loader):
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

#     return samples_test, labels_test, test_data_Bmode, num_test_instances
    return target_names, loader_ts, num_test_instances, test_data_Bmode


def worker_init_fn(worker_id):
    np.random.seed(2021 + worker_id)

# test_data_Bmode_loader, test_data_R1_loader, test_data_R4_loader, test_data_S1_loader, num_test_instances = data_load()
target_names, dataset_test_loader, num_test_instances, test_data_Bmode = data_load()
# labels_test = labels_test.unsqueeze(0)

# # data for testing
# dataset_test = Data.TensorDataset(samples_test, labels_test)
# # identify data loader
# dataset_test_loader = Data.DataLoader(dataset=dataset_test, batch_size=b_size, shuffle=False, num_workers=0,
#                                       pin_memory=True, worker_init_fn=worker_init_fn)

if __name__ == '__main__':
    
    msresnet = MSResNet(input_channel=input_chnl, layers=[1, 1, 1], num_classes=num_cls)
    msresnet = msresnet.cuda()
    
    msresnet.load_state_dict(torch.load("./" + model_name)) # models path
    msresnet.eval()   # Set model to evaluate mode

    y_true = []
    y_pre = []
    y_pred_proba = []
    correct_test = 0

    for i, ((samples_Bmode, labels_Bmode), (samples_Enhanced, labels_Enhanced), (samples_Improved, labels_Improved)) \
        in enumerate(dataset_test_loader):

        samplesX0 = Variable(samples_Bmode.cuda())
        samplesX1 = Variable(samples_Enhanced.cuda())
        samplesX2 = Variable(samples_Improved.cuda())
        
        labels = labels_Bmode.squeeze()
        labelsV = Variable(labels.cuda())
#         predict_label = msresnet(samplesX0)
        predict_label = msresnet(samplesX0, samplesX1, samplesX2)
#         print(predict_label, predict_label.data.max(1))
        if predict_label.dim() == 1:
            prediction = predict_label.data.argmax()
        else:
            prediction = predict_label.data.max(1)[1]
    
        #prediction = predict_label.data.max(1)[1]
        # Softmax ile olasılıkları al (y_pred_proba)
            probs = F.softmax(predict_label, dim=1)  # shape: (N, 2)
            correct_test += prediction.eq(labelsV.data.long()).sum()
            prediction = prediction.data.cpu()
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(prediction.cpu().numpy())
            y_pred_proba.extend(probs.cpu().detach().numpy())  # each line [prob_benign, prob_malignant]
    
    
    print("Test accuracy:", (100 * float(correct_test) / num_test_instances))
    print (f'Model name: {model_name}\n')
    print(classification_report(y_true, y_pre, digits=4, target_names=target_names, zero_division=0))
    print('********************************************************\n')
    
 # Convert to numpy arrays
y_true_np = np.array(y_true)
y_pre_np = np.array(y_pre)
y_pred_proba_np = np.array(y_pred_proba)
                                               
# Confidence (highest probability)
confidence = y_pred_proba_np.max(axis=1)                           
                                               
# Separate probability columns for classes 0 and 1
prob_class_0 = y_pred_proba_np[:, 0]
prob_class_1 = y_pred_proba_np[:, 1]

correct = y_true_np == y_pre_np

# TP, TN, FP, FN columns
true_positive  = (y_true_np == 1) & (y_pre_np == 1)
true_negative  = (y_true_np == 0) & (y_pre_np == 0)
false_positive = (y_true_np == 0) & (y_pre_np == 1)
false_negative = (y_true_np == 1) & (y_pre_np == 0)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = ['', ''] # class names
cm = confusion_matrix(y_true, y_pre, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("./.png", dpi=300, bbox_inches='tight') # create confusion matrix png
plt.show()

with open('./.txt', 'w') as f:
    for i in range(len(y_pre)):
        label = y_pre[i]
        path = test_data_Bmode.samples[i][0][1:]
        if label == 0:
            f.write(f'{path}: \n') # class names
        else:
            f.write(f'{path}: \n') # class names

# Gets patient_id and predicted_label from results.txt file
patient_ids = []
predicted_labels = []

with open("./.txt", "r") as file:
    for line in file:
        parts = line.strip().split(": ")  #    
        if len(parts) == 2:
            path, label = parts
            filename = path.split("/")[-1]
            patient_id = filename.split("_")[0]
            patient_ids.append(patient_id)
            predicted_labels.append(label)
            

results_df = pd.DataFrame({
    "patient_id": patient_ids,
    "true_label": y_true_np,
    "pred_label": y_pre_np,
    "correct": correct,
    "pred_prob_class_0": prob_class_0,
    "pred_prob_class_1": prob_class_1,
    "confidence": confidence,
    "decision_threshold": 0.8,
    "true_positive": true_positive,
    "true_negative": true_negative,
    "false_positive": false_positive,
    "false_negative": false_negative
})

#results_df["correct"] = results_df["predicted_label"] == results_df["true_label"]

metadata_df = pd.read_csv("./.csv") # reading metadata csv
metadata_df["NFER_PID"] = metadata_df["NFER_PID"].astype(str)
results_df["patient_id"] = results_df["patient_id"].astype(str)

selected_cols = [
    "NFER_PID", "STUDY_INSTANCE_UID", "STUDY_DESCRIPTION_x",
    "SERIES_INSTANCE_UID", "SERIES_DESCRIPTION_x", "StudyDate_x", "LOCATION_SITE_NAME", "PATIENT_RACE_NAME", "NFER_AGE", "SYN_CONCEPT_NAME",
    "PATIENT_ETHNICITY_NAME", "density"
]
metadata_df = metadata_df[selected_cols]

# Combine with density information
merged_df = results_df.merge(
    metadata_df,
    left_on="patient_id",
    right_on="NFER_PID",
    how="left"
)
merged_df = merged_df.rename(columns={"LOCATION_SITE_NAME": "location"})
merged_df = merged_df.rename(columns={"PATIENT_RACE_NAME": "race"})
merged_df = merged_df.rename(columns={"PATIENT_ETHNICITY_NAME": "ethnicity"})

merged_df["patient_id"] = merged_df["NFER_PID"]
merged_df = merged_df.drop(columns=["NFER_PID"])

merged_df.to_csv("./.csv", index=False)

print("final_results_with_metadata.csv saved") 