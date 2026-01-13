
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
from einops import rearrange
import albumentations as A


class BP4D(data.Dataset):
    def __init__(self, csv_path, args, transform=None):
        # specify annotation file for dataset
        # self.data = pd.read_csv(csv_path)
        self.data = pd.read_csv(csv_path, on_bad_lines='skip') #sc
        modality_list = args.modalities
        self.transform = transform

        self.need_texture = 'texture' in modality_list
        self.need_depth = 'depth' in modality_list
        self.need_landmark = 'landmark' in modality_list
        self.need_thermal = 'thermal' in modality_list

        if self.need_texture:
            self.texture_root = args.texture_root
            
        if self.need_depth:
            self.depth_root = args.depth_root

        if self.need_thermal:
            self.thermal_root = args.thermal_root
        
        self.has_label = self.data.shape[1]>1

    def __len__(self): 
        return len(self.data)


    def read_landmark(self, idx):
        x, y, w, h = self.data.iloc[idx,13:17].values
        landmark = self.data.iloc[idx, 17:17+49].values
        land_np = np.zeros([49,2])
        for i in range(49):
            land = landmark[i]
            land_x, land_y = land.split(' ')
            land_np[i,0] = (float(land_x) - x) / w *224
            land_np[i,1] = (float(land_y) - y) / h *224
        landmark = land_np
        return landmark.flatten()

    def __getitem__(self, idx):
        

        out_dic = {}

        image_name = self.data.iloc[idx, 0]       

        
        if self.need_texture:
            # image_path = os.path.join(self.texture_root, image_name+'.jpg')
            image_path = os.path.join(self.texture_root, image_name) #sc
            img = np.array(Image.open(image_path), np.float32)
            img = self.transform['rgb'](image=img)['image']
            out_dic['texture'] = img

        depth = None
        if self.need_depth:
            # depth_path = os.path.join(self.depth_root, image_name+'.jpg')
            depth_path = os.path.join(self.depth_root, image_name) #sc
            # depth = np.array(Image.open(depth_path), np.float32)[:,:,0]
            depth = np.array(Image.open(depth_path).convert('L'), np.float32) #sc
            depth = self.transform['gray'](image=depth)['image']
            out_dic['depth'] = depth
            
            # depth = torch.unsqueeze(depth[0], 0) # first channel
            # if 'texture' in out_dic.keys():
            #     x = torch.cat([out_dic['texture'], depth], dim=0)
            #     out_dic['texture'] = x
            # else:
            #     out_dic['texture'] = depth.repeat(3, 1, 1)
        thermal = None
        if self.need_thermal:
            # sub, task, frame = image_name.split('/')
            # thermal_name = f'{sub}/{task}/{int(frame)}' # self.data['Thermal'].iloc[idx]
            # thermal_path = os.path.join(self.thermal_root, thermal_name+'.jpg')
            thermal_path = os.path.join(self.thermal_root, image_name) #sc
            thermal = np.array(Image.open(thermal_path).convert('L'), np.float32)
            thermal = self.transform['gray'](image=thermal)['image']
            out_dic['thermal'] = thermal

        
        if self.need_landmark:
            landmark = self.read_landmark(idx)
            out_dic['landmark'] = torch.Tensor(landmark)

        
        out_dic['name'] = image_name

        if self.has_label:
            label = self.data.iloc[idx, 1:3].values
            label = label.astype(np.float32)
            out_dic['label'] = torch.Tensor(label)

        return out_dic

