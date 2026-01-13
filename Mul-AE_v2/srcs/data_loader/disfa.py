
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np


class DISFA(data.Dataset):
    def __init__(self, csv_path, args, transform=None):
        # specify annotation file for dataset
        self.data = pd.read_csv(csv_path)
        modality_list = args.modalities
        self.transform = transform

        self.need_texture = 'texture' in modality_list
        self.need_landmark = 'landmark' in modality_list

        if self.need_texture:
            self.texture_root = args.texture_root  
        
            

    def __len__(self): 
        return len(self.data)


    def read_landmark(self, idx):
        landmark = self.data.iloc[idx, 17:17+49].values
        land_np = np.zeros([49,2])
        for i in range(49):
            land = landmark[i]
            land_x, land_y = land.split(' ')
            land_np[i,0] = round(float(land_x))
            land_np[i,1] = round(float(land_y))
        landmark = land_np
        return landmark

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 1:9].values
        label = label.astype(np.float32)

        out_dic = {}

        image_name = self.data.iloc[idx, 0]

        if self.need_texture:
            image_path = os.path.join(self.texture_root, image_name+'.jpg')
            img = np.array(Image.open(image_path), np.float32)
            img = self.transform['rgb'](image=img)['image']
            out_dic['texture'] = img

        if self.need_landmark:
            landmark = self.read_landmark(idx)
            out_dic['landmark'] = landmark

        out_dic['name'] = image_name
        label = torch.Tensor(label)
        out_dic['label'] = label

        out_dic['depth'] = torch.zeros((1, 224, 224))

        return out_dic
