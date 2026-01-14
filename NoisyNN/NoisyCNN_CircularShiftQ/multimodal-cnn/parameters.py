import argparse
import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', default= 'impulse', type = str, help='noise types: linear, gaussian, impulse')
parser.add_argument('--gau_mean', default= 0.0, type= float, help='gaussian mean,[-1,1,0.5]')
parser.add_argument('--gau_var', default= 1.0, type= float, help='gaussian variance [0,2,0.5]')
parser.add_argument('--noise_str', default= 0.1, type= float, help='noise strengthen, [0,1]')
parser.add_argument('--noise_layer', default= 4, type = int, help='which layer to add noise, [1, 4]')
parser.add_argument('--sub_noisy_layer', default= 3, type = int, help='the specific layer to add noise, according to RseNet architecture')
parser.add_argument('--epoch', default= 100, type = int, help='training epoches')
parser.add_argument('--lr', default= 0.001, type = float, help='learning rate')
parser.add_argument('--batch_size', default= 128, type = int, help='batch size')
parser.add_argument('--class_num', default= 6, type = int, help='class numbers')
parser.add_argument('--datasets', default= 'ImageNet', type = str, help='what dataset to use')
parser.add_argument('--resnet', default= 'resnet34', type = str, help='resnet architecture')
parser.add_argument('--pretrain', default= True, type = bool, help='use pretrain model or not')
parser.add_argument('--tinyImagenet_path', default= '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_TinyImageNet/TinyImageNet/', type = str, help='tiny imagenet path')
parser.add_argument('--Imagenet_path', default= '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_ImageNet/ImageNet1K/', type = str, help='imagenet path')
parser.add_argument('--gpu_id', default= '1', type = str, help='select gpus')
parser.add_argument('--multimodal_data_path', type=str, default='/content/drive/MyDrive/dataset', help='path to multimodal ultrasound data')
parser.add_argument('--input_size', type=int, default=256, help='input image size for multimodal CNN')
parser.add_argument('--resume', type=str, default='', help='path to a checkpoint .pt to resume training from')
parser.add_argument('--auto_resume_dir', type=str, default='/content/drive/MyDrive/NoisyCNN_CircularShiftQ/saved_models/multimodal', help='directory to auto-pick latest checkpoint from if --resume empty')
#ResNet18 [2,2,2,2]
#ResNet34 [3,4,6,3]
#ResNet50 [3,4,6,3]
#ResNet101 [3,4,23,3]
#ResNet152 [3,8,36,3]
#sub_noisy_layer range from [1,n], n is the number of convs noise_layer, e.g., if ResNet152, noise_layer = 3, 36 sub layers, then n = 36.
args = parser.parse_args()


