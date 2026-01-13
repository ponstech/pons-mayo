import torch
from torchvision.models import resnet18
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_classes, au_dim=32):
        super(BaselineModel,self).__init__()
        self.num_classes = num_classes
        self.au_dim = au_dim

        resnet = resnet18(pretrained=True)
        fc_in_dim = resnet.fc.in_features 
        resnet.fc = nn.Linear(fc_in_dim, au_dim*num_classes)    
        self.res = resnet

        self.fc = nn.Sequential(
            nn.Linear(au_dim*num_classes, num_classes),
            nn.Dropout()
        )
        


    def forward(self, data):
        x = self.res(data)
        out = self.fc(x)
        return out