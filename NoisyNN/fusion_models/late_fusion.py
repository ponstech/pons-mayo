import torch.nn as nn
import torch.nn.functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1], num_classes=2):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.maxpool3 = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.maxpool5 = nn.AvgPool2d(kernel_size=11, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.maxpool7 = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)

        self.fc = nn.Linear(256 * 15, num_classes)


    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)


    def forward(self, a0, a1, a2, a3, a4):
        a0 = self.conv1(a0)
        a0 = self.bn1(a0)
        a0 = self.relu(a0)
        a0 = self.maxpool(a0)

        x0 = self.layer3x3_1(a0)
        x0 = self.layer3x3_2(x0)
        x0 = self.layer3x3_3(x0)
        x0 = self.maxpool3(x0)

        y0 = self.layer5x5_1(a0)
        y0 = self.layer5x5_2(y0)
        y0 = self.layer5x5_3(y0)
        y0 = self.maxpool5(y0)

        z0 = self.layer7x7_1(a0)
        z0 = self.layer7x7_2(z0)
        z0 = self.layer7x7_3(z0)
        z0 = self.maxpool7(z0)

        a1 = self.conv1(a1)
        a1 = self.bn1(a1)
        a1 = self.relu(a1)
        a1 = self.maxpool(a1)

        x1 = self.layer3x3_1(a1)
        x1 = self.layer3x3_2(x1)
        x1 = self.layer3x3_3(x1)
        x1 = self.maxpool3(x1)

        y1 = self.layer5x5_1(a1)
        y1 = self.layer5x5_2(y1)
        y1 = self.layer5x5_3(y1)
        y1 = self.maxpool5(y1)

        z1 = self.layer7x7_1(a1)
        z1 = self.layer7x7_2(z1)
        z1 = self.layer7x7_3(z1)
        z1 = self.maxpool7(z1)

        a2 = self.conv1(a2)
        a2 = self.bn1(a2)
        a2 = self.relu(a2)
        a2 = self.maxpool(a2)

        x2 = self.layer3x3_1(a2)
        x2 = self.layer3x3_2(x2)
        x2 = self.layer3x3_3(x2)
        x2 = self.maxpool3(x2)

        y2 = self.layer5x5_1(a2)
        y2 = self.layer5x5_2(y2)
        y2 = self.layer5x5_3(y2)
        y2 = self.maxpool5(y2)

        z2 = self.layer7x7_1(a2)
        z2 = self.layer7x7_2(z2)
        z2 = self.layer7x7_3(z2)
        z2 = self.maxpool7(z2)

        a3 = self.conv1(a3)
        a3 = self.bn1(a3)
        a3 = self.relu(a3)
        a3 = self.maxpool(a3)

        x3 = self.layer3x3_1(a3)
        x3 = self.layer3x3_2(x3)
        x3 = self.layer3x3_3(x3)
        x3 = self.maxpool3(x3)

        y3 = self.layer5x5_1(a3)
        y3 = self.layer5x5_2(y3)
        y3 = self.layer5x5_3(y3)
        y3 = self.maxpool5(y3)

        z3 = self.layer7x7_1(a3)
        z3 = self.layer7x7_2(z3)
        z3 = self.layer7x7_3(z3)
        z3 = self.maxpool7(z3)

        a4 = self.conv1(a4)
        a4 = self.bn1(a4)
        a4 = self.relu(a4)
        a4 = self.maxpool(a4)

        x4 = self.layer3x3_1(a4)
        x4 = self.layer3x3_2(x4)
        x4 = self.layer3x3_3(x4)
        x4 = self.maxpool3(x4)

        y4 = self.layer5x5_1(a4)
        y4 = self.layer5x5_2(y4)
        y4 = self.layer5x5_3(y4)
        y4 = self.maxpool5(y4)

        z4 = self.layer7x7_1(a4)
        z4 = self.layer7x7_2(z4)
        z4 = self.layer7x7_3(z4)
        z4 = self.maxpool7(z4)

        out = torch.cat([x0, x1, x2, x3, x4,
                         y0, y1, y2, y3, y4,
                         z0, z1, z2, z3, z4], dim=1)

        out = out.squeeze()
        out1 = self.fc(out)

        return out1