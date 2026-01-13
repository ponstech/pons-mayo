import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
	def __init__(self, in_channels: int = 1, num_classes: int = 2, feature_dim: int = 512):
		super(CustomCNN, self).__init__()
		# Convolutional Layers
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(128)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Assuming input 224x224 after three poolings -> 28x28
		self.fc1 = nn.Linear(128 * 28 * 28, feature_dim)
		self.fc2 = nn.Linear(feature_dim, num_classes)
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
		x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
		x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
		x = x.view(x.size(0), -1)
		x = F.leaky_relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

	def extract_features(self, x):
		x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
		x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
		x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		return x
