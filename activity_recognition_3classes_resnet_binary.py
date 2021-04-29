import torch
import numpy as np
import cv2, os, sys
import pandas as pd
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
import torchvision.models
import BinaryNetpytorch.models as models
from BinaryNetpytorch.models.binarized_modules import  BinarizeLinear,BinarizeConv2d
batch_size = 32
num_epoch = 10

train_tfm = transforms.Compose([
#	transforms.RandomHorizontalFlip(),
#	transforms.RandomResizedCrop((40,30)),
    transforms.Grayscale(),
    transforms.Resize((40, 30)),
    transforms.ToTensor(),
    #transforms.RandomResizedCrop((40,30)),
	#transforms.TenCrop((40,30)),
#    transforms.Normalize(0.5,0.5),
])
test_tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((40, 30)),
    transforms.ToTensor()
])
def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)


        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x
class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=3,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(1, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate)
        self.bn3 = nn.BatchNorm1d(3)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(64*self.inflate, 3)
def main():
	train_set = DatasetFolder("pose_data/training/labeled", loader=lambda x: Image.open(x), extensions="bmp", transform=train_tfm)
	test_set = DatasetFolder("pose_data/testing", loader=lambda x: Image.open(x), extensions="bmp", transform=test_tfm)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ResNet_cifar10(num_classes=3,block=BasicBlock,depth=18)
	model = model.to(device)
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epoch):
		running_loss = 0.0
		total = 0
		correct = 0
		for  i, data in enumerate(train_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()

			outputs = model(inputs)

			loss = criterion(outputs, labels)
			loss.backward()

			optimizer.step()

			running_loss += loss.item()
			total += labels.size(0)
			_,predicted = torch.max(outputs.data,1)
			#print(predicted)
			#print("label",labels)
			correct += (predicted == labels).sum().item()
		train_acc = correct / total

		print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {running_loss:.5f}, acc = {train_acc:.5f}")

	model.eval()

	with torch.no_grad():
		correct = 0
		total = 0

		for i, data in enumerate(test_loader):
			inputs, labels = data
			
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_,predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			#print(predicted)
			#print("labels:",labels)
		print('Test Accuracy:{} %'.format((correct / total) * 100))






if __name__ == '__main__':
	main()
