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
import torchvision.models as models
batch_size = 32
num_epoch = 10

train_tfm = transforms.Compose([
    transforms.Grayscale(),
	transforms.RandomResizedCrop((40,30)),
	transforms.Resize((40, 30)),
    transforms.ToTensor(),
    
	#transforms.TenCrop((40,30)),
    #transforms.Normalize(0.5,0.5),
])

test_tfm = transforms.Compose([
    transforms.Grayscale(),
	transforms.Resize((40, 30)),
    transforms.ToTensor()
    ])
'''
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.cnn_layers = nn.Sequential(
			#input_size(1,30,40)
			nn.Conv2d(1, 16, 3, 1), #output_size(16,28,38)
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(16,14,19)

			nn.Conv2d(16, 24, 3, 1), #output_size(24,12,17)
			nn.BatchNorm2d(24),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(24,6,8)

			nn.Conv2d(24, 32, 3, 1), #output_size(32,4,6)
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2) #ouput_size(32,2,3)


			)
		self.fc_layers = nn.Sequential(
			nn.Linear(32 * 2 * 3, 32),
			nn.ReLU(),
			nn.Dropout(0.2),


			nn.Linear(32,8)
			)

	def forward(self, x):
		x = self.cnn_layers(x)

		x = x.flatten(1)

		x = self.fc_layers(x)

		return x 
'''
def main():
	train_set = DatasetFolder("./dataset/data_0705/lepton/train", loader=lambda x: Image.open(x), extensions="bmp", transform=train_tfm)
	test_set = DatasetFolder("./dataset/data_0705/lepton/test", loader=lambda x: Image.open(x), extensions="bmp", transform=test_tfm)
	
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = models.resnet18()
	model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
	model.fc = nn.Linear(512, 3)
	model = model.to(device)
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epoch):
		##Training
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
		
		



	##Testing
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
