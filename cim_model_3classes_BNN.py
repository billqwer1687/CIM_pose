import torch
import numpy as np
import cv2, os, sys
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
from BinaryNetpytorch.models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from BinaryNetpytorch.models.binarized_modules import  Binarize,HingeLoss
import seaborn as sns
import random
batch_size = 8
num_epoch = 10

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


train_tfm = transforms.Compose([
    #transforms.Grayscale(),
    #transforms.RandomHorizontalFlip(),
	#transforms.RandomResizedCrop((40,30)),
	#transforms.RandomCrop((40,30)),
	#transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.RandomResizedCrop((40,30)),
	#transforms.TenCrop((40,30)),
    #transforms.Normalize(0.5,0.5),
])

test_tfm = transforms.Compose([
    #transforms.Grayscale(),
    transforms.ToTensor()
    ])
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.cnn_layers = nn.Sequential(
			# BinarizeConv2d(in_channels=1, out_channels=128, kernel_size=9, padding=9//2, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # BinarizeConv2d(in_channels=128, out_channels=64, kernel_size=1, padding=1//2, bias=False),
            # nn.BatchNorm2d(64),
			#input_size(1,30,40)
			BinarizeConv2d(1, 128, 3, 1), #output_size(16,28,38)
			nn.BatchNorm2d(128),
			nn.ReLU(),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(16,14,19)

			BinarizeConv2d(128, 64, 3, 1), #output_size(24,12,17)
			nn.BatchNorm2d(64),
			nn.ReLU(),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(24,6,8)

			BinarizeConv2d(64, 32, 3, 1), #output_size(32,4,6)
			nn.BatchNorm2d(32),
			nn.ReLU(),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #ouput_size(32,2,3)
			#nn.LogSoftmax(),
			BinarizeConv2d(32, 3, (3,2), 1) #ouput_size(4,2,3) without max :(32,24,34)
						
			)
		
			

	def forward(self, x):
		x = self.cnn_layers(x)
		#x = x.flatten(1)
		#x = self.fc_layers(x)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		#print(x.shape)
		#x = nn.LogSoftmax(x)
		#print(x)

		return x 

def main():
	train_set = DatasetFolder("./dataset/data_0711/grideye/train", loader=lambda x: Image.open(x), extensions="bmp", transform=train_tfm)
	test_set = DatasetFolder("./dataset/data_0711/grideye/test", loader=lambda x: Image.open(x), extensions="bmp", transform=test_tfm)
	val_set =  DatasetFolder("./dataset/data_0711/grideye/train", loader=lambda x: Image.open(x), extensions="bmp", transform=test_tfm)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

	save_path = 'models.ckpt'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Classifier().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	criterion = nn.CrossEntropyLoss()
	best_accuracy = 0.0
	for epoch in range(num_epoch):
		running_loss = 0.0
		total = 0
		correct = 0
		for  i, data in enumerate(train_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			#print(labels)
			optimizer.zero_grad()

			outputs = model(inputs)
			#print(outputs.shape)

			loss = criterion(outputs, labels)
			loss.backward()

			for p in list(model.parameters()):
				if hasattr(p,'org'):
					p.data.copy_(p.org)

			optimizer.step()

			for p in list(model.parameters()):
				if hasattr(p,'org'):
					p.org.copy_(p.data.clamp_(-1,1))

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
			for  i, data in enumerate(val_loader):
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)
				_,predicted = torch.max(outputs.data,1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			val_acc = correct / total
		
		if val_acc > best_accuracy:
			best_accuracy = val_acc
			torch.save(model.state_dict(), save_path)
			print("Save Model")


		print(f"[ Val | {epoch + 1:03d}/{num_epoch:03d} ]  acc = {val_acc:.5f}")

	model = Classifier().to(device)
	model.load_state_dict(torch.load(save_path))
	model.eval()
	stat = np.zeros((3,3))
	with torch.no_grad():
		correct = 0
		total = 0
		print(model)
		for i, data in enumerate(test_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			#print(outputs.data)
			_,predicted = torch.max(outputs.data,1)
			#print(predicted)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			for k in range(len(predicted)):
				if predicted[k] != labels[k]:
					img = inputs[k].mul(255).byte()
					img = img.cpu().numpy().squeeze(0)
					img = np.moveaxis(img, 0, -1)
					
					predict = predicted[k].cpu().numpy()
					label = labels[k].cpu().numpy()
					path = "test_result/predict:"+str(predict)+"_labels:"+str(label)+".jpg"
					stat[int(label)][int(predict)] += 1
		ax = sns.heatmap(stat, linewidth=0.5)
		plt.xlabel('Prediction')
		plt.ylabel('Label')
		plt.savefig('heatmap.jpg')
			#print(predicted)
			#print("labels:",labels)
		print('Test Accuracy:{} %'.format((correct / total) * 100))






if __name__ == '__main__':
	main()
