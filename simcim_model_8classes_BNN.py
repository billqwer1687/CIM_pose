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
from SimBinaryNetpytorch.models.binarized_modules import  BinarizeLinear, CimSimConv2d
from BinaryNetpytorch.models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from BinaryNetpytorch.models.binarized_modules import  Binarize,HingeLoss
import seaborn as sns
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 8
num_epoch = 100

seed = 333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.cnn_layers1 = nn.Sequential(
			
			# CimSimConv2d(in_channels=1, out_channels=128, kernel_size=3, bias=False),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.5),
            # CimSimConv2d(in_channels=128, out_channels=64, kernel_size=3, bias=False),
            # nn.BatchNorm2d(64),
			# nn.LeakyReLU(0.5),
			
			#input_size(1,30,40)
			CimSimConv2d(1, 128, 3, 1), #output_size(16,66,66)
			#nn.BatchNorm2d(128),
			nn.LeakyReLU(0.5),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(16,33,33)
		)
		self.cnn_layers2 = nn.Sequential(

			CimSimConv2d(128, 64, 3, 1), #output_size(24,31,31)
			#nn.BatchNorm2d(64),
			nn.LeakyReLU(0.5),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #output_size(24,15,15)
		)

		self.cnn_layers3 = nn.Sequential(

			CimSimConv2d(64, 32, 3, 1), #output_size(32,13,13)
			#nn.BatchNorm2d(32),
			nn.LeakyReLU(0.5),
			#nn.Dropout(0.2),
			nn.MaxPool2d(kernel_size = 2), #ouput_size(32,6,6)
			#nn.LogSoftmax(),
			BinarizeConv2d(32, 8, (3,2), 1) #ouput_size(4,2,3) without max :(32,24,34)
						
		)
		
			

	def forward(self, x):
		#print(x)
		x = self.cnn_layers1(x)
		#print(x)
		x = self.cnn_layers2(x)
		x = self.cnn_layers3(x)
		#print(x)
		#x = x.flatten(1)
		#x = self.fc_layers(x)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		#print(x.shape)
		#x = nn.LogSoftmax(x)
		#print(x)

		return x 

class Cls_Dataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target
    def __getitem__(self, index):
        return self.input[index], self.target[index]
    def __len__(self):
        return len(self.input)

def Load_data(path, cls):
	data = []
	label = []
	f = 0
	for c in range(cls):
		t = 0
		img = []
		data_path = path + "/0"+str(c)+"/"
		for filename in os.listdir(data_path):
			# with open(data_path + "/" + filename, "rb") as f_in:
			
			tmp_input = cv2.imread(data_path + filename,cv2.IMREAD_UNCHANGED)
			#print(tmp_input)
			tmp_input = cv2.resize(tmp_input, (30,40), interpolation = cv2.INTER_AREA)
			tmp_input = tmp_input//2
			tmp_input = tmp_input - 63
			tmp_input = tmp_input[:, :, np.newaxis]
			tmp_input = tmp_input.transpose(2,0,1)
			
				
			
			if img is not None:
				img.append(tmp_input)
			else:
				img = tmp_input	
		label_tmp = np.full((len(img),1),c)
		
		if f != 0:
			data = np.append(data,img,axis=0)
		else:
			data = img
		if f != 0:
			label = np.append(label,label_tmp,axis=0)
		else:
			label = label_tmp
		f = 1



	label = np.squeeze(label)

	label = np.array(label)
	#print(data)
	data = np.array(data)
	print(label.shape)
	data = data.astype('float32')
	
	return data, label

def main():
	# test_set = DatasetFolder("./dataset/8cls_srcnnimg/test", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
	# val_set =  DatasetFolder("./dataset/8cls_srcnnimg/train", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
	
	
	train_data, train_label = Load_data("./dataset/8cls_grideye/train",8)
	train = Cls_Dataset(train_data, train_label)
	train_loader = DataLoader(
	        train, batch_size=100, shuffle=True,
	        num_workers=4, pin_memory=True, drop_last=True
	    )

	test_data, test_label = Load_data("./dataset/8cls_grideye/test",8)

	test = Cls_Dataset(test_data, test_label)
	
	test_loader = DataLoader(
	        test, batch_size=100, shuffle=True,
	        num_workers=4, pin_memory=True, drop_last=True
	    )

	val_data, val_label = Load_data("./dataset/8cls_grideye/val",8)

	val = Cls_Dataset(val_data, val_label)
	
	val_loader = DataLoader(
	        val, batch_size=100, shuffle=True,
	        num_workers=4, pin_memory=True, drop_last=True
	    )
	
	

	
	save_path = 'models.ckpt'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
			optimizer.zero_grad()

			outputs = model(inputs)
			#print(outputs.shape)

			loss = criterion(outputs, labels)
			loss.backward()
			#print(model)
			#print(model.cnn_layers3[3].weight.grad)


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
	stat = np.zeros((8,8))
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
			#print(labels)
			#print(outputs.size())
			

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
		print('Test Accuracy:{} %'.format((correct / total) * 100))
if __name__ == '__main__':
	main()
