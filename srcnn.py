import torch
import os
import cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.quantized import functional as qF
from torchvision import transforms
import numpy as np
import math
from SimBinaryNetpytorch.models.binarized_modules import  CimSimConv2d
from BinaryNetpytorch.models.binarized_modules import  BinarizeConv2d
import copy
from skimage.metrics import structural_similarity as ssim
batch_size = 100
num_epoch = 50

train_tfm = transforms.Compose([
    #transforms.ToTensor(),
])
class SRCNN_Dataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target
    def __getitem__(self, index):
        return train_tfm(self.input[index]), train_tfm(self.target[index])
    def __len__(self):
        return len(self.input)

class SRCNN(nn.Module):
    def __init__(self, num_channels=1, f1=7, f2=1, f3=5):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Sequential(
            BinarizeConv2d(in_channels=num_channels, out_channels=128, kernel_size=f1, padding=f1//2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.5),
        )
        self.layer2 = nn.Sequential(
            BinarizeConv2d(in_channels=128, out_channels=64, kernel_size=f2, padding=f2//2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.5),
        )
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=f3, padding=f3//2, bias=False)
        
    def forward(self, x):
        #print(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        #print("layer2",torch.min(x),torch.max(x))
        return out


def main():
    #######train load######
    image_LR = []
    cls = 8
    for c in range(cls):
        path_LR = "./dataset/down_8cls_lepton/train/0"+str(c)
        for filename in os.listdir(path_LR):
            img_LR = cv2.imread(os.path.join(path_LR,filename),cv2.IMREAD_UNCHANGED)
            img_LR = img_LR[:, :, np.newaxis]

            if img_LR is not None:
                img_LR = img_LR.transpose(2,0,1)
                image_LR.append(img_LR)
    
    image_HR = []
    for c in range(cls):    
        path_HR = "./dataset/8cls_lepton/train/0"+str(c)
        for filename in os.listdir(path_HR):
            img_HR = cv2.imread(os.path.join(path_HR,filename),cv2.IMREAD_UNCHANGED)
            img_HR = img_HR[:, :, np.newaxis]
            if img_HR is not None:
                img_HR = img_HR.transpose(2,0,1)
                image_HR.append(img_HR)
    

    #######test load######
    image_LR_test = []
    for c in range(cls):
        path_LR_test = "./dataset/down_8cls_lepton/test/0"+str(c)
        for filename in os.listdir(path_LR_test):
            img_LR_test = cv2.imread(os.path.join(path_LR_test,filename),cv2.IMREAD_UNCHANGED)
            img_LR_test = img_LR_test[:, :, np.newaxis]
            if img_LR_test is not None:
                img_LR_test = img_LR_test.transpose(2,0,1)
                image_LR_test.append(img_LR_test)
   

    image_HR_test = []
    for c in range(cls):
        path_HR_test = "./dataset/8cls_lepton/test/0"+str(c)
        for filename in os.listdir(path_HR_test):
            img_HR_test = cv2.imread(os.path.join(path_HR_test,filename),cv2.IMREAD_UNCHANGED)
            img_HR_test = img_HR_test[:, :, np.newaxis]
            
            if img_HR_test is not None:
                img_HR_test = img_HR_test.transpose(2,0,1)
                image_HR_test.append(img_HR_test)

    #######val load######
    image_LR_val = []
    for c in range(cls):
        path_LR_val = "./dataset/down_8cls_lepton/val/0"+str(c)
        for filename in os.listdir(path_LR_val):
            img_LR_val = cv2.imread(os.path.join(path_LR_val,filename),cv2.IMREAD_UNCHANGED)
            img_LR_val = img_LR_val[:, :, np.newaxis]
            if img_LR_val is not None:
                img_LR_val = img_LR_val.transpose(2,0,1)
                image_LR_val.append(img_LR_val)
   

    image_HR_val = []
    for c in range(cls):
        path_HR_val = "./dataset/8cls_lepton/val/0"+str(c)
        for filename in os.listdir(path_HR_val):
            img_HR_val= cv2.imread(os.path.join(path_HR_val,filename),cv2.IMREAD_UNCHANGED)
            img_HR_val = img_HR_val[:, :, np.newaxis]
            
            if img_HR_val is not None:
                img_HR_val = img_HR_val.transpose(2,0,1)
                image_HR_val.append(img_HR_val)

    train = SRCNN_Dataset(image_LR, image_HR)
    t = np.array(image_LR)
    #print(t.shape)
    train_loader = DataLoader(
            train, batch_size=100, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
    test = SRCNN_Dataset(image_LR_test, image_HR_test)
    test_loader = DataLoader(
            test, batch_size=100, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )

    val = SRCNN_Dataset(image_LR_val, image_HR_val)
    val_loader = DataLoader(
            val, batch_size=100, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRCNN().to(device)
    save_path = './models/srcnn_8cls.ckpt'
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    best_accuracy = 100000
    for epoch in range(num_epoch):
        running_loss = 0.0
        for  i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(torch.max(outputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = running_loss / (i+1)
        print(f"[ Train | avg_loss = {train_acc:.5f}")
        ###validation###
        model.eval()
        running_loss = 0
        for  i, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_acc = running_loss / (i+1)

        if val_acc < best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            print("Save Model")


        print(f"[ Val | avg_loss = {val_acc:.5f}")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        running_loss = 0
        ssim_const = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ssim_in = copy.deepcopy(inputs)
            ssim_out = copy.deepcopy(outputs)
            out_img = copy.deepcopy(outputs)
            out_img -= (out_img.min())
            out_img *= (255/out_img.max())
            inputs -= (inputs.min())
            inputs *= (255/inputs.max())
            img_LR = inputs[0].to('cpu').numpy().transpose(1,2,0)
            img_HR = (torch.round(out_img[0])).to('cpu').numpy().transpose(1,2,0)
            img_TR = labels[0].to('cpu').numpy().transpose(1,2,0)
            cv2.imwrite('My Image_LR.jpg', img_LR)
            cv2.imwrite('My Image_HR.jpg', img_HR)
            for k in range(ssim_in.shape[0]):
                im1 = ssim_in[k].to('cpu').numpy().transpose(1,2,0)
                im2 = ssim_out[k].to('cpu').numpy().transpose(1,2,0)
                #print(im2.shape)
                ssim_const += ssim(im1, im2, data_range = im2.max() - im2.min(),multichannel=True)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        test_acc = running_loss / (i+1)
        mse = np.mean((img_HR - img_TR) ** 2 )
        psnr = 10 * math.log10(255*255/mse)
        print(psnr)
        print(ssim_const/800)
           
        print(f"[ Test | avg_loss = {test_acc:.5f}")
    


if __name__ == '__main__':
    main()