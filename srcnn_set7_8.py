import torch
import os
import sys
import cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
from SimBinaryNetpytorch.models.binarized_modules import  CimSimConv2d
from BinaryNetpytorch.models.binarized_modules import  BinarizeConv2d
import copy
from skimage.metrics import structural_similarity as ssim
import argparse
np.set_printoptions(threshold=sys.maxsize)
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--outchannel", type=int,
                    help="input expected output channel")
args = parser.parse_args()
ch = args.outchannel
in_ch = ch * 2
out_ch = ch

batch_size = 1
num_epoch = 100
M = [8, 1, 1]
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
    def __init__(self, num_channels=1, f1=7, f2=5, f3=5):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Sequential(
            CimSimConv2d(in_channels=num_channels, out_channels=in_ch, kernel_size=f1, padding=f1//2, bias=False),
            #nn.BatchNorm2d(128),
        )
        self.layer2 = nn.Sequential(
            CimSimConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=f2, padding=f2//2, bias=False),
            #nn.BatchNorm2d(64),
        )
        self.act = nn.Sequential(
            nn.LeakyReLU(0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=1, kernel_size=f3, padding=f3//2, bias=False),
            #nn.BatchNorm2d(1)
        )
            
        
        
    def forward(self, x):
        #print(x)
        x = self.layer1(x)
        x = x * M[0]
        x = torch.floor(x)
        x = self.act(x)
        x[x >  63] =  63
        x[x < -63] = -63
        #print(x)
        #print("layer1",float(torch.min(x)),float(torch.max(x)))
        x = self.layer2(x)
        x *= M[1]
        x = torch.floor(x)
        x = self.act(x)
        #print("layer2",float(torch.min(x)),float(torch.max(x)))
        out = self.layer3(x)
        return out


def main():
    
    #######train load######
    image_LR = []
    path_LR = "./dataset/Set7_8/Grideye/"
    for filename in sorted(os.listdir(path_LR)):
        img_LR = cv2.imread(os.path.join(path_LR,filename),cv2.IMREAD_UNCHANGED)
        img_LR = img_LR.astype(int)
        img_LR = img_LR - np.min(img_LR)
        img_LR = img_LR * (63 / np.max(img_LR))
        img_LR = np.round(img_LR)
        img_LR = np.where(img_LR > 63, 63, img_LR)
        #print(np.min(img_LR), np.max(img_LR))
        img_LR = img_LR[:, :, np.newaxis]

        if img_LR is not None:
            img_LR = img_LR.transpose(2,0,1)
            image_LR.append(img_LR)
    
    image_HR = []
    path_HR = "./dataset/Set7_8/Lepton/"
    for filename in sorted(os.listdir(path_HR)):
        img_HR = cv2.imread(os.path.join(path_HR,filename),cv2.IMREAD_UNCHANGED)
        # img_HR = img_HR.astype(int)
        # img_HR = img_HR//2
        # img_HR = img_HR - 63
        # img_HR = np.where(img_HR > 63, 63, img_HR)
        img_HR = img_HR[:, :, np.newaxis]
        if img_HR is not None:
            img_HR = img_HR.transpose(2,0,1)
            image_HR.append(img_HR)
    

    #######test load######
    image_LR_test = []

    path_LR_test = "./dataset/Set7_8/Grideye_test/"
    for filename in sorted(os.listdir(path_LR_test)):
        img_LR_test = cv2.imread(os.path.join(path_LR_test,filename),cv2.IMREAD_UNCHANGED)
        img_LR_test = img_LR_test.astype(int)
        img_LR_test = img_LR_test - np.min(img_LR_test)
        img_LR_test = img_LR_test * (63 / np.max(img_LR_test))
        img_LR_test = np.round(img_LR_test)
        img_LR_test = np.where(img_LR_test > 63, 63, img_LR_test)
        img_LR_test = img_LR_test[:, :, np.newaxis]
        if img_LR_test is not None:
            img_LR_test = img_LR_test.transpose(2,0,1)
            image_LR_test.append(img_LR_test)
   

    image_HR_test = []

    path_HR_test = "./dataset/Set7_8/Lepton_test/"
    for filename in sorted(os.listdir(path_HR_test)):
        img_HR_test = cv2.imread(os.path.join(path_HR_test,filename),cv2.IMREAD_UNCHANGED)
        # img_HR_test = img_HR_test.astype(int)
        # img_HR_test = img_HR_test//2
        # img_HR_test = img_HR_test - 63
        # img_HR_test = np.where(img_HR_test > 63, 63, img_HR_test)
        img_HR_test = img_HR_test[:, :, np.newaxis]
        
        if img_HR_test is not None:
            img_HR_test = img_HR_test.transpose(2,0,1)
            image_HR_test.append(img_HR_test)

   

    
    train = SRCNN_Dataset(image_LR, image_HR)
    t = np.array(image_LR)
    #print(t.shape)
    train_loader = DataLoader(
            train, batch_size=10, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True
        )
    test = SRCNN_Dataset(image_LR_test, image_HR_test)
    test_loader = DataLoader(
            test, batch_size=1, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True
        )

    device = torch.device('cuda:1')

    model = SRCNN().to(device)
    save_path = './models/srcnn_Set7_8_' + str(ch) + '.ckpt'
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    best_accuracy = 100000
    print(model)
    for epoch in range(num_epoch):
        running_loss = 0.0
        for  i, data in enumerate(train_loader):
            
            inputs, labels = data
            #print(inputs.shape)
            #print(labels.shape)
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(inputs.shape)
            #print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.requires_grad_(True)
            loss.backward()
            #print(model.layer3[0].weight.grad)
            optimizer.step()
            running_loss += loss.item()
            #if epoch%10 == 0:
            #    torch.save(model.state_dict(), './models/srcnn_Set7_8/'+str(epoch)+'.ckpt')

        train_acc = running_loss / (i+1)
        print(f"[ Train | avg_loss = {train_acc:.5f}")
        model.eval()
        running_loss = 0
        for  i, data in enumerate(test_loader):
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
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            print("Save Model")
        print(f"[ Val | avg_loss = {val_acc:.5f}")
    #print(model.layer3[0].weight.data)
    #torch.save(model.state_dict(), save_path)
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
            ssim_in = copy.deepcopy(labels)
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
        print(ssim_const/57)
        print(psnr)
           
        print(f"[ Test | avg_loss = {test_acc:.5f}")
    


if __name__ == '__main__':
    main()
