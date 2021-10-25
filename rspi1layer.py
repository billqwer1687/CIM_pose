from torch.nn.modules.activation import LeakyReLU
from SimBinaryNetpytorch.models.binarized_modules import  CimSimConv2d
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import copy
from natsort import natsorted
M = [8, 1, 1]
LUT = torch.Tensor([-63, -62, -61, -60,
                    -59, -58, -57, -56, -55, -54, -53, -52, -51, -50,
                    -49, -48, -47, -46, -45, -44, -43, -42, -41, -40,
                    -39, -38, -37, -36, -35, -35, -35, -35, -33, -33,
                    -31, -31, -29, -29, -29, -27, -27, -25, -25, -25,
                    -25, -23, -21, -21, -19, -19, -17, -17, -17, -13,
                    -13, -11, -11, -9, -9, -7, -6, -5, -4, -2,
                    -1, 1, 2, 4, 4, 6, 8, 8, 10,
                    10, 12, 12, 16, 16, 16, 16, 18, 20, 20,
                    24, 24, 24, 26, 26, 28, 28, 28, 30, 30,
                    32, 32, 34, 34, 34, 34, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63]).long()
LUT_OFFSET = 63
class SRCNN(nn.Module):
    def __init__(self, num_channels=1, f1=7, f2=5, f3=5):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Sequential(
            CimSimConv2d(in_channels=num_channels, out_channels=64, kernel_size=f1, padding=f1//2, bias=False),
            #nn.BatchNorm2d(128),
        )
        self.layer2 = nn.Sequential(
            CimSimConv2d(in_channels=64, out_channels=32, kernel_size=f2, padding=f2//2, bias=False),
            #nn.BatchNorm2d(64),
        )
        self.act = nn.Sequential(
            nn.LeakyReLU(0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=f3, padding=f3//2, bias=False),
            #nn.BatchNorm2d(1)
        )
def CimConv2d(input, weight):
    input = torch.round(input) 
    out2 = simconv(input, weight)
    return out2
def simconv(input_a, weight):
    batch_size = input_a.size()[0]
    out_channel = weight.size()[0]
    out_width = input_a.size()[2] #- 2 * (weight.size()[2] // 2)
    out_height = input_a.size()[3] #- 2 * (weight.size()[3] // 2)
    simout = torch.zeros(batch_size, out_channel, out_width, out_height, dtype = input_a.dtype).to(input_a.device)
    first = True
    #''' Mapping Table
    global LUT
    LUT = LUT.to(input_a.device)
    if weight.size()[2] == 7:
      kernel_group = 1
    elif weight.size()[2] == 5:
      kernel_group = 2
    
    Digital_input_split = torch.split(input_a, kernel_group, dim=1)
    binary_weight_split = torch.split(weight, kernel_group, dim=1)
    for i in range(len(Digital_input_split)):
      temp_output = nn.functional.conv2d(Digital_input_split[i], binary_weight_split[i], None, 1, weight.size()[2]//2)
      temp_output = torch.round(temp_output / 64)
      temp_output += LUT_OFFSET
      temp_output[temp_output > 126] = 126
      temp_output[temp_output < 0] = 0
      temp_output = LUT[temp_output.long()]
      if(kernel_group == 5):
        print(temp_output)
      simout += temp_output + 2
    return simout
def main():
    model = SRCNN()
    path = './models/srcnn_Set7_8_32.ckpt'
    model.load_state_dict(torch.load(path))
    l1_weight = model.layer1[0].weight.data
    l2_weight = model.layer2[0].weight.data
    l3_weight = model.layer3[0].weight.data
    l1_weight = torch.tensor(l1_weight).type('torch.DoubleTensor')
    l2_weight = torch.tensor(l2_weight).type('torch.DoubleTensor')
    l3_weight = torch.tensor(l3_weight).type('torch.DoubleTensor')
    path = "./dataset/feature_map_layer1/"
    image = []
    for i in range(57):
        image_LR_test = []
        path_LR_test = path + str(i) + '/'
        for filename in natsorted(os.listdir(path_LR_test)):
            img_LR_test = np.loadtxt(os.path.join(path_LR_test,filename), delimiter = ' ')
            #img_LR_test = img_LR_test - np.min(img_LR_test)
            #img_LR_test = img_LR_test * (63 / np.max(img_LR_test))
            #img_LR_test = np.round(img_LR_test)
            #img_LR_test = np.where(img_LR_test > 63, 63, img_LR_test)
            #img_LR_test = img_LR_test[:, :, np.newaxis]
            if img_LR_test is not None:
                #img_LR_test = img_LR_test.transpose(2,0,1)
                image_LR_test.append(img_LR_test)
        image.append(image_LR_test)

    
    image = torch.tensor(image)
    print(image[0][0])
    m = LeakyReLU(0.5)
    #out = CimConv2d(image_LR_test, l1_weight)
    out = image
    out = 8 * out
    out = torch.floor(out)
    out = m(out)
    out[out > 63] = 63
    out[out < -63] = -63
    out = CimConv2d(out, l2_weight)
    out = torch.floor(out)
    out = m(out)
    out = nn.functional.conv2d(out, l3_weight, None, 1, 2)
    out_img = copy.deepcopy(out)
    out_img -= (out_img.min())
    out_img *= (255/out_img.max())

    img_HR = (torch.round(out_img[30])).to('cpu').numpy().transpose(1,2,0)
    print(img_HR.shape)
    cv2.imwrite('rspi.jpg', img_HR)
    

    
    
if __name__ == '__main__':
    main()