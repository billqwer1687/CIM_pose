import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from decimal import Decimal, ROUND_HALF_UP

import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


def Ninarize(tensor, quant_number, quant_mode='det'):
    #return tensor.add(1).mul(quant_number+1).div(2).floor().clamp(0, quant_number).mul(2).add(-quant_number)
    return tensor.add(quant_number).mul(quant_number+1).div(2*quant_number).floor().clamp(0, quant_number).mul(2).add(-quant_number)

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


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

#import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

#        if input.size(1) != 784:
#            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
#        if input.size(1) != 3:
#            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        #input = torch.round(input)
        #input = input*2-1
        #scale = max(torch.max(input), -torch.min(input)) / 63
        #input = torch.round(input*2 / scale) - 63
        #if scale != 0:
        #  input = torch.round(input / scale) 
        #print (torch.max(input))
        #print(input)
        input = torch.round(input) 
        #print(input)
        #print (torch.max(input))
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        
        #print (torch.min(out), torch.max(out))
        #out = torch.round(out)
        #print (torch.min(out), torch.max(out))
        #print (torch.min(input), torch.max(input))
        #out = torch.round(out / 64 * 36 / 64)
        #print (self.weight.size()[1])
        #if self.weight.size()[1] >= 16 and self.weight.size()[1] <= 24:
        if self.weight.size()[1] >= 4 and self.weight.size()[2] * self.weight.size()[3] == 9:
            out = torch.round(out / 64 * 36 / 64)
        elif self.weight.size()[1] == 1:
            out = torch.round(out * 7 / 64)
        else:
            out = torch.round(out / 64)
        out = out * 4
        out[out >  63] =  63
        out[out < -63] = -63
        #out = out - torch.round(torch.mean(out))
        # out = out*4
        #out[out >  63] =  63
        #out[out < -63] = -63
        #else:
        #    out = torch.round(out * 10 / 64)
        #print (torch.min(out), torch.max(out))

        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class IdealCimConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(IdealCimConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
#        if input.size(1) != 3:
#            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        #input = torch.round(input)
        #input = input*2-1
        #scale = max(torch.max(input), -torch.min(input)) / 63
        #input = torch.round(input*2 / scale) - 63
        #if scale != 0:
        #  input = torch.round(input / scale) 
        #print (torch.max(input))
        #print(input)
        input = torch.round(input) 
        #print(input)
        #print (torch.max(input))
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        out = out / 64
        out = out * 4
        out[out >  63] =  63
        out[out < -63] = -63
        return out
        

device = 'cuda:1'
'''
H = [1024, 512]
sim_model = torch.nn.Sequential(
  torch.nn.Linear(36, H[0]),
  torch.nn.Dropout(p=0.5),
  torch.nn.ReLU(),
  torch.nn.Linear(H[0], H[1]),
  torch.nn.Dropout(p=0.5),
  torch.nn.ReLU(),
  torch.nn.Linear(H[-1], 1),
)
sim_model.load_state_dict(torch.load('model_error.ckpt', map_location=torch.device('cuda:0')))            
sim_model = sim_model.to(device)
sim_model.eval()
'''

class CimSimConv2d(nn.Conv2d):
  def __init__(self, *kargs, **kwargs):
    super(CimSimConv2d, self).__init__(*kargs, **kwargs)
  
    self.device = device
    nn.init.uniform_(self.weight.data, a = -1., b = 1.)
    
  def forward(self, input):
    if not hasattr(self.weight,'org'):
      self.weight.org=self.weight.data.clone()
    #print('In:', torch.max(self.weight.org), torch.min(self.weight.org))
    #self.weight.data=Binarize(self.weight.org)
    self.weight.data=Ninarize(self.weight.org, 1)
    #print('out:', torch.max(self.weight.data), torch.min(self.weight.data))

    #scale = max(torch.max(input), -torch.min(input)) / 63
    #if scale != 0:
    #  input = torch.round(input / scale) 
    #''' random error
    #out = nn.functional.conv2d(input, self.weight, None, self.stride,
    #                           self.padding, self.dilation, self.groups)
    #out = torch.round(out / 64)
    #randrange = (self.weight.size()[1] // 4)
    #for _ in range(randrange):
    #  out += torch.randint(-1, 1, out.size(), device=device)
    #out[out>63] = 63
    #out[out<-63] -63
    #'''
    input = torch.round(input) 
    out2 = self.simconv(input, self.weight)
    '''
    if torch.max(out2) < 32:
      out2 = out2 * 2
    if torch.max(out2) < 32:
      out2 = out2 * 2
    if torch.max(out2) < 32:
      out2 = out2 * 2
    '''
    #print ('in, weight, out')
    '''
    print ('round')
    #print (torch.max(input), torch.min(input))
    #print (torch.sum(input), torch.sum(input))
    #print (torch.max(self.weight), torch.min(self.weight))
    #print (torch.sum(self.weight), torch.sum(self.weight))
    print (torch.max(out), torch.min(out))
    print (torch.max(out2), torch.min(out2))
    #'''
    #out2 = out2 * 4
    #out2[out2 >  63] =  63
    #out2[out2 < -63] = -63
    #print (self.weight.data.size())
    #print (torch.max(out-out2), torch.min(out-out2))
    #out = nn.functional.conv2d(input, self.weight, None, self.stride,
    #                             self.padding, self.dilation, self.groups)
    #print(input.size(), self.weight.size(), out.size())

    #if not self.bias is None:
    #  self.bias.org=self.bias.data.clone()
    #  out += self.bias.view(1, -1, 1, 1).expand_as(out)

    return out2
  
  def simconv(self, input_a, weight):
    #print(input_a.size(), weight.size())
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
    elif weight.size()[2] == 1:
      kernel_group = 64
    elif weight.size()[2] == 3:
      kernel_group = 4
    
    Digital_input_split = torch.split(input_a, kernel_group, dim=1)
    binary_weight_split = torch.split(weight, kernel_group, dim=1)
    for i in range(len(Digital_input_split)):
      #print(len(Digital_input_split))
      #print(len(binary_weight_split))
      temp_output = nn.functional.conv2d(Digital_input_split[i], binary_weight_split[i], None, self.stride, self.padding, self.dilation, self.groups)
      temp_output = torch.round(temp_output / 64)
      temp_output += LUT_OFFSET
      temp_output[temp_output > 126] = 126
      temp_output[temp_output < 0] = 0
      temp_output = LUT[temp_output.long()]
      if(kernel_group == 5):
        print(temp_output)
      simout += temp_output + 2
    #print (torch.max(simout), torch.min(simout))
    #'''
    ''' Error model
    for n in range(batch_size):
        for c in range(out_channel):
            w = torch.reshape(weight[c], (-1,)).to(device)
            inputs = []
            for i in range(out_width):
                for j in range(out_height):
                    input = torch.reshape(input_a[n, :, i: i + weight.size()[2], j: j + weight.size()[3]], (-1,))
                    #print (w.size(), input.size())
                    # simout[n][c][i][j] = sum(w*input)
                    # TODO
                    simout[n][c][i][j] = self.cim_conv_tmp(input, w)
    #'''
    #print (len(input))
    #print (simout.size())
    # out = nn.functional.conv2d(input_a, weight)
    return simout
  
  def cim_conv_tmp(self, input, weight):
    assert len(input) == len(weight)

    raw_sum = 0

    if len(weight) == 3:

      for i in range((len(input)-1) // 36 + 1):
        data_x = input[i*36:i*36+36] * weight[i*36:i*36+36]

        
        row = int(Decimal(float(sum(data_x)/64.0)).quantize(0, ROUND_HALF_UP))
        #''' Error model
        if len(data_x) < 36:
          data_x = torch.cat((data_x, torch.zeros(36 - len(data_x), dtype=data_x.dtype)))
        try:
          #ensor_x = torch.Tensor(data_x).to(self.device)
          tensor_x = data_x.to(device)
        except:
          print (data_x, len())
        y_pred = sim_model(tensor_x)
        if int(y_pred[0]) > 10:
          adjust = 10
        elif int(y_pred[0]) < -10:
          adjust = -10
        else:
          adjust = int(y_pred[0])
        #print (tensor_x, y_pred)
        raw_sum += (row + adjust + 2)
        #'''
      #if row in self.mappingTable:
      #  row = self.mappingTable[row]
      #raw_sum += row 
      #raw_sum += row
      else:
        for i in range((len(input)-1) // 49 + 1):
          data_x = input[i*49:i*49+49] * weight[i*49:i*49+49]

          
          row = int(Decimal(float(sum(data_x)/64.0)).quantize(0, ROUND_HALF_UP))
          #''' Error model
          if len(data_x) < 49:
            data_x = torch.cat((data_x, torch.zeros(49 - len(data_x), dtype=data_x.dtype)))
          try:
            #ensor_x = torch.Tensor(data_x).to(self.device)
            tensor_x = data_x.to(device)
          except:
            print (data_x, len())
          y_pred = sim_model(tensor_x)
          if int(y_pred[0]) > 10:
            adjust = 10
          elif int(y_pred[0]) < -10:
            adjust = -10
          else:
            adjust = int(y_pred[0])
          #print (tensor_x, y_pred)
          raw_sum += (row + adjust + 2)
    #print (raw_sum)
    return raw_sum