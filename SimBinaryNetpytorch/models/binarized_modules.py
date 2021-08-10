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
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        
        #print (torch.min(out), torch.max(out))
        #out = torch.round(out)
        #print (torch.min(out), torch.max(out))
        #print (torch.min(input), torch.max(input))
        #out = torch.round(out / 64 * 36 / 64)
        #print (self.weight.size()[1])
        # if self.weight.size()[1] >= 16 and self.weight.size()[1] <= 24:
        #     out = torch.round(out / 64 * 36 / 64)
        #out = torch.round(out * 7 / 64)
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
device = 'cuda:0'
sim_model = sim_model.to(device)
sim_model.eval()


class CimSimConv2d(nn.Conv2d):
  def __init__(self, *kargs, **kwargs):
    super(CimSimConv2d, self).__init__(*kargs, **kwargs)
  
    self.device = device
    
  def forward(self, input):
    if not hasattr(self.weight,'org'):
      self.weight.org=self.weight.data.clone()
    self.weight.data=Binarize(self.weight.org)

    #scale = max(torch.max(input), -torch.min(input)) / 63
    #if scale != 0:
    #  input = torch.round(input / scale) 
    #''' random error
    #out = nn.functional.conv2d(input, self.weight, None, self.stride,
    #                           self.padding, self.dilation, self.groups)
    #out = torch.round(out / 64 * 36 / 64)
    #randrange = (self.weight.size()[1] // 4)
    #for _ in range(randrange):
    #  out += torch.randint(-1, 1, out.size(), device=device)
    #out[out>63] = 63
    #out[out<-63] -63
    #'''
    input = torch.round(input) 
    out2 = self.simconv(input, self.weight)
    out2 = out2*4
    out2[out2 >  63] =  63
    out2[out2 < -63] = -63
    #print (self.weight.data.size())
    #print (torch.max(out2), torch.min(out2))
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
    out_width = input_a.size()[2] - 2 * (weight.size()[2] // 2)
    out_height = input_a.size()[3] - 2 * (weight.size()[3] // 2)
    simout = torch.zeros(batch_size, out_channel, out_width, out_height, dtype = input_a.dtype).to(device)
    first = True
    #''' Mapping Table
    kernel_group = 4
    Digital_input_split = torch.split(input_a, kernel_group, dim=1)
    binary_weight_split = torch.split(weight, kernel_group, dim=1)
    for i in range(len(Digital_input_split)):
      temp_output = nn.functional.conv2d(Digital_input_split[i], binary_weight_split[i], None, self.stride, self.padding, self.dilation, self.groups)
      #temp_output = torch.round(temp_output / 64 * 36 / 64)
      temp_output = torch.round(temp_output / 64)
      temp_output = Mapping.apply(temp_output)
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

class Mapping(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
        output = input.clone()

        output[input==-1]  = -4
        output[input==-2]  = -5
        output[input==-3]  = -6
        output[input==-4]  = -7
        output[input==-5]  = -9
        output[input==-6]  = -9
        output[input==-7]  = -11
        output[input==-8]  = -11
        output[input==-9]  = -13
        output[input==-10] = -13
        output[input==-11] = -17
        output[input==-12] = -17
        output[input==-13] = -17
        output[input==-14] = -19
        output[input==-15] = -19
        output[input==-16] = -21
        output[input==-17] = -21
        output[input==-18] = -23
        output[input==-19] = -25
        output[input==-20] = -25
        output[input==-21] = -25
        output[input==-22] = -25
        output[input==-23] = -27
        output[input==-24] = -27
        output[input==-25] = -29
        output[input==-26] = -29
        output[input==-27] = -29
        output[input==-28] = -31
        output[input==-29] = -31
        output[input==-30] = -33
        output[input==-31] = -33
        output[input==-32] = -35
        output[input==-33] = -35
        output[input==-34] = -35
        #output[input==-35] = -35

        output[input==0]   = -2
        output[input==1]   = -1
        output[input==2]   = 1
        output[input==3]   = 2
        #output[input==4]   = 4
        output[input==5]   = 4
        #output[input==6]   = 6
        output[input==7]   = 8
        #output[input==8]   = 8
        output[input==9]   = 10
        #output[input==10]  = 10
        output[input==11]  = 12
        #output[input==12]  = 12
        output[input==13]  = 16
        output[input==14]  = 16
        output[input==15]  = 16
        #output[input==16]  = 16
        output[input==17]  = 18
        output[input==18]  = 20
        output[input==19]  = 20
        output[input==20]  = 24
        output[input==21]  = 24
        output[input==22]  = 24
        output[input==23]  = 26
        output[input==24]  = 26
        output[input==25]  = 28
        output[input==26]  = 28
        output[input==27]  = 28
        output[input==28]  = 30
        output[input==29]  = 30
        output[input==30]  = 32
        output[input==31]  = 32
        output[input==32]  = 34
        output[input==33]  = 34
        output[input==34]  = 34
        output[input==35]  = 34
        return output
  def backward(ctx, grad_output):
    return grad_output
