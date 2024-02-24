from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASC(nn.Module):
  def __init__(self):
    super(ASC, self).__init__()
  
  def forward(self, input):

    constrained = F.softmax(input,dim=1) 
    return constrained
  
class SAD(nn.Module):
  def __init__(self, num_bands: int=224):
    super(SAD, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    top=torch.mul(input,target).sum(dim=1)
    bottom=(input.square().sum(dim=1).sqrt()+ 1e-6) *(target.square().sum(dim=1).sqrt()+ 1e-6)
    angle=torch.acos(top/bottom)
    return angle
 