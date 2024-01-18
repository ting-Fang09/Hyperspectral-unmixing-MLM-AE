from __future__ import print_function

import torch
import torch.nn as nn

            

class ASC(nn.Module):
  def __init__(self):
    super(ASC, self).__init__()
  
  def forward(self, input):
    constrained = (input.abs().T/torch.sum(input.abs().T, dim=0)).T
    return constrained