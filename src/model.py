from __future__ import print_function
import torch.nn as nn
from HyperNet import Decoder
from HyperNet import Square_Layer
from HyperNet import SiameseNetwork
from HyperNet import Combination
from HyperNet import EncoderNet


class Globalmodel(nn.Module):
    def __init__(self, num_bands: int=224, end_members: int=4,training=True,patch_size:int=5):    
   
      super(Globalmodel, self).__init__()
      self.net0=EncoderNet(num_bands, end_members,training)
      self.net1=Decoder(num_bands, end_members)
      self.net3=SiameseNetwork(num_bands, end_members,patch_size)
      self.net2=Square_Layer(num_bands,patch_size)
      self.net4=Combination(num_bands)

    def forward(self, img):

      output1=self.net0(img)
      output2=self.net1(output1)
      output3=self.net2(output2,img)
      pred,p_learn=self.net3(img,output2,output3)
      self.p_learn=p_learn
      self.y=output2
      self.yx=output3
      output4=p_learn
      output5=self.net4(output2,output3,output4)

      return output1,output5