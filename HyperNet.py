
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from layers import ASC

class EncoderNet(nn.Module):
    def __init__(self, num_bands: int=224, end_members: int=4 ,training=True):
        super().__init__()

        self.training = training
        self.asc = ASC()
        self.input_channels = 224
        self.end_members=end_members

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, end_members*4, (7,3,3), stride=1, padding=(0,0,0), dilation=1, groups=1, bias=True),
            nn.LeakyReLU(),
            nn.MaxPool3d((3,1,1))
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(end_members*4, end_members*4, (7,3,3), stride=1, padding=(0,0,0)),
            nn.LeakyReLU(),
            nn.MaxPool3d((3,1,1)),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(end_members*4, end_members*2, (7,1,1), stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool3d((3,1,1))
        )
        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(end_members*2, end_members, (5,1,1), stride=1, padding='valid'),
            # nn.BatchNorm3d(end_members),
            nn.LeakyReLU(),
        )
        self.encoder_stage6 = nn.Sequential(
            self.asc
        )


    def forward(self, inputs):

        output1 = self.encoder_stage1(inputs) 
        self.output1=output1
        output2 = self.encoder_stage2(output1) 
        output3 = self.encoder_stage3(output2)
        output4 = self.encoder_stage4(output3)
        output4=output4.view( -1,self.end_members)
        output6 = self.encoder_stage6(output4)
        return output6


class Decoder(nn.Module):

    def __init__(self,num_bands: int=156, end_members: int=3):
        super(Decoder,self).__init__()
        self.decoder = nn.Linear(end_members, num_bands,bias=False)
        
    def forward(self,encoded):
        decoded=self.decoder(encoded)
        return decoded  

class Square_Layer(nn.Module):

    def __init__(self,num_bands,patch_size):
        super(Square_Layer, self).__init__()
        self.patch_size=patch_size
        self.bands=num_bands

    def forward(self,y,x):    
        x1=x[:,:,:,self.patch_size//2,self.patch_size//2]
        x1=x1.view(-1,self.bands)
        output= t.multiply(y,x1)
        return output 

class SiameseNetwork(nn.Module):
    def __init__(self,num_bands,end_members,patch_size):
        super(SiameseNetwork,self).__init__()
        self.patch_size=patch_size
        self.bands=num_bands
        self.stage1 = nn.Sequential(
            nn.Linear(num_bands*2, num_bands),
            nn.Tanh(),
            nn.Linear(num_bands,112),
            nn.Tanh(),
            nn.Linear(112,56),
            )
        self.stage2 = nn.Sequential(
            nn.Linear(56, 2),
        )      
        self.relu_1 = nn.Sequential(
        nn.Tanh(),
        )

    def forward(self,x,input1,input2):
        
        x=x.squeeze()
        x1=torch.cat((input1,input2),1)
        x1 = self.stage1(x1)
        x = self.relu_1(x1)
        x2 = self.stage2(x)
        pred = x2   
        p_learn=F.softmax(pred, dim=1) 
        return pred,p_learn


class  Combination(nn.Module):

  def __init__(self,num_bands):
    super().__init__()
    self.num_bands=num_bands

  def forward(self,input1,input2,input3):

    temp1=input3[:,1].repeat(self.num_bands,1).permute([1,0])
    top= t.multiply(input1,input3[:,0].repeat(self.num_bands,1).permute([1,0]))
    bottom=1-t.multiply(input1,temp1)
    output=top/bottom
    return output


