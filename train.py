from __future__ import print_function

import os
import torch
import torch.optim as optim
from getdatasets import get_dataloader
from model import Globalmodel
from utils.parse import ArgumentParser
import utils.opts as opts
import matplotlib.pyplot as plt
import numpy
import time
from scipy.io import loadmat
import random


def train(opt):

    num_runs=1
    results_floder='../results'
    method_name = 'Proposed_3d_MLM'
    dataset='Synthetic_SNR30'
    save_folder = results_floder+'/'+method_name+'/'+dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(num_runs):

        seed=random.randint(1,1000)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train_dataloader, test_set = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)
        max_batches = len(train_dataloader)
    
        model = Globalmodel(opt.num_bands, opt.end_members, opt.training,opt.patch_size)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter:" ,total, (total*4/(1024**2)))

        
        optimizer_base = optim.Adam(model.net0.parameters(), lr=opt.learning_rate)
        optimizer_net3 = optim.Adam(model.net3.parameters() , lr=opt.learning_rate)

        optimizer_end = optim.Adam([
            {'params': model.net1.decoder.parameters(), 'lr':5.0e-7}
        ])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        abun_truth = loadmat("../data/Synthetic/GroundTruth/end4.mat")
        abun_truth_data= abun_truth['M']
        abun_truth_data=torch.tensor(abun_truth_data).to(torch.float32)
        abun_truth_data=abun_truth_data.to(device)

        m = loadmat("../data/Synthetic/VCA_endmember.mat")
        endm=m['Ae']
        endm=torch.tensor(endm).to(torch.float32)
        endm=endm.to(device)
        model.net1.decoder.weight.data=endm
    
        model=model.to(device)
        print(model)
        loss_record=numpy.zeros(opt.epochs)

        i=0
        for epoch in range(opt.epochs):
            batch__=0
            for batch in train_dataloader:
                X, p_label = batch
                X=X.to(device)
                p_label = p_label.to(device)
                if opt.patch_size == 1:
                    X=X.unsqueeze(1)
                    X=X.unsqueeze(3)
                    X=X.unsqueeze(4) 
                enc_out, dec_out = model(X.float())
                x1=X[:,:,:,opt.patch_size//2,opt.patch_size//2]
                x1=x1.view(-1,opt.num_bands)
                bn,bn1=x1.shape
                loss = ((dec_out- x1) ** 2).sum() / (bn*bn1)

                optimizer_base.zero_grad()
                optimizer_net3.zero_grad()
                optimizer_end.zero_grad()
                loss.backward()
            
                optimizer_base.step()
                optimizer_end.step()
                optimizer_net3.step()

                model.net1.decoder.weight.data = torch.clamp(model.net1.decoder.weight.data, 0, 1)
                batch__=batch__+1


            if (epoch+1)%20==0:
                print('epoch:',epoch,'batch__:',batch__)
                print("Loss: %.8f" %(loss.item())) 
            loss_record[i]=loss.item()
            i=i+1
        torch.save(model, "../logs/hyperspecae_final_best.pt")
        plt.figure()
        plt.plot(loss_record)
        plt.savefig('../imgs/loss.png')
        plt.show()
        print('Training Finished!')
      
def _get_parser():
    parser = ArgumentParser(description='train.py')
    opts.model_opts(parser)
    opts.train_opts(parser)

    return parser
    
def main():
    t1=time.time()
    parser = _get_parser()
    opt = parser.parse_args()
    train(opt)
    t2=time.time()
    t=t2-t1
    print('time:',t)

if __name__ == "__main__":
    main()