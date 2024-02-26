from __future__ import print_function
import torch
from utils.parse import ArgumentParser
import utils.extract_opts as opts
import matplotlib.pyplot as plt
from model import Globalmodel
from getdatasetsExtract import get_dataloader
import scipy.io as io
from scipy.io import loadmat
import datetime
from layers import SAD
import os
import numpy

def extract_abundances(opt,now):

    test_set,_ = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)
    model = Globalmodel(opt.num_bands, opt.end_members, opt.training,opt.patch_size)
    max_batches = len(test_set)
    print('max_batches:',max_batches)
    model.eval()
    model=torch.load("../logs/hyperspecae_final_best.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    iterator = iter(test_set)
    abundance=torch.tensor([])
    abundance=abundance.to(device)
    p_learn=torch.tensor([])
    p_learn=p_learn.to(device)
    p_soft=torch.tensor([])
    p_soft=p_soft.to(device)
    y=torch.tensor([])
    y=y.to(device)
    x_img = torch.tensor([])
    x_img = x_img.to(device)
    model.eval()
    y_loss=torch.tensor([])
    y_loss = y_loss.to(device)
    with torch.no_grad():
        for batch__ in range(max_batches):
            img = next(iterator)
            img=img.to(device)
            if opt.patch_size==1:
                img=img.unsqueeze(1)
                img=img.unsqueeze(3)
                img=img.unsqueeze(4)
            e,y1 = model(img.float())
            abundance=torch.cat([abundance,e],dim=0)
            y_SAD = SAD()
            img=img.squeeze()
            if opt.patch_size != 1:
                img=img[:,:,opt.patch_size//2,opt.patch_size//2]
                img=img.view(-1,opt.num_bands)
            y_loss_SAD = y_SAD(img, y1)
            y_loss=torch.cat([y_loss,y_loss_SAD],dim=0)
            y=torch.cat([y,y1],dim=0)
            x_img=torch.cat([x_img,img],dim=0)
            p_learn_temp=model.p_learn
            p_learn=torch.cat([p_learn,p_learn_temp],dim=0)
        
    N_COLS = opt.end_members
    N_ROWS = 1
    abun_learn = abundance.cpu().detach().numpy()
    y=y.cpu().detach().numpy()
    x_img=x_img.cpu().detach().numpy()
    p_learn=p_learn.cpu().detach().numpy()

    path = "../results/Proposed_3d_MLM/Synthetic_SNR30/"+now.strftime("%Y%m%d_%H%M%S") + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"result.txt", "w") as f:
        import sys
        sys.stdout = f
        print("Result")
        sys.stdout = sys.__stdout__
    
    io.savemat(path+'y.mat',{'y':y})
    io.savemat(path+'abun_learn.mat',{'abun_learn':abun_learn})
    io.savemat(path+'p_learn.mat',{'p_learn':p_learn})
    m = loadmat("../data/Synthetic/GroundTruth/end4.mat")

    p_truth = m['p']
    p_learn_2 = p_learn[:,1]
    p_learn_2=p_learn_2.reshape(opt.image_size, opt.image_size).T.reshape(opt.image_size*opt.image_size,-1)
    loss_p = numpy.sqrt(((p_learn_2 - p_truth) ** 2).sum() / (256*256))

    with open(path+"result.txt", "a") as f:
        import sys
        sys.stdout = f
        print('p_loss',loss_p)
        sys.stdout = sys.__stdout__

    abun_truth = m['A']
    N_COLS = opt.end_members
    N_ROWS = 1
    with torch.no_grad():
        for i in range(N_ROWS * N_COLS):
            abundance_cpu = abundance.cpu()
            loss_abun = ((abundance_cpu.squeeze().T[i].reshape(opt.image_size, opt.image_size).T - abun_truth.squeeze()[i].reshape(opt.image_size, opt.image_size)) ** 2).sum() / (opt.image_size*opt.image_size)
            print('abun_loss',i,torch.sqrt(loss_abun))
            with open(path+"result.txt", "a") as f:
                import sys
                sys.stdout = f
                print('abun_loss',i,torch.sqrt(loss_abun))
                sys.stdout = sys.__stdout__
   
    loss_abun_mse = ((abundance_cpu.squeeze().T.reshape(opt.end_members,opt.image_size,opt.image_size).permute(0,2,1).reshape(opt.end_members,-1) - abun_truth.squeeze()) ** 2).sum() / (opt.end_members*opt.image_size*opt.image_size)
    with open(path+"result.txt", "a") as f:
                import sys
                sys.stdout = f
                print('abun_loss',torch.sqrt(loss_abun_mse),abundance_cpu.squeeze().T.size())
                sys.stdout = sys.__stdout__

def extract_endmembers(opt,now):

    _, test_set = get_dataloader(BATCH_SIZE=opt.batch_size, DIR=opt.src_dir)
    model = Globalmodel(opt.num_bands, opt.end_members, opt.training,opt.patch_size)
    path = "../results/Proposed_3d_MLM/Synthetic_SNR30/"+now.strftime("%Y%m%d_%H%M%S") + "/"

    model=torch.load("../logs/hyperspecae_final_best.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    m = loadmat("../data/Synthetic/GroundTruth/end4.mat")
    end_truth=m['M']#224x4
    end_truth=torch.tensor(end_truth).to(torch.float32)
    end_truth=end_truth.to(device)

    N_COLS = opt.end_members
    N_ROWS = 1
    model.eval()

    endm_le = model.net1.decoder.weight
    end_learn=endm_le.cpu().detach().numpy()
    io.savemat(path+'end_learn.mat',{'end_learn':end_learn})
    endmember_SAD = SAD()
    endmember_loss = endmember_SAD(end_truth.T, endm_le.T)
    endmember_loss = torch.sum(endmember_loss).float()/opt.end_members
    with open(path+"result.txt", "a") as f:
        import sys
        sys.stdout = f
        print('endmember_SAD',endmember_loss)
        print('train time',opt.time)
        sys.stdout = sys.__stdout__

def _get_parser():
    parser = ArgumentParser(description='extract.py')
    opts.model_opts(parser)
    opts.extract_opts(parser)
    return parser
    
def main():
    now = datetime.datetime.now()
    parser = _get_parser()
    opt = parser.parse_args()
    extract_abundances(opt,now)
    extract_endmembers(opt,now)


if __name__ == "__main__":
    main()