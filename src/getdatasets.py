from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import scipy.io
import torchvision.transforms as tvtf
import numpy as np


class Samson(data.Dataset):

    img_folder = 'Data_Matlab'
    gt_folder = 'GroundTruth'
    training_file = 'synthetic_data3dpad5.mat'
    labels_file = 'end4.mat'

    def __init__(self, root, transform=None, target_transform=None,**hyperparams):
        """Init Samson dataset."""
        super(Samson, self).__init__()

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.patch_size = 5

        PATH = os.path.join(self.root, self.img_folder, self.training_file)
        PATH_L = os.path.join(self.root, self.gt_folder, self.labels_file)

        training_data = scipy.io.loadmat(PATH)
        labels = scipy.io.loadmat(PATH_L)

        
        self.train_data = training_data['V']
        # self.train_data = self.train_data[2:258,2:258,:]#patch 1
        self.label_data = labels['p']
        self.label_data =np.reshape(self.label_data,(256,256)).transpose(1,0)

        # mask = np.ones((256,256))#1
        mask = np.ones((260,260))#5
        x_pos, y_pos= np.nonzero(mask)
        p = self.patch_size // 2

        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x >= p and x < self.train_data.shape[0] - p and y >= p and y < self.train_data.shape[1] - p ])
        np.random.shuffle(self.indices)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2= x1 + self.patch_size, y1 + self.patch_size,

        data = self.train_data[x1:x2, y1:y2]
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data = torch.from_numpy(data)
        if self.patch_size == 1:
            data = data[:, 0, 0]
        if self.patch_size > 1:
            data = data.unsqueeze(0)
        if x<256 and  y<255:
            p_label=self.label_data[x,y]
        else:   
            p_label=0
        return data,p_label
   

    def _check_exists(self):
        """Check if the path specified exists."""
        return os.path.exists(os.path.join(self.root, self.img_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.gt_folder, self.labels_file)
        )
        
        
def get_dataloader(BATCH_SIZE: int, DIR):
    """Create a DataLoader for input data stream."""
    trans = tvtf.Compose([tvtf.ToTensor()])

    source_domain = Samson(root=DIR, transform=trans, target_transform=trans)
    source_dataloader = torch.utils.data.DataLoader(source_domain, BATCH_SIZE)
    
    return source_dataloader, source_domain