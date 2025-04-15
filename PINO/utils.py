# coding=utf-8
import os
import h5py
import torch
import torch.nn.functional as F
from torch.nn import Identity
import scipy.io as scio
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import random
import math as mt
import collections
from functools import wraps
from time import time
from sklearn.preprocessing import StandardScaler

def _get_act(act):
    if callable(act):
        return act

    if act == 'tanh':
        func = torch.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    elif act == 'none':
        func = Identity()
    else:
        raise ValueError(f'{act} is not supported')
    return func

def _get_initializer(initializer: str = "Glorot normal"):

    INITIALIZER_DICT = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return INITIALIZER_DICT[initializer]


def timer(func):
    @wraps(func)
    def func_wrapper(*args,**kwargs):
        start_time=time()
        result=func(*args,**kwargs)
        end_time=time()
        print(f"{func.__name__} cost time: {end_time-start_time:.4f} s")
        return result, end_time-start_time
    return func_wrapper

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1, 2), keepdim=True)
        self.std = X.std(dim=(0, 1, 2), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X * (self.std[:, component] - 1e-8) + self.mean[:, component]).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]

def get_dataloader(pde_name,ntrain,ntest,batch_size):
    if pde_name=='Darcy_Flow':
        r=4
        s=64
        filepath='../data_folder/darcyflow.mat'
        data = scio.loadmat(filepath)
        features=data['a']
        label=data['u']
        scaler=StandardScaler()
        features=scaler.fit_transform(features.reshape(-1,1)).reshape(features.shape)
        x_train = features[:ntrain, ::r, ::r]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain, ::r, ::r]
        y_train = torch.from_numpy(y_train)

        x_test = features[-ntest:, ::r, ::r]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:, ::r, ::r]
        y_test = torch.from_numpy(y_test)

        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        x, y = np.meshgrid(x, y)
        grid = np.stack((x,y),axis=-1)
        grid = torch.tensor(grid, dtype=torch.float)

        grid_train = grid.repeat(ntrain, 1, 1, 1)
        grid_test = grid.repeat(ntest, 1, 1, 1)
    
    elif pde_name=='Navier_Stokes_2D':
        filepath='../data_folder/ns_2d'
        features=np.load(filepath+'/in_f.npy')
        label=np.load(filepath+'/out_f.npy')
        grid=np.load(filepath+'/grid.npy')
        grid = torch.tensor(grid, dtype=torch.float)


        x_train = features[:ntrain]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain]
        y_train = torch.from_numpy(y_train)

        x_test = features[-ntest:]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:]
        y_test = torch.from_numpy(y_test)

        grid_train = grid.repeat(ntrain, 1, 1, 1, 1)
        grid_test = grid.repeat(ntest, 1, 1, 1, 1)

    elif pde_name=='Burgers':
        filepath='../data_folder/burgers.mat'
        # burgers parameters
        r=1
        s=128
        data = scio.loadmat(filepath)
        features=data['input']
        label=data['output']
        label=label[:,-1,:]
        # scaler=StandardScaler()
        # features=scaler.fit_transform(features.reshape(-1,1)).reshape(features.shape)
        x_train = features[:ntrain]
        x_train = torch.from_numpy(x_train).float()
        y_train = label[:ntrain]
        y_train = torch.from_numpy(y_train)

        x_test = features[-ntest:]
        x_test = torch.from_numpy(x_test).float()
        y_test = label[-ntest:]
        y_test = torch.from_numpy(y_test)

        grid = np.linspace(0, 1, s)
        grid = torch.tensor(grid, dtype=torch.float)

        grid_train = grid.repeat(ntrain, 1)
        grid_test = grid.repeat(ntest, 1)

    train_loader = DataLoader(TensorDataset(x_train, y_train,grid_train),
                                            batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test,grid_test),
                                            batch_size=batch_size, shuffle=False)

    return train_loader,test_loader


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def to_device(li,device):
    if isinstance(li,collections.Iterable):
        for i in range(len(li)):
            li[i]=li[i].to(device)
        return li
    else: # object
        return li.to(device)

def generate_input(a,grid):
    # a : shape of [bs, x1, ..., xd, init_t, v]
    # grid : shape of [bs, x1, ..., xd, t, d+1]
    T_step=grid.shape[-2]
    a_shape = list(a.shape)[:-2]+[-1] #(bs, x1, ..., xd, -1)
    a=a.reshape(a_shape).unsqueeze(-2)   # (bs, x1, ..., xd, 1, init_t*v )
    temp=[1]*len(a.shape)
    temp[-2]=T_step  # [1,...,t,1]
    input=a.repeat(temp)   # (bs, x1, ..., xd, t, init_t*v )
    input=torch.concat([input,grid], dim=-1)  # (bs, x1, ..., xd, t, init_t*v+d+1)
    return input 