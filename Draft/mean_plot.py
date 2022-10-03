import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import time

import torch
from models.NODEmodels import *
from models.utils import ObservedData as od
from models.training import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, default='VMncvt')
parser.add_argument('--data',type=str)
parser.add_argument('--outdir',type=str)
parser.add_argument('--epoch',type=int)

args = parser.parse_args()

class adniData(Dataset):
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file):
        self.adni = pd.read_csv(csv_file)
        self.SparseData = []
        valid1 = self.adni[(np.isnan(self.adni[args.var]) == 0)
                           & (np.isnan(self.adni.tage) == 0) & (self.adni.tage != 0)]
        # & ((self.adni['DX.bl'] =='CN')|(self.adni['DX.bl'] =='SMC'))]
        filter = valid1['APOE4'].isin([1, 2])
        valid1 = valid1[filter]

        valid2 = self.adni[(np.isnan(self.adni[args.var]) == 0)
                           & (np.isnan(self.adni.tage) == 0) & (self.adni.tage != 0)]
        # & ((self.adni['DX.bl'] == 'CN') | (self.adni['DX.bl'] == 'SMC'))]
        filter = valid2['APOE4'].isin([0])
        valid2 = valid2[filter]

        # t1 = torch.tensor(av45_valid.ageScaled.to_numpy()/10)
        t1 = torch.tensor(valid1['tage'].to_numpy())
        x1 = torch.tensor(np.expand_dims(np.expand_dims(valid1[args.var].to_numpy(), 1), 1))
        # t2 = torch.tensor(tau_valid.ageScaled.to_numpy()/10)
        t2 = torch.tensor(valid2['tage'].to_numpy())
        x2 = torch.tensor(np.expand_dims(np.expand_dims(valid2[args.var].to_numpy(), 1), 1))

        self.SparseData.append((t1, t2, x1, x2, x1, x2))
        # for i in range(0,min(len(av45_valid),len(tau_valid)),5):
        #     t1 = torch.tensor(av45_valid.ageScaled[i:i+5].to_numpy())
        #     x1 = torch.tensor(np.expand_dims(np.expand_dims(av45_valid.AV45[i:i+5].to_numpy(), 1), 1))
        #     t2 = torch.tensor(tau_valid.ageScaled[i:i+5].to_numpy())
        #     x2 = torch.tensor(np.expand_dims(np.expand_dims(tau_valid.TAU[i:i+5].to_numpy(), 1), 1))
        #     self.SparseData.append((t1, t2, x1, x2, x1, x2))

    def __len__(self):
        return len(self.SparseData)

    def __getitem__(self, index):
        return self.SparseData[index]

data_loader = adniData('/data/draftData/adData.csv')
for i, data in enumerate(data_loader):
    t1, t2, x1_obs, x2_obs, x1_true, x2_true = data
    #
    # t1 = t1.to(self.device)
    # t2 = t2.to(self.device)
    #
    # x1_obs = x1_obs.to(self.device)
    # x2_obs = x2_obs.to(self.device)
    #
    # x1_true = x1_true.to(self.device)
    # x2_true = x2_true.to(self.device)

    sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
    sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)
    print(sort_t1)
    print(sort_x1_obs)

    plt.scatter(sort_t1,sort_x1_obs)
plt.show()

