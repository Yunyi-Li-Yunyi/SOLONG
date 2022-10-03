import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import json
import torch
from models.NODEmodels import *
from models.utils import ObservedData as od
from models.training import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--age',type=str)
parser.add_argument('--var1', type=str)
parser.add_argument('--var2', type=str)
parser.add_argument('--t1', type=str)
parser.add_argument('--t2', type=str)
parser.add_argument('--data',type=str)
parser.add_argument('--outdir',type=str)
parser.add_argument('--epoch',type=int,default=2001)
parser.add_argument('--x_dim',type=int,default=1)
parser.add_argument('--y_dim',type=int,default=2)
parser.add_argument('--h_dim',type=int,default=125)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--decay',type=float,default=0.0)
args = parser.parse_args()

class adniData(Dataset):
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file):
        self.adni = pd.read_csv(csv_file)
        self.SparseData=[]
        valid1 = self.adni[(np.isnan(self.adni[args.var1]) == 0)
                           & (np.isnan(self.adni[args.t1]) == 0)
                           & (self.adni[args.t1]>=0)]
                           # & (self.adni[args.age]!= 0)]
                               # & ((self.adni['DX.bl'] =='CN')|(self.adni['DX.bl'] =='SMC'))]
        valid1['normalized'] = preprocessing.scale(valid1[args.var1])
        filter1=valid1['APOE4'].isin([0])
        valid1=valid1[filter1]
        print(valid1["APOE4"])

        valid2 = self.adni[(np.isnan(self.adni[args.var2]) == 0)
                           & (np.isnan(self.adni[args.t2]) == 0)
                           & (self.adni[args.t2]>=0)]
                           # & (self.adni[args.age]!= 0)]
                              # & ((self.adni['DX.bl'] == 'CN') | (self.adni['DX.bl'] == 'SMC'))]
        valid2['normalized'] = preprocessing.scale(valid2[args.var2])
        filter2=valid2['APOE4'].isin([0])
        valid2=valid2[filter2]
        print(valid2["APOE4"])


        # t1 = torch.tensor(av45_valid.ageScaled.to_numpy()/10)
        t1 = torch.tensor(valid1[args.t1].to_numpy())
        # x1 = torch.tensor(np.expand_dims(np.expand_dims(valid1[args.var1].to_numpy(), 1), 1))
        # x1 = torch.tensor(np.expand_dims(np.expand_dims(preprocessing.scale(valid1[args.var1]), 1), 1))
        x1 = torch.tensor(np.expand_dims(np.expand_dims((valid1["normalized"]), 1), 1))

        # t2 = torch.tensor(tau_valid.ageScaled.to_numpy()/10)
        t2 = torch.tensor(valid2[args.t2].to_numpy())
        # x2 = torch.tensor(np.expand_dims(np.expand_dims(valid2[args.var2].to_numpy(), 1), 1))
        # x2 = torch.tensor(np.expand_dims(np.expand_dims(preprocessing.scale(valid2[args.var2]), 1), 1))
        x2 = torch.tensor(np.expand_dims(np.expand_dims((valid2["normalized"]), 1), 1))

        self.SparseData.append((t1, t2, x1, x2, x1, x2))

    def __len__(self):
        return len(self.SparseData)

    def __getitem__(self, index):
        return self.SparseData[index]

if __name__ == "__main__":
    folder = args.outdir

    if not osp.exists(folder):
        os.makedirs(folder)

    dataset = adniData(args.data)

    data_loader = DataLoader(dataset, shuffle=True)
    for i, data in enumerate(data_loader):
        t1, t2, x1_obs, x2_obs, x1_true, x2_true = data

        # sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
        # sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)

    time = np.load(args.outdir + '/timeX_.npy')
    predX = np.load(args.outdir + '/predX_5000.npy', allow_pickle=True)
    minT = min(time)
    maxT = max(time)
    Xfull = torch.tensor(np.linspace(minT, maxT, 100))
    plt.figure()
    plt.xlim(0,26)
    plt.ylim(-3,8)
    # full prediction curve
    plt.plot(Xfull, predX[:, 0], label="Amyloid fitted", c='#FF8000')
    plt.plot(Xfull, predX[:, 1], label="Total Tau fitted", c='#27408B')
    # observed data with noise
    plt.scatter(t1.cpu().numpy(), x1_obs.cpu().numpy()[:], marker='x', c='#8B5A2B', alpha=0.6,
                label='Amyloid aggregated')
    plt.scatter(t2.cpu().numpy(), x2_obs.cpu().numpy()[:], marker='x', c='#9FB6CD', alpha=0.6,
                label='Total Tau aggregated')
    # plt.scatter(t1.cpu().numpy()[0, :], x1_obs.cpu().numpy()[0, :], color="none", s=20,
    #             edgecolor='r')
    # plt.scatter(t2.cpu().numpy()[0, :], x2_obs.cpu().numpy()[0, :], color="none", s=13,
    #             edgecolor='g')
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    plt.legend()
    plt.savefig(folder + "/AdniPlot_v2" )
    # plt.plot(Xfull, predX[:, 0], c='#6E8B3D', linewidth=0.3, alpha=0.4)
    # plt.plot(Xfull, predX[:, 1], c='#8B8378', linewidth=0.3, alpha=0.4)
    #

