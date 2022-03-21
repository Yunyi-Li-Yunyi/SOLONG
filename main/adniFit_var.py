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
parser.add_argument('--var', type=str)
parser.add_argument('--data',type=str)
parser.add_argument('--outdir',type=str)
parser.add_argument('--epoch',type=int)

args = parser.parse_args()

class adniData(Dataset):
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file):
        self.adni = pd.read_csv(csv_file)
        self.SparseData=[]
        valid1 = self.adni[(np.isnan(self.adni[args.var]) == 0)
                           & (np.isnan(self.adni.tage) == 0) & (self.adni.tage!= 0)]
                               # & ((self.adni['DX.bl'] =='CN')|(self.adni['DX.bl'] =='SMC'))]
        filter=valid1['APOE4'].isin([1,2])
        valid1=valid1[filter]

        valid2 = self.adni[(np.isnan(self.adni[args.var]) == 0)
                           & (np.isnan(self.adni.tage) == 0) & (self.adni.tage!= 0)]
                              # & ((self.adni['DX.bl'] == 'CN') | (self.adni['DX.bl'] == 'SMC'))]
        filter=valid2['APOE4'].isin([0])
        valid2=valid2[filter]

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



x_dim = 1
y_dim = 2
h_dim = 32

func = None

func = VanillaODEFunc(x_dim, h_dim, y_dim)

optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
dataset = adniData(args.data)

data_loader = DataLoader(dataset)
sim=False
ts_equal=False
folder=args.outdir
if not osp.exists(folder):
    os.makedirs(folder)
seed=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(sim, device, ts_equal, func, optimizer, folder, seed,True)
print('Training...')
start_time = time.time()
trainer.train(data_loader, args.epoch, seed)
torch.save(func, osp.join(folder, ('trained_model_' + str(seed) + '.pth')))
end_time = time.time()

# func = torch.load(osp.join(folder, 'trained_model.pth')).to(device)
pred = torch.tensor(np.load(folder + '/predX_500.npy', allow_pickle=True))
time = torch.tensor(np.load(folder + '/timeX_.npy', allow_pickle=True))
dt_dense = torch.roll(time, -1, 0) - time  # delta time
dv_pred = torch.transpose(torch.roll(pred, -1, 0) - pred, 0, 1)  # delta values
dr_pred = dv_pred / dt_dense

# print(time)
plt.subplot(2, 1, 1)
plt.plot(time[:len(time) - 1].detach().numpy(),
         dr_pred[0][:len(time) - 1].detach().numpy(), linewidth=1, alpha=0.7)
# plt.xlim([0, 15])
plt.title('Accumulative Rate for APOE4 Carrier Pred '+str(args.var))
plt.subplot(2, 1, 2)
plt.plot(time[:len(time) - 1].detach().numpy(),
         dr_pred[1][:len(time) - 1].detach().numpy(), linewidth=1, alpha=0.7)
# plt.xlim([0, 15])
plt.title('Accumulative Rate for APOE4 Non-carrier Pred '+str(args.var))
# plt.show()
plt.savefig(folder + "/AccRate")


