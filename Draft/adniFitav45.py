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

class adniData(Dataset):
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file,var1,var2):
        self.adni = pd.read_csv(csv_file)
        self.var1=var1
        self.var2=var2
        self.SparseData=[]
        self.adni["Age"]=round(self.adni["Age"],2)
        var1_valid = self.adni[(np.isnan(self.adni[self.var1]) == 0)
                               & (np.isnan(self.adni.Age) == 0) ]

        var2_valid = self.adni[(np.isnan(self.adni[self.var2]) == 0)
                              & (np.isnan(self.adni.Age) == 0)]

        t1 = torch.tensor(var1_valid['Age'].to_numpy())
        x1 = torch.tensor(np.expand_dims(np.expand_dims(var1_valid[self.var1].to_numpy(), 1), 1))
        t2 = torch.tensor(var2_valid['Age'].to_numpy())
        x2 = torch.tensor(np.expand_dims(np.expand_dims(var2_valid[self.var2].to_numpy(), 1), 1))

        self.SparseData.append((t1, t2, x1, x2, x1, x2))
    def __len__(self):
        return len(self.SparseData)

    def __getitem__(self, index):
        return self.SparseData[index]

x_dim = 1
y_dim = 2
h_dim = 100

func = None

func = ApplicationODEFunc(x_dim, h_dim, y_dim)

optimizer = torch.optim.Adam(func.parameters(), lr=1e-4)
dataset = adniData('/data/draftData/av45.csv'
                   ,'CSF_SUVR','WHOLECEREBELLUM_SUVR')

data_loader = DataLoader(dataset)
sim=False
ts_equal=False
folder='/N/slate/liyuny/cnode_ffr_main/results/adni/AV45CSF_Whole_56_initial0_lr4/'
if not osp.exists(folder):
    os.makedirs(folder)
seed=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(sim, device, ts_equal, func, optimizer, folder, seed,True)
print('Training...')
start_time = time.time()
trainer.train(data_loader, 5001, seed)
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
         dr_pred[0][:len(time) - 1].detach().numpy(), linewidth=0.3, alpha=0.7)
# plt.xlim([0, 15])
plt.title('Accumulative rate for dense pred X1')
plt.subplot(2, 1, 2)
plt.plot(time[:len(time) - 1].detach().numpy(),
         dr_pred[1][:len(time) - 1].detach().numpy(), linewidth=0.3, alpha=0.7)
# plt.xlim([0, 15])
plt.title('Accumulative rate for dense pred X2')
# plt.show()
plt.savefig(folder + "/AccRate")


