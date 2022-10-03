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

class adniData(Dataset):
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file,group):
        self.adni = pd.read_csv(csv_file)
        self.SparseData=[]
        self.group=group
        if self.group=='CN':
            checklist=['CN','SMC']
        elif self.group=='AD':
            checklist=['AD','EMCI','LMCI']
        elif self.group=='PureAD':
            checklist=['AD']
        elif self.group=='ALL':
            checklist=['AD','EMCI','LMCI','SMC']
        av45_valid = self.adni[(np.isnan(self.adni.AV45) == 0)
                               & (np.isnan(self.adni.ageScaled) == 0)]
        filter=av45_valid['DX.bl'].isin(checklist)

        av45_valid=av45_valid[filter]
        tau_valid = self.adni[(np.isnan(self.adni.TAU) == 0)
                              & (np.isnan(self.adni.ageScaled) == 0)]
                              # & ((self.adni['DX.bl'] == 'CN') | (self.adni['DX.bl'] == 'SMC'))]
        filter=tau_valid['DX.bl'].isin(checklist)
        tau_valid=tau_valid[filter]

        # t1 = torch.tensor(av45_valid.ageScaled.to_numpy()/10)
        t1 = torch.tensor(av45_valid['t.age'].to_numpy())
        x1 = torch.tensor(np.expand_dims(np.expand_dims(av45_valid.AV45.to_numpy(), 1), 1))
        # t2 = torch.tensor(tau_valid.ageScaled.to_numpy()/10)
        t2 = torch.tensor(tau_valid['t.age'].to_numpy())
        x2 = torch.tensor(np.expand_dims(np.expand_dims(tau_valid.TAU.to_numpy(), 1), 1))

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
dataset = adniData('/data/draftData/adni_tau_amyloid.csv', "AD")

data_loader = DataLoader(dataset)
sim=False
ts_equal=False
folder='/N/slate/liyuny/cnode_ffr_main/results/adni/adniFit_AD_meanInitial/'
if not osp.exists(folder):
    os.makedirs(folder)
seed=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(sim, device, ts_equal, func, optimizer, folder, seed,True)
print('Training...')
start_time = time.time()
trainer.train(data_loader, 501, seed)
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


