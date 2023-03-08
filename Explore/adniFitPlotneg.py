import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
from torchdiffeq import odeint
from data.dataset import DeterministicLotkaVolterraData
import argparse
from torch.utils.data import DataLoader
from models.utils import prediction
from sklearn import preprocessing
import json

folder = '/N/slate/liyuny/cnode_ffr_main/results/AdniNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE12_finalized_rerun/1'
with open(osp.join(folder, 'args.txt')) as f:
    args = json.load(f)
time= np.load(folder + '/timeX_.npy')
func = torch.load(osp.join(folder, 'trained_model_0'+'.pth'))
csv_file = args['data']
adni = pd.read_csv(csv_file)
adni = adni[adni['APOE4']!=0]
adni['normV1']=preprocessing.scale(adni[args['var1']])
adni['normV2']=preprocessing.scale(adni[args['var2']])
# adni['normV3']=preprocessing.scale(adni[args['var3']])
id = 2239
test = adni[adni['RID']==id]
# 2234 4036 4489
x0 = torch.tensor(np.asarray(test[["normV1","normV2"]].iloc[[2]],dtype='float32'))
time = test['YearsOnsetAv45'].iloc[[2]]
timepos = torch.tensor(np.linspace(time, time+10., 100)).squeeze(1)
timeneg = torch.flip(torch.tensor(np.linspace(time-10.,time,100)),dims=[0]).squeeze(1)
pred_pos = odeint(func, x0, timepos)
pred_neg = odeint(func, x0, timeneg)
plt.figure()
plt.plot(timepos,pred_pos.detach().numpy()[:,0,0],c='#FF8000',label="Amyloid Forward Prediction")
plt.plot(timeneg,pred_neg.detach().numpy()[:,0,0],c='#FF8000',linestyle='dashed',label="Amyloid Backward Prediction")
plt.plot(timepos,pred_pos.detach().numpy()[:,0,1],c='#27408B',label="Tay Forward Prediction")
plt.plot(timeneg,pred_neg.detach().numpy()[:,0,1],c='#27408B',linestyle='dashed',label="Amyloid Backward Prediction")
# plt.plot(timepos,pred_pos.detach().numpy()[:,0,2])
# plt.plot(timeneg,pred_neg.detach().numpy()[:,0,2])
plt.scatter(test['YearsOnsetAv45'],test['normV1'],c='#FF8000',label="Amyloid Observed")
plt.scatter(test['YearsOnsetAv1451'],test['normV2'],c='#27408B',label="Tau Observed")
# plt.scatter(test['YearsOnsetADAS13'],test['normV3'])
plt.axvline(x = time.to_numpy(), color = 'green', label = 'Initial Point for Prediction')
plt.legend()
plt.savefig(folder + "/"+str(id))
plt.close()