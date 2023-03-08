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

if __name__ == "__main__":
    folderAPOE0 = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE0_7var_earlystop5e4_block_random/1'
    folderAPOE12 = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE12_7var_earlystop5e4_block_random/1'
    # folderAPOE012 = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE012_7var/1'

    # cvs_file = '/N/u/liyuny/Quartz/cnode_ffr_main/data/av45mav1451PETAPOE_finalized.csv'
    # var1 = 'WHOLECEREBELLUM_SUVR'
    # var2 = 'META_TEMPORAL_SUVR'
    # t1 = 'YearsOnsetAv45'
    # t2 = 'YearsOnsetAv1451'
    func0 = torch.load(folderAPOE0+'/trained_model_0.pth')
    func12 = torch.load(folderAPOE12 + '/trained_model_0.pth')

    timeAPOE0 = np.load(folderAPOE0 + '/timeX_.npy')
    timeAPOE12 = np.load(folderAPOE12 + '/timeX_.npy')
    # timeAPOE012 = np.load(folderAPOE012 + '/timeX_.npy')
    # 
    predXAPOE0 = np.load(folderAPOE0 + '/predX_9999_299000.npy', allow_pickle=True)
    predXAPOE12 = np.load(folderAPOE12 + '/predX_9999_299000.npy', allow_pickle=True)
    # predXAPOE012 = np.load(folderAPOE012 + '/predX_5000.npy', allow_pickle=True)

    minTAPOE0 = min(timeAPOE0)
    maxTAPOE0 = max(timeAPOE0)
    minTAPOE12 = min(timeAPOE12)
    maxTAPOE12 = max(timeAPOE12)
    # minTAPOE012 = min(timeAPOE012)
    # maxTAPOE012 = max(timeAPOE012)

    XfullAPOE0 = torch.tensor(np.linspace(minTAPOE0, maxTAPOE0, 100))
    XfullAPOE12 = torch.tensor(np.linspace(minTAPOE12, maxTAPOE12, 100))
    # XfullAPOE012 = torch.tensor(np.linspace(minTAPOE012, maxTAPOE012, 100))

    # xend = min(maxTAPOE0,maxTAPOE12)
    time = np.linspace(0,maxTAPOE0+0,100)
    predArb0 = odeint(func0, torch.tensor(predXAPOE0[0,:]), torch.tensor(time))
    predArb12 = odeint(func12, torch.tensor(predXAPOE12[0,:]), torch.tensor(time))

    fig, ax = plt.subplots()
    ax.axvspan(maxTAPOE0,max(time) , alpha=0.2, color='gray')
    plt.xlim(0,max(time))
    plt.ylim(-1.5,3.0)
    plt.plot(time, predArb0[:, 0].detach().numpy(), label="APOE4 non-carrier: Amyloid PET", c='#FF8000',linestyle='solid')
    plt.plot(time, predArb0[:, 1].detach().numpy(), label="APOE4 non-carrier: Total Tau PET", c='#27408B',linestyle='solid')
    plt.plot(time, predArb0[:, 2].detach().numpy(), label="APOE4 non-carrier: ADAS13", c='red',linestyle='solid')
    plt.plot(time, predArb0[:, 3].detach().numpy(), label="APOE4 non-carrier: CSF TAU", c='black',linestyle='solid')
    plt.plot(time, predArb0[:, 4].detach().numpy(), label="APOE4 non-carrier: CSF PTAU", c='green',linestyle='solid')
    plt.plot(time, predArb0[:, 5].detach().numpy(), label="APOE4 non-carrier: FDG", c='blue',linestyle='solid')
    plt.plot(time, predArb0[:, 6].detach().numpy(), label="APOE4 non-carrier: AV45", c='pink',linestyle='solid')

    plt.plot(XfullAPOE0, predXAPOE0[:, 0],  c='#FF8000')
    plt.plot(XfullAPOE0, predXAPOE0[:, 1], c='#27408B')
    plt.plot(XfullAPOE0, predXAPOE0[:, 2],  c='red')
    plt.plot(XfullAPOE0, predXAPOE0[:, 3],  c='black')
    plt.plot(XfullAPOE0, predXAPOE0[:, 4], c='green')
    plt.plot(XfullAPOE0, predXAPOE0[:, 5],  c='blue')
    plt.plot(XfullAPOE0, predXAPOE0[:, 6],  c='pink')
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    plt.legend(loc=0, prop={'size': 6})
    # plt.show()
    plt.savefig(folderAPOE0 + "/AdniPlot0_")
    plt.close()

    #
    # plt.figure()
    # xend = min(maxTAPOE0,maxTAPOE12)
    # plt.xlim(0,xend)
    # plt.ylim(-1,1.5)
    # # full prediction curve
    # plt.title('Fitted Values using NODE')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 0], label="APOE4 non-carrier: Amyloid PET", c='#FF8000')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 1], label="APOE4 non-carrier: Total Tau PET", c='#27408B')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 2], label="APOE4 non-carrier: ADAS13", c='red')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 3], label="APOE4 non-carrier: CSF TAU", c='black')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 4], label="APOE4 non-carrier: CSF PTAU", c='green')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 5], label="APOE4 non-carrier: FDG", c='blue')
    # # plt.plot(XfullAPOE0, predXAPOE0[:, 6], label="APOE4 non-carrier: AV45", c='pink')
    # #
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 0], label="APOE4 carrier: Amyloid", c='#FF8000',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 1], label="APOE4 carrier: Total Tau", c='#27408B',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 2], label="APOE4 carrier: ADAS13", c='red',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 3], label="APOE4 carrier: CSF TAU", c='black',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 4], label="APOE4 carrier: CSF PTAU", c='green',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 5], label="APOE4 carrier: FDG", c='blue',linestyle='dashed')
    # # plt.plot(XfullAPOE12, predXAPOE12[:, 6], label="APOE4 carrier: AV45", c='pink',linestyle='dashed')
    #
    # plt.plot(XfullAPOE012, predXAPOE012[:, 0], label="APOE4 carrier & noncarrier: Amyloid", c='#FF8000',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 1], label="APOE4 carrier & noncarrier: Total Tau", c='#27408B',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 2], label="APOE4 carrier & noncarrier: ADAS13", c='red',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 3], label="APOE4 carrier & noncarrier: CSF TAU", c='black',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 4], label="APOE4 carrier & noncarrier: CSF PTAU", c='green',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 5], label="APOE4 carrier & noncarrier: FDG", c='blue',linestyle='dashdot')
    # plt.plot(XfullAPOE012, predXAPOE012[:, 6], label="APOE4 carrier & noncarrier: AV45", c='pink',linestyle='dashdot')
    #
    # plt.xlabel("Years From Onset")
    # plt.ylabel("Linear Transformed Values")
    # plt.legend()
    # plt.savefig(folderAPOE0 + "/AdniPlot012")