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
    folderAPOE0 = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_intial1_appfunc_3Relu_decay4_APOE0_finalized_v2'
    folderAPOE12 = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_intial1_appfunc_3Relu_decay4_APOE12_finalized'

    # cvs_file = '/N/u/liyuny/Quartz/cnode_ffr_main/data/av45mav1451PETAPOE_finalized.csv'
    # var1 = 'WHOLECEREBELLUM_SUVR'
    # var2 = 'META_TEMPORAL_SUVR'
    # t1 = 'YearsOnsetAv45'
    # t2 = 'YearsOnsetAv1451'

    timeAPOE0 = np.load(folderAPOE0 + '/timeX_.npy')
    timeAPOE12 = np.load(folderAPOE12 + '/timeX_.npy')

    predXAPOE0 = np.load(folderAPOE0 + '/predX_5000.npy', allow_pickle=True)
    predXAPOE12 = np.load(folderAPOE12 + '/predX_5000.npy', allow_pickle=True)

    minTAPOE0 = min(timeAPOE0)
    maxTAPOE0 = max(timeAPOE0)
    minTAPOE12 = min(timeAPOE12)
    maxTAPOE12 = max(timeAPOE12)

    XfullAPOE0 = torch.tensor(np.linspace(minTAPOE0, maxTAPOE0, 100))
    XfullAPOE12 = torch.tensor(np.linspace(minTAPOE12, maxTAPOE12, 100))

    plt.figure()
    xend = min(maxTAPOE0,maxTAPOE12)
    plt.xlim(0,xend)
    plt.ylim(-3,8)
    # full prediction curve
    plt.title('Fitted Values using NODE')
    plt.plot(XfullAPOE0, predXAPOE0[:, 0], label="APOE4 non-carrier: Amyloid", c='#FF8000')
    plt.plot(XfullAPOE0, predXAPOE0[:, 1], label="APOE4 non-carrier: Total Tau", c='#27408B')

    plt.plot(XfullAPOE12, predXAPOE12[:, 0], label="APOE4 carrier: Amyloid", c='#FF8000',linestyle='dashed')
    plt.plot(XfullAPOE12, predXAPOE12[:, 1], label="APOE4 carrier: Total Tau", c='#27408B',linestyle='dashed')

    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    plt.legend()
    plt.savefig(folderAPOE0 + "/AdniPlotComb")