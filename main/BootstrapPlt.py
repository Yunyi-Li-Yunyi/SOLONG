import argparse
import os
import os.path as osp
import json
# from tqdm import tqdm

from models.NODEmodels import *
from data.dataset import DeterministicLotkaVolterraData
from models.training import Trainer

# from models.utils import PredData
from torch.utils.data import DataLoader
from models.utils import ObservedData as od
from models.utils import prediction
#
from joblib import Parallel, delayed
import multiprocessing
import time
from evaluation.eval import final_eval_default
import matplotlib.pyplot as plt
from models.utils import obsPercent as op


def BootstrapPlt():
    folder = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_v3_intial1_appfunc_3Relu_decay4'
    # plot the fitted curve
    time = np.load(folder + '/timeX_.npy')
    predX = np.load(folder + '/predX_5000.npy', allow_pickle=True)
    minT=min(time)
    maxT=max(time)
    Xfull = torch.tensor(np.linspace(minT, maxT, 100))
    plt.figure()
    plt.plot(Xfull, predX[:, 0], c='#6E8B3D', linewidth=1.5, alpha=1.0,label="Fitted Total Tau (av1451)")
    plt.plot(Xfull, predX[:, 1], c='#8B8378', linewidth=1.5, alpha=1.0,label="Fitted Amyloid (av45)")
    plt.ylim(-2.5,7)
    # plot bootstrap predicted X values
    Nboots = 100
    Iterfolder = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_v3_intial1_appfunc_3Relu_decay4_bootstrap/'
    for i in range(Nboots):
        time = np.load(Iterfolder +str(i+1)+ '/timeX_.npy')
        predX = np.load(Iterfolder + str(i+1)+'/predX_5000.npy', allow_pickle=True)
        minT = min(time)
        maxT = max(time)
        Xfull = torch.tensor(np.linspace(minT, maxT, 100))
        plt.plot(Xfull,predX[:, 0],c='#6E8B3D',linewidth=0.3,alpha=0.4)
        plt.plot(Xfull,predX[:, 1],c='#8B8378',linewidth=0.3,alpha=0.4)
    plt.title("100 Bootstrap Fitted Curves")
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    plt.legend()
    # plt.show()
    plt.savefig(Iterfolder + "/BootstrapPlt" )

if __name__ == "__main__":
    BootstrapPlt()
