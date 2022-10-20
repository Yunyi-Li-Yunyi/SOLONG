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
parser.add_argument('--t_dim',type=int,default=1)
parser.add_argument('--y_dim',type=int,default=2)
parser.add_argument('--h_dim',type=int,default=125)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--apoe4',type=str,choices=['carrier','noncarrier','all'], default='carrier')
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
        if args.apoe4 == 'carrier':
            filter1=valid1['APOE4'].isin([1,2])
        elif args.apoe4 == 'noncarrier':
            filter1=valid1['APOE4'].isin([0])
        else:
            filter1=valid1['APOE4'].isin([0,1,2])

        valid1=valid1[filter1]
        print(valid1["APOE4"])

        valid2 = self.adni[(np.isnan(self.adni[args.var2]) == 0)
                           & (np.isnan(self.adni[args.t2]) == 0)
                           & (self.adni[args.t2]>=0)]
                           # & (self.adni[args.age]!= 0)]
                              # & ((self.adni['DX.bl'] == 'CN') | (self.adni['DX.bl'] == 'SMC'))]
        valid2['normalized'] = preprocessing.scale(valid2[args.var2])
        if args.apoe4 == 'carrier':
            filter2 = valid2['APOE4'].isin([1, 2])
        elif args.apoe4 == 'noncarrier':
            filter2 = valid2['APOE4'].isin([0])
        else:
            filter2 = valid2['APOE4'].isin([0, 1, 2])

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

def figures(folder,csv_file,nepoch,var1,var2):
    rawData = pd.read_csv(csv_file)
    scaledv1=preprocessing.scale(rawData[var1])
    scaledv2=preprocessing.scale(rawData[var2])
    pred = torch.tensor(np.load(folder + '/predX_' + str(nepoch) + '.npy', allow_pickle=True))
    time = torch.tensor(np.load(folder + '/timeX_.npy', allow_pickle=True))
    uniRID=set(rawData['RID'])
    plt.figure()
    for id in uniRID:
        # colors = {'EMCI': 'green', 'LMCI': 'blue', 'AD': 'yellow'}
        x=rawData['YearsOnset'][rawData['RID']==id]
        y=scaledv1[rawData['RID']==id]
        if rawData['DX_bl'][rawData['RID']==id]=="AD":
            color="red"
        elif rawData['DX_bl'][rawData['RID']==id]=="EMCI":
            color="blue"
        elif rawData['DX_bl'][rawData['RID']==id]=="LMCI":
            color="green"
        plt.plot(x, y, alpha=0.7,label='X1',c=color)
    plt.show()
    # ax.plot(rawData["YearsOnset"], scaledv2,  marker='x', c=rawData["DX_bl"].map(colors), alpha=0.7,
    #             label='X2')
    #
    # plt.savefig(folder + "/predPlot")
    import seaborn as sns
    sns.lmplot('YearsOnset', scaledv1, data=rawData, hue='DX_bl', fit_reg=False)
    plt.show()
    # phase plot
    plt.figure()
    plt.quiver(pred[:-1, 0], pred[:-1, 1], pred[1:, 0] - pred[:-1, 0], pred[1:, 1] - pred[:-1, 1], scale_units='xy',
               angles='xy', scale=1)
    plt.xlabel("Scaled META_TEMPORAL_SUVR")
    plt.ylabel("Scaled WHOLECEREBELLUM_SUVR")
    plt.show()
    # plt.savefig(folder + "/phasePlot")
    return 0

if __name__ == "__main__":
    folder = args.outdir

    if not osp.exists(folder):
        os.makedirs(folder)

    with open(osp.join(args.outdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    t_dim = args.t_dim
    y_dim = args.y_dim
    h_dim = args.h_dim
    func = None

    func = ApplicationODEFunc(t_dim, h_dim, y_dim)

    optimizer = torch.optim.Adam(func.parameters(), lr=args.lr, weight_decay=args.decay)
    dataset = adniData(args.data)

    data_loader = DataLoader(dataset, shuffle=True)
    sim = False
    ts_equal = False
    if not osp.exists(folder):
        os.makedirs(folder)
    seed = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(sim, device, ts_equal, func, optimizer, folder, seed, True)
    print('Training...')
    start_time = time.time()
    trainer.train(data_loader, args.epoch, seed)
    torch.save(func, osp.join(folder, ('trained_model_' + str(seed) + '.pth')))
    end_time = time.time()

    # func = torch.load(osp.join(folder, 'trained_model.pth')).to(device)
    # if __name__ == "__main__":
    #     folder="/N/slate/liyuny/cnode_ffr_main/results/adniCogPet/whole_csf_lr3_onset"

    pred = torch.tensor(np.load(folder + '/predX_' + str(args.epoch - 1) + '.npy', allow_pickle=True))
    plt.figure()
    plt.quiver(pred[:-1, 0], pred[:-1, 1], pred[1:, 0] - pred[:-1, 0], pred[1:, 1] - pred[:-1, 1], scale_units='xy',
               angles='xy', scale=1)
    plt.xlabel("X1")
    plt.ylabel("X2")
    # plt.show()
    plt.savefig(folder + "/phasePlot")


# if __name__ == "__main__":
#     folder="/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_2"
#     csv_file="/N/u/liyuny/Quartz/cnode_ffr_main/data/av45mav1451PET_v3.csv"
#     nepoch=5000
#     var1="META_TEMPORAL_SUVR"
#     var2="WHOLECEREBELLUM_SUVR"
#     figures(folder,csv_file,nepoch,var1,var2)
# dt_dense = torch.roll(time, -1, 0) - time  # delta time
# dv_pred = torch.transpose(torch.roll(pred, -1, 0) - pred, 0, 1)  # delta values
# dr_pred = dv_pred / dt_dense

# print(time)
# plt.subplot(2, 1, 1)
# plt.plot(time[:len(time) - 1].detach().numpy(),
#          dr_pred[0][:len(time) - 1].detach().numpy(), linewidth=1, alpha=0.7)
# plt.xlim([0, 15])
# plt.title('Accumulative Rate for APOE4 Carrier Pred '+str(args.var))
# plt.subplot(2, 1, 2)
# plt.plot(time[:len(time) - 1].detach().numpy(),
#          dr_pred[1][:len(time) - 1].detach().numpy(), linewidth=1, alpha=0.7)
# # plt.xlim([0, 15])
# plt.title('Accumulative Rate for APOE4 Non-carrier Pred '+str(args.var))
# # plt.show()
# plt.savefig(folder + "/AccRate")

