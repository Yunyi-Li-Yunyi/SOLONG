import pandas as pd
import os
import os.path as osp
import time
import json

from models.NODEmodels import *
from models.training import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
import argparse
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--age',type=str)
parser.add_argument('--var1', type=str)
parser.add_argument('--var2', type=str)
parser.add_argument('--var3', type=str)
parser.add_argument('--var4', type=str)
parser.add_argument('--var5', type=str)
parser.add_argument('--var6', type=str)
parser.add_argument('--var7', type=str)
parser.add_argument('--t1', type=str)
parser.add_argument('--t2', type=str)
parser.add_argument('--t3', type=str)

parser.add_argument('--data',type=str)
parser.add_argument('--exclude_time', type=bool, default=True)
parser.add_argument('--outdir',type=str)
parser.add_argument('--Bepoch',type=int,default=100)
parser.add_argument('--epoch',type=int,default=2000)
parser.add_argument('--t_dim',type=int,default=1)
parser.add_argument('--y_dim',type=int,default=2)
parser.add_argument('--h_dim',type=int,default=125)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--earlyStop', type=float, default=1e-5)
parser.add_argument('--apoe4',type=str,choices=['carrier','noncarrier','all'], default='carrier')
parser.add_argument('--decay',type=float,default=0.0)
parser.add_argument('--bootstrap', type=bool, default=False)
parser.add_argument('--btseedStart',type=int,default=1)
parser.add_argument('--btseedEnd',type=int,default=1)

args = parser.parse_args()

class adniData(Dataset):
    def cleanXT(self,input_x,input_t,boot,btseed):
        if boot == False:
            valid = self.adni[(np.isnan(self.adni[input_x]) == 0)
                               & (np.isnan(self.adni[input_t]) == 0)
                               & (self.adni[input_t]>=0)]
        elif boot == True:
            uRID = pd.DataFrame(set(self.adni['RID']))
            rRID = uRID.sample(frac=1, replace=True, random_state=btseed)
            rSample = rRID.merge(self.adni, how='left', left_on=0, right_on='RID')

            valid = rSample[(np.isnan(rSample[input_x]) == 0)
                             & (np.isnan(rSample[input_x]) == 0)]

        valid['normalized'] = preprocessing.scale(valid[input_x])
        if args.apoe4 == 'carrier':
            filter = valid['APOE4'].isin([1, 2])
        elif args.apoe4 == 'noncarrier':
            filter = valid['APOE4'].isin([0])
        elif args.apoe4 == 'all':
            filter = valid['APOE4'].isin([0, 1, 2])

        valid = valid[filter]
        t = torch.tensor(valid[input_t].to_numpy())
        x = torch.tensor(np.expand_dims(np.expand_dims((valid["normalized"]), 1), 1))
        return x, t
    """read in adni Data ad Dataset"""
    def __init__(self,csv_file,btseed):
        self.adni = pd.read_csv(csv_file)
        self.SparseData=[]
        x1, t1 = self.cleanXT(args.var1, args.t1, args.bootstrap,btseed)
        x2, t2 = self.cleanXT(args.var2, args.t2, args.bootstrap,btseed)
        x3, t3 = self.cleanXT(args.var3, args.t3, args.bootstrap,btseed)
        x4, t4 = self.cleanXT(args.var4, args.t3, args.bootstrap,btseed)
        x5, t5 = self.cleanXT(args.var5, args.t3, args.bootstrap,btseed)
        x6, t6 = self.cleanXT(args.var6, args.t3, args.bootstrap,btseed)
        x7, t7 = self.cleanXT(args.var7, args.t3, args.bootstrap,btseed)
        self.SparseData.append((t1, t2, t3, t4, t5, t6, t7, x1, x2, x3, x4, x5, x6, x7))

    def __len__(self):
        return len(self.SparseData)

    def __getitem__(self, index):
        return self.SparseData[index]

def run(btseed):
    folder = osp.join(args.outdir, str(btseed))

    if not osp.exists(folder):
        os.makedirs(folder)

    with open(osp.join(folder, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    t_dim = args.t_dim
    y_dim = args.y_dim
    h_dim = args.h_dim
    func = None

    func = ApplicationODEFunc(t_dim, h_dim, y_dim,args.exclude_time)

    # optimizer = torch.optim.Adam(func.parameters(), lr=args.lr, weight_decay=args.decay)
    dataset = adniData(args.data,btseed)
    torch.save(dataset,osp.join(folder,('Data.pt')))

    data_loader = DataLoader(dataset, shuffle=True)
    sim = False
    ts_equal = False
    if not osp.exists(folder):
        os.makedirs(folder)
    seed = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(sim, device, ts_equal, func, folder, seed, True,lr=args.lr, weight_decay=args.decay,
                      early_stop=args.earlyStop)
    print('Training...')
    start_time = time.time()
    trainer.train(data_loader, args.Bepoch, args.epoch, seed)
    torch.save(func, osp.join(folder, ('trained_model_' + str(seed) + '.pth')))
    end_time = time.time()

if __name__ == "__main__":
    rep = range(args.btseedStart,args.btseedEnd)
    num_cores = multiprocessing.cpu_count()
    print('num_cores:' + str(num_cores))
    Parallel(n_jobs=num_cores)(delayed(run)(i) for i in rep)




