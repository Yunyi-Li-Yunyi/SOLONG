import argparse
import os
import os.path as osp
import json
from tqdm import tqdm

from models.NODEmodels import *
from data.dataset import DeterministicLotkaVolterraData
from models.training import Trainer

# from models.utils import PredData
from torch.utils.data import DataLoader
from models.utils import ObservedData as od

from joblib import Parallel, delayed
import multiprocessing
import time
from evaluation.eval import final_eval
import matplotlib.pyplot as plt
from evaluation.obsPercent import obsPercent as op

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name', type=str, required=True)

parser.add_argument('--model', type=str, choices=['vnode'], default='vnode')
parser.add_argument('--scenario', type=str, choices=['simA','simB','simC','simB2'],default='simA') # simulation senario
parser.add_argument('--sd_v', type=float, default=0.) # sd for error of X1
parser.add_argument('--sd_u', type=float, default=0.) # sd for error of X2
parser.add_argument('--rho', type=float, default=0.)  # correlation coefficient when X1, X2 not ind.
parser.add_argument('--lambdaX1', type=float, default=2.) # scale parameter for duration of follow up of X: exp(\lambda)
parser.add_argument('--lambdaX2', type=float, default=2.) # scale parameter for duration of follow up of X: exp(\lambda)

parser.add_argument('--data', type=str, choices=['deterministic_lv'], default='deterministic_lv')
parser.add_argument('--h_dim', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=66)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iter_start', type=int, default=0)
parser.add_argument('--iter_end', type=int, default=0)

parser.add_argument('--num_samples', type=int, default=300)

parser.add_argument('--ts_equal',type=eval, choices=[True, False], default=False)
parser.add_argument('--num_obs_x1',type=int,default=5)
parser.add_argument('--num_obs_x2',type=int,default=9)
parser.add_argument('--ifplot', type=eval, choices=[True, False], default=False)

parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--outdir',type=str,default='/N/u/liyuny/Carbonate/thindrives/Dissertation/node_ffr-main/results')
args = parser.parse_args()

def run(device,seed):

    # Create dataset
    print('Generating Data')
    print('Scenario: '+args.scenario)
    if args.data == 'deterministic_lv':
        sdense = np.linspace(0, 15, 100)
        dataset = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10,
                                                num_samples=args.num_samples, scenario=args.scenario,sd_u=args.sd_u,
                                                sd_v=args.sd_v, rho=args.rho, lambdaX1=args.lambdaX1,
                                                lambdaX2=args.lambdaX2,sdense=sdense,
                                                ts_equal=args.ts_equal,
                                                num_obs_x1=args.num_obs_x1,num_obs_x2=args.num_obs_x2,seed=seed)
    x_dim = 1
    y_dim = 2

    func = None
    if args.model == 'vnode':
        h_dim = args.h_dim
        func = VanillaODEFunc(x_dim, h_dim, y_dim).to(device)

    if args.load:
        func = torch.load(osp.join(folder, 'trained_model_'+str(seed)+'.pth')).to(device)
    else:
        torch.save(func, osp.join(folder, 'untrained_model.pth'))

    # training

    batch_size = args.num_samples
    # batch_size = 100

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)
    sim=True
    ts_equal = args.ts_equal
    trainer = Trainer(sim,device,ts_equal, func, optimizer, folder,seed,args.ifplot)

    print('Training...')
    start_time = time.time()
    trainer.train(data_loader, args.epochs,seed)
    end_time = time.time()
    print('Total time = ' + str(end_time - start_time))

    # np.save(osp.join(folder, ('loss_history_'+str(seed)+'.npy')), np.array(trainer.epoch_loss_history))
    # np.save(osp.join(folder, ('training_time_'+str(seed)+'.npy')), np.array([end_time - start_time]))
    np.save(osp.join(folder, ('Data_'+str(seed)+'.npy')), dataset,allow_pickle=True)

    torch.save(func, osp.join(folder, ('trained_model_'+str(seed)+'.pth')))

    if args.ts_equal==True:
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            t, x_obs, x_true = data[:][1]
            t = t.to(device)
            x_obs = x_obs.to(device)
            x_true = x_true.to(device)

            sort_t, sort_x_obs, sort_x_true = od(t, x_obs, x_true)
            x0 = sort_x_obs[0].to(device)
    else:
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            t1, t2, x1_obs, x2_obs, x1_true, x2_true = data[:][1]
            t1 = t1.to(device)
            t2 = t2.to(device)

            x1_obs = x1_obs.to(device)
            x2_obs = x2_obs.to(device)

            x1_true = x1_true.to(device)
            x2_true = x2_true.to(device)

            sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
            sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)

            # sort_t12, counts = torch.unique(sort_t1.extend(sort_t2), sorted=True, return_counts=True)

            x0 = torch.tensor([sort_x1_obs[0], sort_x2_obs[0]]).to(device)

    predX_full = odeint(func, x0, torch.tensor(sdense)).to(device)
    np.save(osp.join(folder, 'predX_'+str(seed)+'.npy'), predX_full.detach().numpy(),allow_pickle=True)


    # Sfull, predX_full = PredData(device,func, 1., 1.,timepoint=sdense)
    # predX_full=predX_full.to(device)
    # Sfull=Sfull.to(device)
    return predX_full

def iteration(device,iter_start,iter_end):
    predX = []
    rep = range(iter_start,iter_end)
    num_cores = multiprocessing.cpu_count()
    print('num_cores:' + str(num_cores))
    predX=Parallel(n_jobs=num_cores)(delayed(run)(device,i) for i in rep)

    # for i in range(iter):
    #     print('Iteration number: ' + str(iter))
    #     predX_full = run(device,i)
    #     predX.append(predX_full)
    np.save(osp.join(folder, 'predX.npy'), predX,allow_pickle=True)

def outputPlot():
    datasets = np.load(folder + '/Data_0.npy', allow_pickle=True)

    # plot true curve
    time,x_true = datasets[0][0]
    plt.figure()
    plt.plot(time.numpy(), x_true.numpy()[:, 0])
    plt.plot(time.numpy(), x_true.numpy()[:, 1])

    # plot predicted X values
    predX = np.load(folder + '/predX.npy', allow_pickle=True)
    for i in range(len(predX)):
        plt.plot(time.numpy(),predX[i][:, 0].cpu().detach().numpy(),c='#6E8B3D',linewidth=0.3,alpha=0.4)
        plt.plot(time.numpy(),predX[i][:, 1].cpu().detach().numpy(),c='#8B8378',linewidth=0.3,alpha=0.4)

    # plt.show()
    plt.savefig(folder + "/outputPlot" )

if __name__ == "__main__":
    # Make folder
    whole_start_time = time.time()
    folder = osp.join(args.outdir, args.scenario, args.exp_name)
    print(folder)

    if not osp.exists(folder):
        os.makedirs(folder)

    with open(osp.join(folder, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    iteration(device,args.iter_start,args.iter_end)
    outputPlot()
    final_eval(folder,args.iter_end)
    whole_end_time = time.time()
    op(folder,args.iter_end,args.ts_equal)
    print('Replication total time = ' + str(whole_end_time - whole_start_time))





