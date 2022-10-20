import argparse
import os
import os.path as osp
import json
# from tqdm import tqdm

from models.NODEmodels import *
from data.dataset import *
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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--rangeMax', type=int, default=15)

parser.add_argument('--model', type=str, choices=['vnode','timevnode','application'], default='vnode')
parser.add_argument('--scenario', type=str, choices=['simA','simB','simC','simB2'],default='simA') # simulation scenario
parser.add_argument('--sd_v', type=float, default=0.) # sd for error of X1
parser.add_argument('--sd_u', type=float, default=0.) # sd for error of X2
parser.add_argument('--rho_w', type=float, default=0.)  # correlation coefficient of X1(t), X1(s).
parser.add_argument('--rho_b', type=float, default=0.)  # correlation coefficient of X1(t), X2(t).

parser.add_argument('--lambdaX1', type=float, default=2.) # scale parameter for duration of follow up of X: exp(\lambda)
parser.add_argument('--lambdaX2', type=float, default=2.) # scale parameter for duration of follow up of X: exp(\lambda)

parser.add_argument('--data', type=str, choices=['deterministic_lv','functional','functional1'], default='deterministic_lv')
parser.add_argument('--h_dim', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=66)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iter_start', type=int, default=0)
parser.add_argument('--iter_end', type=int, default=0)

parser.add_argument('--num_samples', type=int, default=300)

parser.add_argument('--ts_equal',type=eval, choices=[True, False], default=False)
parser.add_argument('--num_obs_x1',type=int,default=5)
parser.add_argument('--num_obs_x2',type=int,default=5)
parser.add_argument('--ifplot', type=eval, choices=[True, False], default=False)

parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--outdir',type=str,default='/N/u/liyuny/Carbonate/results/test')
args = parser.parse_args()

def run(device,seed):
    # Create dataset
    print('Generating Data...')
    print('Scenario: '+args.scenario)
    if args.data == 'deterministic_lv':
        sdense = np.linspace(0, args.rangeMax, 100)
        dataset = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10,
                                                num_samples=args.num_samples, scenario=args.scenario,sd_u=args.sd_u,
                                                sd_v=args.sd_v, lambdaX1=args.lambdaX1,rho_w=args.rho_w,rho_b=args.rho_b,
                                                lambdaX2=args.lambdaX2,sdense=sdense,
                                                ts_equal=args.ts_equal,
                                                num_obs_x1=args.num_obs_x1,num_obs_x2=args.num_obs_x2,seed=seed)
    elif args.data == 'functional':
        sdense = np.linspace(0, args.rangeMax, 100)
        dataset = FunctionalData(sdense=sdense, num_samples=args.num_samples, sd_u=args.sd_u, sd_v=args.sd_v,
                                  rho_b=args.rho_b,rho_w=args.rho_w, scenario=args.scenario,num_obs_x1=args.num_obs_x1,
                                  num_obs_x2=args.num_obs_x2,lambdaX1=args.lambdaX1,lambdaX2=args.lambdaX2,ts_equal=args.ts_equal,seed=seed)
    elif args.data == 'functional1':
        sdense = np.linspace(0, args.rangeMax, 100)
        dataset = FunctionalData1(sdense=sdense, num_samples=args.num_samples, sd_u=args.sd_u, sd_v=args.sd_v,
                                  rho_b=args.rho_b,rho_w=args.rho_w, scenario=args.scenario,num_obs_x1=args.num_obs_x1,
                                  num_obs_x2=args.num_obs_x2,lambdaX1=args.lambdaX1,lambdaX2=args.lambdaX2,ts_equal=args.ts_equal,seed=seed)
    t_dim = 1
    y_dim = 2
    h_dim = args.h_dim

    func = None
    if args.model == 'vnode':
        func = VanillaODEFunc(t_dim, h_dim, y_dim, exclude_time=True).to(device)
    elif args.model == 'timevnode':
        func = ODEFuncTimeVariate(t_dim, h_dim, y_dim, exclude_time=False).to(device)
    elif args.model == 'application':
        func = ApplicationODEFunc(t_dim, h_dim, y_dim, exclude_time=True).to(device)

    if args.load:
        func = torch.load(osp.join(folder, 'trained_model_'+str(seed)+'.pth')).to(device)
    else:
        torch.save(func, osp.join(folder, 'untrained_model.pth'))

    # training

    batch_size = args.num_samples
    # batch_size = 100

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)
    sim = True
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
    pred=prediction(args.ts_equal,func,data_loader,sdense,seed,device,folder)
    return pred


def iteration(device,iter_start,iter_end):
    predX = []
    rep = range(iter_start,iter_end)
    num_cores = multiprocessing.cpu_count()
    print('num_cores:' + str(num_cores))
    predX = Parallel(n_jobs=num_cores)(delayed(run)(device,i) for i in rep)

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
    # print(parser.parse_args())
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
    final_eval_default(folder,args.iter_end)
    whole_end_time = time.time()
    op(folder,args.iter_end,args.ts_equal,args.rangeMax,folder)
    print('Replication total time = ' + str(whole_end_time - whole_start_time))





