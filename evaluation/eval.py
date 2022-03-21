import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
from torchdiffeq import odeint
from data.dataset import DeterministicLotkaVolterraData
import argparse
from torch.utils.data import DataLoader
from models.utils import prediction
import json

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


def final_eval_default(folder,iter_end):
    datasets = np.load(folder + '/Data_0.npy', allow_pickle=True)
    time,x_true = datasets[0][0] #true dense values
    # time_s,x_obs_s,x_true_s = datasets[0][1]
    predX = []
    for iter in range(iter_end):
        predX.append(np.load(folder + '/predX_'+str(iter)+'.npy', allow_pickle=True))

    # predX_all=torch.stack(predX)
    predX_all=torch.tensor(np.array(predX))

    bias_total=torch.mean(torch.mean(torch.abs(predX_all-x_true),0),0)
    mse_total=torch.mean(torch.mean(torch.square(predX_all-x_true),0),0)
    var_total=torch.mean(torch.var(predX_all,0),0)
    print(bias_total)
    print(mse_total)
    print(var_total)

    np.savetxt(osp.join(folder, ('biasTotal.txt')), bias_total.detach().numpy())
    np.savetxt(osp.join(folder, ('mseTotal.txt')), mse_total.detach().numpy())
    np.savetxt(osp.join(folder, ('varsTotal.txt')), var_total.detach().numpy())

    empirical_mean=torch.mean(predX_all,0)
    print(torch.mean(empirical_mean,0))
    empirical_sd=torch.std(predX_all,0)
    print(torch.mean(empirical_sd**2,0))

    # empirical_sd = torch.sqrt(torch.mean(torch.square(predX_-empirical_mean),0))

    #plot true curve, empirical mean and empirical sd
    meanX1=empirical_mean.detach().numpy()[:,0]
    meanX2=empirical_mean.detach().numpy()[:,1]
    stdX1 = empirical_sd.detach().numpy()[:,0]
    stdX2 = empirical_sd.detach().numpy()[:,1]

    plt.figure()
    plt.plot(time.numpy(), x_true.numpy()[:, 0],label='True X1')
    plt.plot(time.numpy(), x_true.numpy()[:, 1],label='True X2')
    # plt.plot(time.numpy(),meanX1,label='Empirical Mean of X1')
    # plt.plot(time.numpy(),meanX2,label='Empirical Mean of X2')
    # plt.fill_between(time.numpy(), meanX1-stdX1,meanX1+stdX1,alpha=0.3,label='One std of X1 prediction')
    # plt.fill_between(time.numpy(), meanX2-stdX2,meanX2+stdX2,alpha=0.3,label='One std of X2 prediction')
    print("Number of predX: "+str(len(predX_all)))
    for i in range(len(predX_all)):
        plt.plot(time.numpy(),predX_all[i,:, 0].cpu().detach().numpy(),c='#6E8B3D',linewidth=0.1,alpha=0.3)
        plt.plot(time.numpy(),predX_all[i,:, 1].cpu().detach().numpy(),c='#8B8378',linewidth=0.1,alpha=0.3)

    plt.legend()

    # plt.show()
    plt.savefig(folder + "/summaryPlot_lines")

    plt.figure()
    plt.plot(time.numpy(), x_true.numpy()[:, 0], label='True X1')
    plt.plot(time.numpy(), x_true.numpy()[:, 1], label='True X2')
    plt.plot(time.numpy(),meanX1,label='Empirical Mean of X1')
    plt.plot(time.numpy(),meanX2,label='Empirical Mean of X2')
    plt.fill_between(time.numpy(), meanX1-stdX1,meanX1+stdX1,alpha=0.3,label='One std of X1 prediction')
    plt.fill_between(time.numpy(), meanX2-stdX2,meanX2+stdX2,alpha=0.3,label='One std of X2 prediction')
    # for i in range(len(predX_all)):
    #     plt.plot(time.numpy(), predX_all[i, :, 0].cpu().detach().numpy(), c='#6E8B3D', linewidth=0.1, alpha=0.3)
    #     plt.plot(time.numpy(), predX_all[i, :, 1].cpu().detach().numpy(), c='#8B8378', linewidth=0.1, alpha=0.3)

    plt.legend()

    # plt.show()
    plt.savefig(folder + "/summaryPlot_mean")

    print('evaluation done!')

def final_eval_range(folder,iter_end,sdense,outdir):
    if args.data == 'deterministic_lv':
        dataset = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10,
                                                 num_samples=args.num_samples, scenario=args.scenario, sd_u=args.sd_u,
                                                 sd_v=args.sd_v, rho=args.rho, lambdaX1=args.lambdaX1,
                                                 lambdaX2=args.lambdaX2, sdense=sdense,
                                                 ts_equal=args.ts_equal,
                                                 num_obs_x1=args.num_obs_x1, num_obs_x2=args.num_obs_x2)

    for iter in range(iter_end):
        # datasets = np.load(folder + '/Data_0.npy', allow_pickle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        time,x_true = dataset[0][0] #true dense values
        # time_s,x_obs_s,x_true_s = datasets[0][1]
        func = torch.load(osp.join(folder, 'trained_model_'+str(iter)+'.pth')).to(device)
        batch_size = args.num_samples

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        prediction(args.ts_equal,func,data_loader,sdense,iter,device,outdir)

    predX = []
    for iter in range(iter_end):
        predX.append(np.load(outdir + '/predX_'+str(iter)+'.npy', allow_pickle=True))

    # predX_all=torch.stack(predX)
    predX_all=torch.tensor(np.array(predX))

    bias_total=torch.mean(torch.mean(torch.abs(predX_all-x_true),0),0)
    mse_total=torch.mean(torch.mean(torch.square(predX_all-x_true),0),0)
    var_total=torch.mean(torch.var(predX_all,0),0)
    print(bias_total)
    print(mse_total)
    print(var_total)

    np.savetxt(osp.join(outdir, ('biasTotal.txt')), bias_total.detach().numpy())
    np.savetxt(osp.join(outdir, ('mseTotal.txt')), mse_total.detach().numpy())
    np.savetxt(osp.join(outdir, ('varsTotal.txt')), var_total.detach().numpy())

    empirical_mean=torch.mean(predX_all,0)
    print(torch.mean(empirical_mean,0))
    empirical_sd=torch.std(predX_all,0)
    print(torch.mean(empirical_sd**2,0))

    # empirical_sd = torch.sqrt(torch.mean(torch.square(predX_-empirical_mean),0))

    #plot true curve, empirical mean and empirical sd
    meanX1=empirical_mean.detach().numpy()[:,0]
    meanX2=empirical_mean.detach().numpy()[:,1]
    stdX1 = empirical_sd.detach().numpy()[:,0]
    stdX2 = empirical_sd.detach().numpy()[:,1]

    plt.figure()
    plt.plot(time.numpy(), x_true.numpy()[:, 0],label='True X1')
    plt.plot(time.numpy(), x_true.numpy()[:, 1],label='True X2')
    # plt.plot(time.numpy(),meanX1,label='Empirical Mean of X1')
    # plt.plot(time.numpy(),meanX2,label='Empirical Mean of X2')
    # plt.fill_between(time.numpy(), meanX1-stdX1,meanX1+stdX1,alpha=0.3,label='One std of X1 prediction')
    # plt.fill_between(time.numpy(), meanX2-stdX2,meanX2+stdX2,alpha=0.3,label='One std of X2 prediction')
    print("Number of predX: "+str(len(predX_all)))
    for i in range(len(predX_all)):
        plt.plot(time.numpy(),predX_all[i,:, 0].cpu().detach().numpy(),c='#6E8B3D',linewidth=0.1,alpha=0.3)
        plt.plot(time.numpy(),predX_all[i,:, 1].cpu().detach().numpy(),c='#8B8378',linewidth=0.1,alpha=0.3)

    plt.legend()

    # plt.show()
    plt.savefig(outdir + "/summaryPlot_lines")

    plt.figure()
    plt.plot(time.numpy(), x_true.numpy()[:, 0], label='True X1')
    plt.plot(time.numpy(), x_true.numpy()[:, 1], label='True X2')
    plt.plot(time.numpy(),meanX1,label='Empirical Mean of X1')
    plt.plot(time.numpy(),meanX2,label='Empirical Mean of X2')
    plt.fill_between(time.numpy(), meanX1-stdX1,meanX1+stdX1,alpha=0.3,label='One std of X1 prediction')
    plt.fill_between(time.numpy(), meanX2-stdX2,meanX2+stdX2,alpha=0.3,label='One std of X2 prediction')
    # for i in range(len(predX_all)):
    #     plt.plot(time.numpy(), predX_all[i, :, 0].cpu().detach().numpy(), c='#6E8B3D', linewidth=0.1, alpha=0.3)
    #     plt.plot(time.numpy(), predX_all[i, :, 1].cpu().detach().numpy(), c='#8B8378', linewidth=0.1, alpha=0.3)

    plt.legend()

    # plt.show()
    plt.savefig(outdir + "/summaryPlot_mean")

    print('evaluation done!')

if __name__ == "__main__":

    # Make folder
    folder = osp.join(args.outdir, args.scenario, args.exp_name)
    outdir = osp.join(args.outdir, args.scenario, args.exp_name,'eval20')
    if not osp.exists(outdir):
        os.makedirs(outdir)

    with open(osp.join(outdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    sdense = np.linspace(0, 20, 100)
    final_eval_range(folder,args.iter_end,sdense,outdir)

