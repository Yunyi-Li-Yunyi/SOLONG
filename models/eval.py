import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
from torchdiffeq import odeint

def final_eval(folder,iter_end):
    datasets = np.load(folder + '/Data_0.npy', allow_pickle=True)
    time,x_true = datasets[0][0] #true dense values
    # time_s,x_obs_s,x_true_s = datasets[0][1]
    predX = []
    for seed in range(iter_end):
        predX.append(np.load(folder + '/predX_'+str(seed)+'.npy', allow_pickle=True))

    # predX_all=torch.stack(predX)
    predX_all=torch.tensor(np.array(predX))

    bias_total=torch.mean(torch.mean(torch.abs(predX_all-x_true),0),0)
    var_total=torch.mean(torch.mean(torch.square(predX_all-x_true),0),0)
    np.savetxt(osp.join(folder, ('biasTotal.txt')), bias_total.detach().numpy())
    np.savetxt(osp.join(folder, ('varsTotal.txt')), var_total.detach().numpy())

    empirical_mean=torch.mean(predX_all,0)
    empirical_sd=torch.std(predX_all,0)

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

# if __name__ == "__main__":
#     folder = '/N/u/liyuny/Quartz/cnode_ffr_main/results/formal_sim/simA/300_0.3_10_10'
#     final_eval(folder,1000)
# plt.figure()
# plt.plot(time.numpy(), x_true.numpy()[:, 0],label='True X1')
# plt.plot(time.numpy(), x_true.numpy()[:, 1],label='True X2')
#
# for i in range(len(datasets)):
#     time_s,x_obs_s,x_true_s = datasets[i][1]
#     plt.scatter(time_s.numpy(),x_obs_s.detach().numpy()[:,0],marker='x', c='#7AC5CD')
#     plt.scatter(time_s.numpy(),x_obs_s.detach().numpy()[:,1],marker='x', c='#C1CDCD')
#
# plt.show()