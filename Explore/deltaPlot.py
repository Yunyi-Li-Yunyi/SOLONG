import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def deltaPlot(folder,iter_end,ts_equal):
    for iter in range(iter_end):
        datasets = np.load(folder + '/Data_'+str(iter)+'.npy', allow_pickle=True)
        for i in range(len(datasets)):
            if ts_equal==True:
                timeSparse, states_obsSparse, states_trueSparse=datasets[i][1]
                dt = torch.roll(timeSparse,-1,0)-timeSparse #delta time
                dv_obs = torch.transpose(torch.roll(states_obsSparse,-1,0)-states_obsSparse,0,1) #delta values
                dr_obs = dv_obs/dt

                dv_true = torch.transpose(torch.roll(states_trueSparse,-1,0)-states_trueSparse,0,1) #delta values
                dr_true = dv_true/dt
                # print(timeSparse[:len(timeSparse)-1])
                # print(dt)
                # print(states_trueSparse)
                # print(dv)
                # print(dr_obs)
                # print(dr[0])
                time,x_true =datasets[i][0]
                dt_dense = torch.roll(time,-1,0)-time #delta time
                dv_dense = torch.transpose(torch.roll(x_true,-1,0)-x_true,0,1) #delta values
                dr_dense = dv_dense/dt_dense

                pred=torch.tensor(np.load(folder + '/predX_' + str(iter) + '.npy', allow_pickle=True))
                dv_pred = torch.transpose(torch.roll(pred,-1,0)-pred,0,1) #delta values
                dr_pred=dv_pred/dt_dense


            plt.subplot(4,2,1)
            plt.plot(timeSparse[:len(timeSparse)-1].detach().numpy(),
                     dr_obs[0][:len(timeSparse)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for observation X1')
            plt.subplot(4,2,2)
            plt.plot(timeSparse[:len(timeSparse)-1].detach().numpy(),
                     dr_obs[1][:len(timeSparse)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for observation X2')

            plt.subplot(4,2,3)
            plt.plot(timeSparse[:len(timeSparse)-1].detach().numpy(),
                     dr_true[0][:len(timeSparse)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for true X1')
            plt.subplot(4,2,4)
            plt.plot(timeSparse[:len(timeSparse)-1].detach().numpy(),
                     dr_true[1][:len(timeSparse)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for true X2')

            plt.subplot(4,2,5)
            plt.plot(time[:len(time)-1].detach().numpy(),
                     dr_pred[0][:len(time)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for dense pred X1')
            plt.subplot(4,2,6)
            plt.plot(time[:len(time)-1].detach().numpy(),
                     dr_pred[1][:len(time)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for dense pred X2')

            plt.subplot(4,2,7)
            plt.plot(time[:len(time)-1].detach().numpy(),
                     dr_dense[0][:len(time)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for dense true X1')
            plt.subplot(4,2,8)
            plt.plot(time[:len(time)-1].detach().numpy(),
                     dr_dense[1][:len(time)-1].detach().numpy(),linewidth=0.3,alpha=0.7)
            plt.xlim([0,15])
            plt.title('Accumulative rate for dense true X2')

        plt.show()

            #
            # else:
            #     t1, t2, x1_obs,x2_obs,x1_true,x2_true = datasets[i][1]
            #     # time, x_true = datasets[0][0]
            #     valid1=valid1+len(t1[t1<=15])
            #     valid2=valid2+len(t2[t2<=15])
            #     total1=total1+len(t1)
            #     total2=total2+len(t2)
if __name__ == "__main__":
    folder='/N/slate/liyuny/cnode_ffr_main/results/formal_sim/simA/100_0.3_10_True_0.2'
    deltaPlot(folder,1,True)
