import numpy as np
import os
import os.path as osp
import torch

def obsPercent(folder,iter_end,ts_equal):
    valid1=0
    total1=0
    valid2=0
    total2=0
    for iter in range(iter_end):
        datasets = np.load(folder + '/Data_'+str(iter)+'.npy', allow_pickle=True)
        for i in range(len(datasets)):
            if ts_equal==True:
                timeSparse, states_obsSparse, states_trueSparse=datasets[i][1]
                valid1=valid1+len(timeSparse[timeSparse<=15])
                total1=total1+len(timeSparse)
            else:
                t1, t2, x1_obs,x2_obs,x1_true,x2_true = datasets[i][1]
                # time, x_true = datasets[0][0]
                valid1=valid1+len(t1[t1<=15])
                valid2=valid2+len(t2[t2<=15])
                total1=total1+len(t1)
                total2=total2+len(t2)
    valid=valid1+valid2
    total=total1+total2
    percent = torch.tensor([valid/total,valid,total])
    # print(valid)
    # print(total)
    # print(percent)
    np.savetxt(osp.join(folder, ('obsPercent.txt')), percent.detach().numpy())



# if __name__ == "__main__":
#     folder='/N/slate/liyuny/cnode_ffr_main/results/formal_sim/lambda_4.0/simB2/100_0.3_5_True_0.2_4.0'
#     obsPercent(folder,10,True)


