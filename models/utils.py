import torch
import numpy as np
from torchdiffeq import odeint
import os
import os.path as osp

def ObservedData(TimesSparse, StatesObsSparse, StatesTrueSparse):
    """
    Given Sparse inputs x and y (with nans) return combined x and y

    Parameters
    ----------
    TimesSparse: torch.Tensor
        Shape(batch_size, num_points, x_dim)
    StatesObsSparse: torch.Tensor
        Shape(batch_size, num_points, y_dim)
    StatesTrueSparse: torch.Tensor
        Shape(batch_size, num_points, y_dim)

    return: combined torch.Tensor obsX, obsY
    """
    # maskX = torch.tensor(torch.isnan(TimesSparse) == 0)
    maskX = (torch.isnan(TimesSparse) == 0).clone().detach()

    # maskY = torch.tensor(torch.isnan(StatesObsSparse) == 0)
    maskY = (torch.isnan(StatesObsSparse) == 0).clone().detach()

    obsX = torch.masked_select(TimesSparse, maskX)

    obsY = torch.masked_select(StatesObsSparse, maskY)
    obsY = torch.reshape(obsY, (-1, StatesObsSparse.shape[2]))

    trueY = torch.masked_select(StatesTrueSparse, maskY)
    trueY = torch.reshape(trueY, (-1, StatesTrueSparse.shape[2]))

    # sortX = torch.sort(obsX)[0]
    # index = torch.sort(obsX)[1]

    sortX, counts = torch.unique(obsX, sorted=True, return_counts=True)

    sortYobs = torch.empty((len(sortX), StatesObsSparse.shape[2]))
    sortYtrue = torch.empty((len(sortX), StatesTrueSparse.shape[2]))

    for i in range(sortX.size(0)):
        groupYobs = obsY[obsX == sortX[i]]
        groupYtrue = trueY[obsX == sortX[i]]
        sortYobs[i] = torch.mean(groupYobs, 0)
        sortYtrue[i] = torch.mean(groupYtrue, 0)

    return sortX, sortYobs, sortYtrue

def prediction(ts_equal,func,data_loader,sdense,seed,device,folder):
    if ts_equal==True:
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            t, x_obs, x_true = data[:][1]
            t = t.to(device)
            x_obs = x_obs.to(device)
            x_true = x_true.to(device)

            sort_t, sort_x_obs, sort_x_true = ObservedData(t, x_obs, x_true)
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

            sort_t1, sort_x1_obs, sort_x1_true = ObservedData(t1, x1_obs, x1_true)
            sort_t2, sort_x2_obs, sort_x2_true = ObservedData(t2, x2_obs, x2_true)

            # sort_t12, counts = torch.unique(sort_t1.extend(sort_t2), sorted=True, return_counts=True)

            x0 = torch.tensor([sort_x1_obs[0], sort_x2_obs[0]]).to(device)

    predX_full = odeint(func, x0, torch.tensor(sdense)).to(device)
    np.save(osp.join(folder, 'predX_'+str(seed)+'.npy'), predX_full.detach().numpy(),allow_pickle=True)
    return predX_full


def obsPercent(folder,iter_end,ts_equal,rangeMax,outdir):
    valid1=0
    total1=0
    valid2=0
    total2=0
    for iter in range(iter_end):
        datasets = np.load(folder + '/Data_'+str(iter)+'.npy', allow_pickle=True)
        for i in range(len(datasets)):
            if ts_equal==True:
                timeSparse, states_obsSparse, states_trueSparse=datasets[i][1]
                valid1=valid1+len(timeSparse[timeSparse<=rangeMax])
                total1=total1+len(timeSparse)
            else:
                t1, t2, x1_obs,x2_obs,x1_true,x2_true = datasets[i][1]
                # time, x_true = datasets[0][0]
                valid1=valid1+len(t1[t1<=rangeMax])
                valid2=valid2+len(t2[t2<=rangeMax])
                total1=total1+len(t1)
                total2=total2+len(t2)
    valid=valid1+valid2
    total=total1+total2
    percent = torch.tensor([valid/total,valid,total])
    # print(valid)
    # print(total)
    # print(percent)
    np.savetxt(osp.join(outdir, ('obsPercent.txt')), percent.detach().numpy())

