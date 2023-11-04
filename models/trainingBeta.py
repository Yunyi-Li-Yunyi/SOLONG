import torch
import numpy as np
# from tqdm import tqdm
from typing import Tuple
from random import randint
from models.utils import ObservedData as od
from torch.utils.data import DataLoader
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import time
import os.path as osp
import math
import pandas as pd
from torch import nn
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
class Trainer:
    """
    Class to handle training

    Parameters
    ----------
    device: torch.device
    ODEFunc: deep learning ODE function
    optimizer: one of torch.optim optimizers
    folder: output folder path

    sim: Bool
        Training based on simulation data

    """

    def __init__(self,
                 sim,
                 device: torch.device,
                 ts_equal,
                 ODEFunc,
                 folder,
                 seed,
                 ifplot,
                 lr,
                 weight_decay,
                 early_stop,
                 optimizer,
                 initialrefine,
                 initbeta
                 ):
        self.sim = sim
        self.device = device
        # self.optimizer = optimizer
        self.ts_equal = ts_equal
        self.ODEFunc = ODEFunc
        self.epoch_loss_history = []
        self.folder = folder
        self.seed = seed
        self.ifplot = ifplot
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop = early_stop
        self.optimizer = optimizer
        self.initialrefine = initialrefine
        self.initbeta = initbeta

        np.random.seed(self.seed)

    def train(self, train_data_loader: DataLoader, Bepochs, epochs: int, seed):
        """
        Train Neural ODE.

        Parameters
        ----------
        DataLoader:
            Dataloader
        Bepochs: int
            Number of block epochs to train for
        epochs: int
            Number of epochs to train for
        """
        # self.train()
        updatedx0 = torch.tensor([0.1,0.1])
        x0Record = []
        betaHat = torch.tensor(self.initbeta, dtype=torch.float32, device=self.device)[:,None]
        if self.initialrefine==False:
            for epoch in range(epochs):
                self.train_epoch(train_data_loader, epoch, betaHat, seed)
        else:
            for blockEpoch in range(Bepochs):
                print(f'BlockEpoch {blockEpoch}')
                print(f'Seed {seed}')
                epoch_start_time = time.time()

                for epoch in range(1):
                    if self.sim == True:
                        epoch_loss,updatedx0 = self.train_epoch_TrainInit(train_data_loader, blockEpoch,epoch, seed,"theta",updatedx0)
                        # during theta training phase, set updatedx0=0 means there's no update for x0
                    else:
                        epoch_loss,updatedx0 = self.train_epoch_RealData_TrainInit(train_data_loader, blockEpoch,epoch,"theta",updatedx0)
                    self.epoch_loss_history.append(epoch_loss)
                    print('theta_B:{},\n theta_epoch: {}, \n theta_epoch_loss: {}'.format(blockEpoch,epoch,epoch_loss))
                    epoch_end_time = time.time()
                    print('theta_Each Epoch time = ' + str(epoch_end_time - epoch_start_time))
                    print('theta_updatedx0 output:{}'.format(updatedx0))

                if self.sim == True:
                    epoch_loss,updatedx0 = self.train_epoch_TrainInit(train_data_loader, blockEpoch, epoch, seed, "init",updatedx0)
                else:
                    epoch_loss,updatedx0 = self.train_epoch_RealData_TrainInit(train_data_loader, blockEpoch, epoch,"init", updatedx0)
                x0Record.append(updatedx0)
                print('init_B:{},\n init_epoch: {}, \n init_epoch_loss: {}'.format(blockEpoch, epoch,epoch_loss))
                print('init_updatedx0 output:{}'.format(updatedx0))
            torch.save(x0Record, osp.join(self.folder, 'x0Record' + '.pt'))


    def train_epoch(self, data_loader, epoch, betaHat,seed):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()

            # Extract data
            if self.sim == True and self.ts_equal == True:
                t, x_obs, x_true, c = data[:][1]
                t = t.to(self.device)
                x_obs = x_obs.to(self.device)
                x_true = x_true.to(self.device)
                sort_t, sort_x_obs, sort_x_true = od(t, x_obs, x_true)
                x0 = sort_x_obs[0].to(self.device)
                sort_t = sort_t.to(self.device)
                sort_x_obs = sort_x_obs.to(self.device)

                # x0_guess=torch.nn.Parameter(torch.randint(2,4,(2,)).float(), requires_grad=False)
                # print(x0_guess)
                # x0 = torch.tensor(self.initialNet(0,x0_guess))
                # print(x0)
                pred_x = odeint(self.ODEFunc, x0, sort_t).to(self.device)
                pred_reshape = pred_x.view(-1,2)
                print(pred_reshape.size())
                print(sort_x_obs.size())
                print(betaHat.size())
                print(c.size())
                loss = torch.mean(torch.square(pred_reshape - sort_x_obs - betaHat@c))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()

                ymat = sort_x_obs - pred_reshape # num_samples*num_obs_x1*2 = Y-mu(t)
                data = pd.DataFrame({
                    "y_mu1": list(ymat[:,0].detach().numpy().reshape(-1)),
                    "y_mu2": list(ymat[:,1].detach().numpy().reshape(-1)),
                    "c": list(np.repeat(c.detach().numpy(), x_obs.size(1)))
                })
                X = data[['c']]
                y1 = data['y_mu1']
                y2 = data['y_mu2']
                model1 = sm.OLS(y1,X)
                model2 = sm.OLS(y2, X)
                result1 = model1.fit()
                result2 = model2.fit()
                print(result1.summary())
                print(result2.summary())

            elif self.sim == True and self.ts_equal == False:
                t1, t2, x1_obs, x2_obs, x1_true, x2_true = data[:][1]
                t1 = t1.to(self.device)
                t2 = t2.to(self.device)

                x1_obs = x1_obs.to(self.device)
                x2_obs = x2_obs.to(self.device)

                x1_true = x1_true.to(self.device)
                x2_true = x2_true.to(self.device)

                sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
                sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)

                # sort_t12, counts = torch.unique(sort_t1.extend(sort_t2), sorted=True, return_counts=True)

                x0 = torch.tensor([sort_x1_obs[0], sort_x2_obs[0]]).to(self.device)
                # sort_t12=sort_t12.to(self.device)
                sort_t1 = sort_t1.to(self.device)
                sort_t2 = sort_t2.to(self.device)
                sort_x1_obs = sort_x1_obs.to(self.device)
                sort_x2_obs = sort_x2_obs.to(self.device)

                pred_x1 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t1).to(self.device)[:, 0], 1)
                pred1_reshape = pred_x1.view(-1, 2)
                pred_x2 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t2).to(self.device)[:, 1], 1)
                pred2_reshape = pred_x2.view(-1, 2)
                loss_x1 = torch.mean(torch.square(pred1_reshape - sort_x1_obs))
                loss_x2 = torch.mean(torch.square(pred2_reshape - sort_x2_obs))
                loss = torch.mean(torch.stack([loss_x1, loss_x2]))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()

            # plot checking
            if self.ifplot==True:
                if epoch % 500 == 0:
                    # full time points
                    Xfull, trueYfull = data[:][0]
                    Xfull = Xfull.to(self.device)
                    trueYfull = trueYfull.to(self.device)
                    # y0true = trueYfull[0, 0, :]
                    pred_full = odeint(self.ODEFunc, x0, Xfull[0, :])

                    plt.figure()
                    # full true curve
                    plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 0], label="X1_true", c='r')
                    plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 1], label='X2_true', c='g')
                    # full prediction curve
                    plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 0], label="X1_pred", c='orange')
                    plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 1], label="X2_pred", c='c')
                    # observed data with noise
                    if self.ts_equal == True:
                        plt.scatter(t.cpu().numpy(), x_obs.cpu().numpy()[:, :, 0], marker='x', c='#7AC5CD', alpha=0.7)
                        plt.scatter(t.cpu().numpy(), x_obs.cpu().numpy()[:, :, 1], marker='x', c='#C1CDCD', alpha=0.7)
                        plt.scatter(t.cpu().numpy()[0, :], x_obs.cpu().numpy()[0, :, 0], color="none", edgecolor='r', s=20)
                        plt.scatter(t.cpu().numpy()[0, :], x_obs.cpu().numpy()[0, :, 1], color="none", edgecolor='g', s=13)
                    else:
                        plt.scatter(t1.cpu().numpy(), x1_obs.cpu().numpy()[:], marker='x', c='#7AC5CD', alpha=0.7)
                        plt.scatter(t2.cpu().numpy(), x2_obs.cpu().numpy()[:], marker='x', c='#C1CDCD', alpha=0.7)
                        plt.scatter(t1.cpu().numpy()[0, :], x1_obs.cpu().numpy()[0, :], color="none", s=20, edgecolor='r')
                        plt.scatter(t2.cpu().numpy()[0, :], x2_obs.cpu().numpy()[0, :], color="none", s=13, edgecolor='g')

                    plt.legend()
                    plt.savefig(self.folder + "/plot" + str(seed) + '_' + str(epoch))
        print("epoch_loss:{}".format(epoch_loss))
        return epoch_loss, betaHat

    def train_epoch_TrainInit(self, data_loader, blockEpoch, epoch, seed, param, betaHat, updatedx0):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()

            # Extract data
            if self.sim == True and self.ts_equal == True:
                t, x_obs, x_true,c = data[:][1]
                t = t.to(self.device)
                x_obs = x_obs.to(self.device)
                x_true = x_true.to(self.device)
                sort_t, sort_x_obs, sort_x_true = od(t, x_obs, x_true)
                x0 = updatedx0.view(1, 2)
                print('initial value:{}'.format(str(x0)))
                sort_t = sort_t.to(self.device)
                sort_x_obs = sort_x_obs.to(self.device)
                pred_x = odeint(self.ODEFunc, x0, sort_t).to(self.device)
                pred_reshape = pred_x.view(-1,2)

                if param == "theta":
                    params = (list(self.ODEFunc.parameters()))
                    loss = torch.mean(torch.square(pred_reshape - sort_x_obs - betaHat @ c))
                    l1_lambda = 0.0001
                    l1_norm = sum(torch.linalg.norm(p, 1) for p in params)
                    loss = loss + l1_lambda * l1_norm
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.cpu().item()

                elif param == 'init':
                    # print('calculate initial value based on trapezoidal rule...')
                    updatedx0=(torch.mean(sort_x_obs-(pred_reshape-x0),dim=0)).detach()
                    print('updated initial value:{}'.format(updatedx0))
                else:
                    print('please specify optimizing parameter!')



            elif self.sim == True and self.ts_equal == False:
                t1, t2, x1_obs, x2_obs, x1_true, x2_true = data[:][1]
                t1 = t1.to(self.device)
                t2 = t2.to(self.device)

                x1_obs = x1_obs.to(self.device)
                x2_obs = x2_obs.to(self.device)

                x1_true = x1_true.to(self.device)
                x2_true = x2_true.to(self.device)

                sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
                sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)

                x0 = updatedx0.view(1, 2)
                print('initial value:{}'.format(str(x0)))

                # x0 = torch.tensor([sort_x1_obs[0], sort_x2_obs[0]]).to(self.device)
                # sort_t12=sort_t12.to(self.device)
                sort_t1 = sort_t1.to(self.device)
                sort_t2 = sort_t2.to(self.device)
                sort_x1_obs = sort_x1_obs.to(self.device)
                sort_x2_obs = sort_x2_obs.to(self.device)

                pred_x1 = odeint(self.ODEFunc, x0, sort_t1).to(self.device)
                pred_x2 = odeint(self.ODEFunc, x0, sort_t2).to(self.device)
                pred1_reshape = torch.unsqueeze(pred_x1.view(-1,2)[:,0],1)
                pred2_reshape = torch.unsqueeze(pred_x2.view(-1,2)[:,1],1)
                # print(sort_x1_obs.size())
                # print(sort_x2_obs.size())
                # print(pred_x1.size())
                # print(pred_x2.size())
                # print(odeint(self.ODEFunc, x0, sort_t1).size())
                # print(odeint(self.ODEFunc, x0, sort_t2).size())
                # print(pred1_reshape.size())
                # print(pred2_reshape.size())
                if param == "theta":
                    loss_x1 = torch.mean(torch.square(pred1_reshape - sort_x1_obs))
                    loss_x2 = torch.mean(torch.square(pred2_reshape - sort_x2_obs))
                    loss = torch.mean(torch.stack([loss_x1, loss_x2]))
                    params = (list(self.ODEFunc.parameters()))
                    # params = (list(x0func.parameters()))
                    optimizer = torch.optim.Adam(params, lr=self.lr)
                    optimizer.zero_grad()
                    l1_lambda = 0.001
                    l1_norm = sum(torch.linalg.norm(p, 1) for p in params)
                    loss = loss + l1_lambda * l1_norm
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.cpu().item()

                elif param == 'init':
                    part1 = sort_x1_obs - (pred1_reshape - x0[:,0])
                    part2 = sort_x2_obs - (pred2_reshape - x0[:,1])
                    updatedx0 = torch.tensor([torch.mean(part1,dim=0).detach(),torch.mean(part2,dim=0).detach()])
                print('updated initial value:{}'.format(updatedx0))
                # mse = ((sortYobs - pred_y.mean)**2).mean()


            # plot checking
            if blockEpoch % 299 ==0:
                # full time points
                Xfull, trueYfull = data[:][0]
                Xfull = Xfull.to(self.device)
                trueYfull = trueYfull.to(self.device)
                # y0true = trueYfull[0, 0, :]
                pred_full = odeint(self.ODEFunc, updatedx0, Xfull[0, :])

                plt.figure()
                # full true curve
                plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 0], label="X1_true", c='r')
                plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 1], label='X2_true', c='g')
                # full prediction curve
                plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 0], label="X1_pred", c='orange')
                plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 1], label="X2_pred", c='c')
                # observed data with noise
                if self.ts_equal == True:
                    plt.scatter(t.cpu().numpy(), x_obs.cpu().numpy()[:, :, 0], marker='x', c='#7AC5CD', alpha=0.7)
                    plt.scatter(t.cpu().numpy(), x_obs.cpu().numpy()[:, :, 1], marker='x', c='#C1CDCD', alpha=0.7)
                    plt.scatter(t.cpu().numpy()[0, :], x_obs.cpu().numpy()[0, :, 0], color="none", edgecolor='r', s=20)
                    plt.scatter(t.cpu().numpy()[0, :], x_obs.cpu().numpy()[0, :, 1], color="none", edgecolor='g', s=13)
                else:
                    plt.scatter(t1.cpu().numpy(), x1_obs.cpu().numpy()[:], marker='x', c='#7AC5CD', alpha=0.7)
                    plt.scatter(t2.cpu().numpy(), x2_obs.cpu().numpy()[:], marker='x', c='#C1CDCD', alpha=0.7)
                    plt.scatter(t1.cpu().numpy()[0, :], x1_obs.cpu().numpy()[0, :], color="none", s=20, edgecolor='r')
                    plt.scatter(t2.cpu().numpy()[0, :], x2_obs.cpu().numpy()[0, :], color="none", s=13, edgecolor='g')

                plt.legend()
                plt.savefig(self.folder + "/plot" + str(seed) + '_' + str(blockEpoch)+'_' + str(epoch))
                plt.close()
        print(epoch_loss)
        return epoch_loss, updatedx0

    def train_epoch_RealData(self, data_loader, epoch):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()
            if self.sim == False and self.ts_equal == False:
                t1, t2, t3, t4, t5, t6,t7, x1_obs, x2_obs, x3_obs ,x4_obs,x5_obs,x6_obs,x7_obs= data

                t1 = t1.to(self.device)
                t2 = t2.to(self.device)
                t3 = t3.to(self.device)
                t4 = t4.to(self.device)
                t5 = t5.to(self.device)
                t6 = t6.to(self.device)
                t7 = t7.to(self.device)
                x1_obs = x1_obs.to(self.device)
                x2_obs = x2_obs.to(self.device)
                x3_obs = x3_obs.to(self.device)
                x4_obs = x4_obs.to(self.device)
                x5_obs = x5_obs.to(self.device)
                x6_obs = x6_obs.to(self.device)
                x1_true = x1_obs.to(self.device)
                x2_true = x2_obs.to(self.device)
                x3_true = x3_obs.to(self.device)
                x4_true = x4_obs.to(self.device)
                x5_true = x5_obs.to(self.device)
                x6_true = x6_obs.to(self.device)
                x7_true = x7_obs.to(self.device)
                sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
                sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)
                sort_t3, sort_x3_obs, sort_x3_true = od(t3, x3_obs, x3_true)
                sort_t4, sort_x4_obs, sort_x4_true = od(t4, x4_obs, x4_true)
                sort_t5, sort_x5_obs, sort_x5_true = od(t5, x5_obs, x5_true)
                sort_t6, sort_x6_obs, sort_x6_true = od(t6, x6_obs, x6_true)
                sort_t7, sort_x7_obs, sort_x7_true = od(t7, x7_obs, x7_true)

                mint1 = math.floor(min(sort_t1)) + 1
                mint2 = math.floor(min(sort_t2)) + 1
                mint3 = math.floor(min(sort_t3)) + 1
                mint4 = math.floor(min(sort_t4)) + 1
                mint5 = math.floor(min(sort_t5)) + 1
                mint6 = math.floor(min(sort_t6)) + 1
                mint7 = math.floor(min(sort_t7)) + 1
                x0 = torch.tensor([torch.mean(sort_x1_obs[sort_t1<mint1]),
                                   torch.mean(sort_x2_obs[sort_t2<mint2]),
                                   torch.mean(sort_x3_obs[sort_t3<mint3]),
                                   torch.mean(sort_x4_obs[sort_t4<mint4]),
                                   torch.mean(sort_x5_obs[sort_t5<mint5]),
                                   torch.mean(sort_x6_obs[sort_t6<mint6]),
                                   torch.mean(sort_x7_obs[sort_t7<mint7])]).to(self.device)
                # x0 = torch.tensor([torch.mean(sort_x1_obs[0]), torch.mean(sort_x2_obs[0])]).to(self.device)

                # sort_t12=sort_t12.to(self.device)
                sort_t1 = sort_t1.to(self.device)
                sort_t2 = sort_t2.to(self.device)
                sort_t3 = sort_t3.to(self.device)
                sort_t4 = sort_t4.to(self.device)
                sort_t5 = sort_t5.to(self.device)
                sort_t6 = sort_t6.to(self.device)
                sort_t7 = sort_t7.to(self.device)

                sort_x1_obs = sort_x1_obs.to(self.device)
                sort_x2_obs = sort_x2_obs.to(self.device)
                sort_x3_obs = sort_x3_obs.to(self.device)
                sort_x4_obs = sort_x4_obs.to(self.device)
                sort_x5_obs = sort_x5_obs.to(self.device)
                sort_x6_obs = sort_x6_obs.to(self.device)
                sort_x7_obs = sort_x7_obs.to(self.device)

                pred_x1 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t1).to(self.device)[:, 0], 1)
                pred_x2 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t2).to(self.device)[:, 1], 1)
                pred_x3 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t3).to(self.device)[:, 2], 1)
                pred_x4 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t4).to(self.device)[:, 3], 1)
                pred_x5 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t5).to(self.device)[:, 4], 1)
                pred_x6 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t6).to(self.device)[:, 5], 1)
                pred_x7 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t7).to(self.device)[:, 6], 1)

                loss_x1 = torch.mean(torch.square(pred_x1 - sort_x1_obs))
                loss_x2 = torch.mean(torch.square(pred_x2 - sort_x2_obs))
                loss_x3 = torch.mean(torch.square(pred_x3 - sort_x3_obs))
                loss_x4 = torch.mean(torch.square(pred_x4 - sort_x4_obs))
                loss_x5 = torch.mean(torch.square(pred_x5 - sort_x5_obs))
                loss_x6 = torch.mean(torch.square(pred_x6 - sort_x6_obs))
                loss_x7 = torch.mean(torch.square(pred_x7 - sort_x7_obs))

                loss = torch.mean(torch.stack([loss_x1, loss_x2, loss_x3, loss_x4, loss_x5, loss_x6, loss_x7]))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()
            # plot checking
            if epoch % 1000 == 0:
                minT = torch.min(torch.hstack((sort_t1, sort_t2, sort_t3, sort_t4, sort_t5, sort_t6, sort_t7)))
                maxT = torch.max(torch.hstack((sort_t1, sort_t2, sort_t3, sort_t4, sort_t5, sort_t6, sort_t7)))
                print(minT, maxT)
                Xfull = torch.tensor(np.linspace(minT, maxT, 100))
                # y0true = trueYfull[0, 0, :]
                pred_full = odeint(self.ODEFunc, x0, Xfull[:])
                np.save(osp.join(self.folder, 'timeX_' + '.npy'), Xfull.detach().numpy(), allow_pickle=True)
                np.save(osp.join(self.folder, 'predX_' + str(epoch) + '.npy'), pred_full.detach().numpy(),
                        allow_pickle=True)

                plt.figure()
                # full prediction curve
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 0], label="Amyloid fitted",
                         c='#FF8000')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 1], label="Total Tau fitted",
                         c='#27408B')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 2], label="ADAS13", c='red')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 3], label="CSF TAU", c='black')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 4], label="CSF PTAU", c='green')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 5], label="FDG", c='blue')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 6], label="AV45", c='pink')

                plt.xlabel("Years From Onset")
                plt.ylabel("Linear Transformed Values")

                plt.legend()
                plt.savefig(self.folder + "/AdniPlot" + '_' + str(epoch))
        print(epoch_loss)
        return epoch_loss

    def train_epoch_RealData_TrainInit(self, data_loader, blockEpoch, epoch, param, updatedx0):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            if self.sim == False and self.ts_equal == False:
                t1, t2, t3, t4, t5, t6,t7, x1_obs, x2_obs, x3_obs ,x4_obs,x5_obs,x6_obs,x7_obs=data

                t1 = t1.to(self.device)
                t2 = t2.to(self.device)
                t3 = t3.to(self.device)
                t4 = t4.to(self.device)
                t5 = t5.to(self.device)
                t6 = t6.to(self.device)
                t7 = t7.to(self.device)
                x1_obs = x1_obs.to(self.device)
                x2_obs = x2_obs.to(self.device)
                x3_obs = x3_obs.to(self.device)
                x4_obs = x4_obs.to(self.device)
                x5_obs = x5_obs.to(self.device)
                x6_obs = x6_obs.to(self.device)
                x1_true = x1_obs.to(self.device)
                x2_true = x2_obs.to(self.device)
                x3_true = x3_obs.to(self.device)
                x4_true = x4_obs.to(self.device)
                x5_true = x5_obs.to(self.device)
                x6_true = x6_obs.to(self.device)
                x7_true = x7_obs.to(self.device)
                sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
                sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)
                sort_t3, sort_x3_obs, sort_x3_true = od(t3, x3_obs, x3_true)
                sort_t4, sort_x4_obs, sort_x4_true = od(t4, x4_obs, x4_true)
                sort_t5, sort_x5_obs, sort_x5_true = od(t5, x5_obs, x5_true)
                sort_t6, sort_x6_obs, sort_x6_true = od(t6, x6_obs, x6_true)
                sort_t7, sort_x7_obs, sort_x7_true = od(t7, x7_obs, x7_true)

                mint1 = math.floor(min(sort_t1)) + 1
                mint2 = math.floor(min(sort_t2)) + 1
                mint3 = math.floor(min(sort_t3)) + 1
                mint4 = math.floor(min(sort_t4)) + 1
                mint5 = math.floor(min(sort_t5)) + 1
                mint6 = math.floor(min(sort_t6)) + 1
                mint7 = math.floor(min(sort_t7)) + 1

                if torch.sum(updatedx0) == 0:
                    x0init = torch.tensor([torch.mean(sort_x1_obs[sort_t1 < mint1]),
                                              torch.mean(sort_x2_obs[sort_t2 < mint2]),
                                              torch.mean(sort_x3_obs[sort_t3 < mint3]),
                                              torch.mean(sort_x4_obs[sort_t4 < mint4]),
                                              torch.mean(sort_x5_obs[sort_t5 < mint5]),
                                              torch.mean(sort_x6_obs[sort_t6 < mint6]),
                                              torch.mean(sort_x7_obs[sort_t7 < mint7])])

                    x0init = torch.reshape(x0init, (7, 1))
                    print('x0: {}'.format(str(x0init)))
                    x0func = Initfunc(x0init)
                    x0 = x0func(torch.tensor(1.0).unsqueeze(0))
                    print('initial value:{}'.format(str(x0)))
                else:
                    updatedx0 = torch.reshape(updatedx0,(7, 1))
                    x0func = Initfunc(updatedx0)
                    x0 = x0func(torch.tensor(1.0).unsqueeze(0))
                    print('initial value:{}'.format(str(x0)))

                # sort_t12=sort_t12.to(self.device)
                sort_t1 = sort_t1.to(self.device)
                sort_t2 = sort_t2.to(self.device)
                sort_t3 = sort_t3.to(self.device)
                sort_t4 = sort_t4.to(self.device)
                sort_t5 = sort_t5.to(self.device)
                sort_t6 = sort_t6.to(self.device)
                sort_t7 = sort_t7.to(self.device)

                sort_x1_obs = sort_x1_obs.to(self.device)
                sort_x2_obs = sort_x2_obs.to(self.device)
                sort_x3_obs = sort_x3_obs.to(self.device)
                sort_x4_obs = sort_x4_obs.to(self.device)
                sort_x5_obs = sort_x5_obs.to(self.device)
                sort_x6_obs = sort_x6_obs.to(self.device)
                sort_x7_obs = sort_x7_obs.to(self.device)

                pred_x1 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t1).to(self.device)[:, 0], 1)
                pred_x2 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t2).to(self.device)[:, 1], 1)
                pred_x3 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t3).to(self.device)[:, 2], 1)
                pred_x4 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t4).to(self.device)[:, 3], 1)
                pred_x5 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t5).to(self.device)[:, 4], 1)
                pred_x6 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t6).to(self.device)[:, 5], 1)
                pred_x7 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t7).to(self.device)[:, 6], 1)


                # delta_t1 = 1/np.diff(sort_t1)
                # delta_t1 = torch.tensor(np.hstack([1.0, delta_t1]),dtype=torch.float32 )
                # delta_t2 = 1/np.diff(sort_t2)
                # delta_t2 = torch.tensor(np.hstack([1.0, delta_t2]),dtype=torch.float32 )
                # delta_t3 = 1/np.diff(sort_t3)
                # delta_t3 = torch.tensor(np.hstack([1.0, delta_t3]),dtype=torch.float32 )
                # delta_t4 = 1/np.diff(sort_t4)
                # delta_t4 = torch.tensor(np.hstack([1.0, delta_t4]),dtype=torch.float32 )
                # delta_t5 = 1/np.diff(sort_t5)
                # delta_t5 = torch.tensor(np.hstack([1.0, delta_t5]),dtype=torch.float32 )
                # delta_t6 = 1/np.diff(sort_t6)
                # delta_t6 = torch.tensor(np.hstack([1.0, delta_t6]),dtype=torch.float32 )
                # delta_t7 = 1/np.diff(sort_t7)
                # delta_t7 = torch.tensor(np.hstack([1.0, delta_t7]),dtype=torch.float32 )

                # print("pred_x1 size:{},\n obs_x1 size:{},\n delta_t1 size:{}".format(pred_x1.size(),sort_x1_obs.size(),delta_t1.size()))
                # print("square size:{}".format(torch.square(pred_x1 - sort_x1_obs).size()))
                # loss_x1 = torch.matmul(delta_t1,torch.square(pred_x1 - sort_x1_obs))/pred_x1.size(0)
                # loss_x2 = torch.matmul(delta_t2,torch.square(pred_x2 - sort_x2_obs))/pred_x2.size(0)
                # loss_x3 = torch.matmul(delta_t3,torch.square(pred_x3 - sort_x3_obs))/pred_x3.size(0)
                # loss_x4 = torch.matmul(delta_t4,torch.square(pred_x4 - sort_x4_obs))/pred_x4.size(0)
                # loss_x5 = torch.matmul(delta_t5,torch.square(pred_x5 - sort_x5_obs))/pred_x5.size(0)
                # loss_x6 = torch.matmul(delta_t6,torch.square(pred_x6 - sort_x6_obs))/pred_x6.size(0)
                # loss_x7 = torch.matmul(delta_t7,torch.square(pred_x7 - sort_x7_obs))/pred_x7.size(0)

                loss_x1 = torch.mean(torch.square(pred_x1 - sort_x1_obs))
                loss_x2 = torch.mean(torch.square(pred_x2 - sort_x2_obs))
                loss_x3 = torch.mean(torch.square(pred_x3 - sort_x3_obs))
                loss_x4 = torch.mean(torch.square(pred_x4 - sort_x4_obs))
                loss_x5 = torch.mean(torch.square(pred_x5 - sort_x5_obs))
                loss_x6 = torch.mean(torch.square(pred_x6 - sort_x6_obs))
                loss_x7 = torch.mean(torch.square(pred_x7 - sort_x7_obs))

                loss = torch.mean(torch.stack([loss_x1, loss_x2, loss_x3, loss_x4, loss_x5, loss_x6, loss_x7]))
                if param == 'theta':
                    params = (list(self.ODEFunc.parameters()))
                    # params = (list(x0func.parameters()))
                    optimizer = torch.optim.Adam(params, lr=self.lr)
                    optimizer.zero_grad()
                    l1_lambda = 0.001
                    l1_norm = sum(torch.linalg.norm(p, 1) for p in params)
                    loss = loss + l1_lambda * l1_norm

                elif param == 'init':
                    params = (list(x0func.parameters()))
                    optimizer = torch.optim.Adam(params,lr=0.001)
                    optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                updatedx0 = x0func(torch.tensor(1.0).unsqueeze(0))
                print('updated initial value:{}'.format(str(updatedx0)))
                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()
            # plot checking
            if epoch % 299 ==0:
                # full time points
                minT = torch.min(torch.hstack((sort_t1, sort_t2, sort_t3,sort_t4,sort_t5,sort_t6, sort_t7)))
                maxT = torch.max(torch.hstack((sort_t1, sort_t2, sort_t3,sort_t4,sort_t5,sort_t6 ,sort_t7)))
                print(minT, maxT)
                Xfull = torch.tensor(np.linspace(minT, maxT, 100))
                # y0true = trueYfull[0, 0, :]
                pred_full = odeint(self.ODEFunc, updatedx0, Xfull[:])
                np.save(osp.join(self.folder, 'timeX_' + '.npy'), Xfull.detach().numpy(), allow_pickle=True)
                np.save(osp.join(self.folder, 'predX_' + str(blockEpoch)+'_'+ str(epoch) + '.npy'), pred_full.detach().numpy(), allow_pickle=True)

                plt.figure()
                # full prediction curve
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 0], label="Amyloid fitted", c='#FF8000')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 1], label="Total Tau fitted", c='#27408B')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 2], label="ADAS13", c='red')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 3], label="CSF TAU", c='black')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 4], label="CSF PTAU", c='green')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 5], label="FDG", c='blue')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 6], label="AV45", c='pink')

                # observed data with noise
                # plt.scatter(t1.cpu().numpy(), x1_obs.cpu().numpy()[:], marker='x', c='#8B5A2B', alpha=0.6,
                #             label='Amyloid Observed')
                # plt.scatter(t2.cpu().numpy(), x2_obs.cpu().numpy()[:], marker='x', c='#9FB6CD', alpha=0.6,
                #             label='Total Tau Observed')
                # plt.scatter(t3.cpu().numpy(), x3_obs.cpu().numpy()[:], marker='x', c='red', alpha=0.6,
                # #             label='ADAS13')
                # plt.scatter(t1.cpu().numpy()[0, :], x1_obs.cpu().numpy()[0, :], color="none", s=20,
                #             edgecolor='r')
                # plt.scatter(t2.cpu().numpy()[0, :], x2_obs.cpu().numpy()[0, :], color="none", s=13,
                #             edgecolor='g')
                plt.xlabel("Years From Onset")
                plt.ylabel("Linear Transformed Values")

                plt.legend()
                plt.savefig(self.folder + "/AdniPlot" + '_' + str(blockEpoch)+'_'+ str(epoch))
                # plt.close()
        print(epoch_loss)
        return epoch_loss,updatedx0
