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
from joblib import Parallel, delayed
import multiprocessing


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
                 optimizer: torch.optim.Optimizer,
                 folder,
                 seed,
                 ifplot
                 ):
        self.sim = sim
        self.device = device
        self.optimizer = optimizer
        self.ts_equal = ts_equal
        self.ODEFunc = ODEFunc
        self.epoch_loss_history = []
        self.folder = folder
        self.seed = seed
        self.ifplot = ifplot

        np.random.seed(self.seed)

    def train(self, train_data_loader: DataLoader, epochs: int, seed):
        """
        Train Neural ODE.

        Parameters
        ----------
        DataLoader:
            Dataloader
        epochs: int
            Number of epochs to train for
        """
        # self.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            print(f'Seed {seed}')
            epoch_start_time = time.time()
            if self.sim == True:
                epoch_loss = self.train_epoch(train_data_loader, epoch, seed)
            else:
                epoch_loss = self.train_epoch_RealData(train_data_loader, epoch)
            self.epoch_loss_history.append(epoch_loss)
            epoch_end_time = time.time()
            print('Each Epoch time = ' + str(epoch_end_time - epoch_start_time))

    def train_epoch(self, data_loader, epoch, seed):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()

            # Extract data
            if self.sim == True and self.ts_equal == True:
                t, x_obs, x_true = data[:][1]
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
                loss = torch.mean(torch.square(pred_x - sort_x_obs))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()
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

                pred_x2 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t2).to(self.device)[:, 1], 1)

                loss_x1 = torch.mean(torch.square(pred_x1 - sort_x1_obs))
                loss_x2 = torch.mean(torch.square(pred_x2 - sort_x2_obs))
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
        print(epoch_loss)
        return epoch_loss

    def train_epoch_RealData(self, data_loader, epoch):
        epoch_loss = 0.
        # self.train()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(data_loader):
            self.optimizer.zero_grad()
            if self.sim == False and self.ts_equal == False:
                t1, t2, x1_obs, x2_obs, x1_true, x2_true = data

                t1 = t1.to(self.device)
                t2 = t2.to(self.device)

                x1_obs = x1_obs.to(self.device)
                x2_obs = x2_obs.to(self.device)

                x1_true = x1_true.to(self.device)
                x2_true = x2_true.to(self.device)

                sort_t1, sort_x1_obs, sort_x1_true = od(t1, x1_obs, x1_true)
                sort_t2, sort_x2_obs, sort_x2_true = od(t2, x2_obs, x2_true)

                mint1=math.floor(min(sort_t1))+1
                mint2=math.floor(min(sort_t2))+1
                x0 = torch.tensor([torch.mean(sort_x1_obs[sort_t1<mint1]),torch.mean(sort_x2_obs[sort_t2<mint2])]).to(self.device)
                # x0 = torch.tensor([torch.mean(sort_x1_obs[0]), torch.mean(sort_x2_obs[0])]).to(self.device)

                # sort_t12=sort_t12.to(self.device)
                sort_t1 = sort_t1.to(self.device)
                sort_t2 = sort_t2.to(self.device)
                sort_x1_obs = sort_x1_obs.to(self.device)
                sort_x2_obs = sort_x2_obs.to(self.device)

                pred_x1 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t1).to(self.device)[:, 0], 1)
                pred_x2 = torch.unsqueeze(odeint(self.ODEFunc, x0, sort_t2).to(self.device)[:, 1], 1)

                loss_x1 = torch.mean(torch.square(pred_x1 - sort_x1_obs))
                loss_x2 = torch.mean(torch.square(pred_x2 - sort_x2_obs))
                loss = torch.mean(torch.stack([loss_x1, loss_x2]))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()
            # plot checking
            if epoch % 1000 == 0:
                # full time points
                minT = torch.min(torch.hstack((sort_t1, sort_t2)))
                maxT = torch.max(torch.hstack((sort_t1, sort_t2)))
                Xfull = torch.tensor(np.linspace(minT, maxT, 100))
                # y0true = trueYfull[0, 0, :]
                pred_full = odeint(self.ODEFunc, x0, Xfull[:])
                np.save(osp.join(self.folder, 'timeX_' + '.npy'), Xfull.detach().numpy(), allow_pickle=True)
                np.save(osp.join(self.folder, 'predX_' + str(epoch) + '.npy'), pred_full.detach().numpy(), allow_pickle=True)

                plt.figure()
                # full prediction curve
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 0], label="Amyloid fitted",
                         c='#FF8000')
                plt.plot(Xfull.cpu().numpy(), pred_full.cpu().detach().numpy()[:, 1], label="Total Tau fitted", c='#27408B')
                # observed data with noise
                plt.scatter(t1.cpu().numpy(), x1_obs.cpu().numpy()[:], marker='x', c='#8B5A2B', alpha=0.6,
                            label='Amyloid Observed')
                plt.scatter(t2.cpu().numpy(), x2_obs.cpu().numpy()[:], marker='x', c='#9FB6CD', alpha=0.6,
                            label='Total Tau Observed')
                # plt.scatter(t1.cpu().numpy()[0, :], x1_obs.cpu().numpy()[0, :], color="none", s=20,
                #             edgecolor='r')
                # plt.scatter(t2.cpu().numpy()[0, :], x2_obs.cpu().numpy()[0, :], color="none", s=13,
                #             edgecolor='g')
                plt.xlabel("Years From Onset")
                plt.ylabel("Linear Transformed Values")

                plt.legend()
                plt.savefig(self.folder + "/AdniPlot" + '_' + str(epoch))
        print(epoch_loss)
        return epoch_loss
