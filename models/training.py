import torch

from tqdm import tqdm
from typing import Tuple
from random import randint
from models.utils import ObservedData as od
from torch.utils.data import DataLoader
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import time

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
                 device: torch.device,
                 ODEFunc,
                 optimizer: torch.optim.Optimizer,
                 folder,
                 seed
                 ):
        self.device = device
        self.optimizer = optimizer
        self.sim = True
        self.ODEFunc = ODEFunc
        self.epoch_loss_history = []
        self.folder = folder
        self.seed = seed

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
            epoch_loss = self.train_epoch(train_data_loader, epoch,seed)
            self.epoch_loss_history.append(epoch_loss)
            epoch_end_time = time.time()
            print('Each Epoch time = ' + str(epoch_end_time - epoch_start_time))

    def train_epoch(self, data_loader, epoch,seed):
        epoch_loss = 0.
        # self.train()
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()

            # Extract data
            if self.sim:
                x, yobs, ytrue = data[:][1]
                x=x.to(self.device)
                yobs=yobs.to(self.device)
                ytrue=ytrue.to(self.device)
                sortX, sortYobs, sortYtrue = od(x, yobs, ytrue)
                y0 = sortYobs[0].to(self.device)
                sortX=sortX.to(self.device)
                sortYobs=sortYobs.to(self.device)
                sortYtrue=sortYtrue.to(self.device)
                pred_y = odeint(self.ODEFunc, y0, sortX).to(self.device)

                loss = torch.mean(torch.square(pred_y - sortYobs))
                loss.backward()
                self.optimizer.step()

                # mse = ((sortYobs - pred_y.mean)**2).mean()
                epoch_loss += loss.cpu().item()
            # plot checking
            if epoch % 500 == 0:
                # full time points
                Xfull, trueYfull = data[:][0]
                Xfull=Xfull.to(self.device)
                trueYfull=trueYfull.to(self.device)
                y0true = trueYfull[0, 0, :]
                pred_full = odeint(self.ODEFunc, y0true, Xfull[0, :])

                plt.figure()
                # full true curve
                plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 0], label="Rabbit")
                plt.plot(Xfull.cpu().numpy()[0, :], trueYfull.cpu().numpy()[1, :, 1], label='Fox')
                # full prediction curve
                plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 0], label="Rabbit_pred")
                plt.plot(Xfull.cpu().numpy()[0, :], pred_full.cpu().detach().numpy()[:, 1], label="Fox_pred")
                # observed data with noise
                plt.scatter(sortX.cpu().numpy(), sortYobs.cpu().numpy()[:, 0], marker='x', c='#7AC5CD')
                plt.scatter(sortX.cpu().numpy(), sortYobs.cpu().numpy()[:, 1], marker='x', c='#C1CDCD')

                plt.legend()
                plt.savefig(self.folder + "/plot" +str(seed)+'_'+ str(epoch))
            #
        print(epoch_loss)
        return epoch_loss
