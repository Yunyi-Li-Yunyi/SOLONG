import torch
import os.path as osp
import os
import numpy as np
import pandas as pd
import hickle as hkl
import matplotlib.pyplot as plt
from math import pi
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from six.moves import urllib
from scipy.io import loadmat
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from scipy.integrate import odeint
# from torchdiffeq import odeint
from scipy import ndimage
import scipy.spatial.distance

class DeterministicLotkaVolterraData(Dataset):
    """
    Dataset of Lotka-Volterra time series.
        Populations (u,v) evolve according to

            u' = u*(1-u) -\alpha * u*v/(u+c)
            v' = \beta*v*(1-(v/u))

        with the dataset sampled with varying duration of follow-up time t{i,k} ~ exp(\lamda)
        for a fixed set of greeks.
        X1i(t) = X1(t)+vi+\epsilon(t)
        X2i(t) = X2(t)+ui+\eta(t)
        if x1x2Ind==True:
            vi ~ N(0, st_v^2)
            ui ~ N(0, st_u^2)
        else:
            (u,v) ~ BivariateNormal([0,0],[0,\rho,\rho,0])

        epsilon, eta ~ N(0, sd) for any t

    ---------
    fixed_alpha : int
        fixed initial value for \alpha
    fixed_beta  : int
        fixed initial value for \beta
    fixed_gamma : int
        fixed initial value for \gamme
    num_samples : int
        Number of samples of the function contained in dataset.
    lambdaX : float
        the duration of follow up ~ exp(\lambdaX)
    sdense: array
        the time points for dense data
    sd : float
        the standard deviation of random normal noise
    sd_v: float
        the standard deviation of random normal noise for X1
    sd_u: float
        the standard deviation of random normal noise for X2
    rho: float
        the covariance between x1 and x2 if x1x2Ind==False
    num_context_range : Tuple[int,int]
        The range of number of context points.

    lambdaY: float
        the duration of follow up ~ exp(\lambdaY)
    sd_y: float
        the standard deviation of random normal noise
    """

    def __init__(self,
                 alpha=None, beta=None, gamma=None,
                 num_samples=1000, lambdaX=1., sdense=None, sd=0., num_context_range=(5, 6),
                 sd_v=0., sd_u=0., rho=0.,scenario=None,
                 lambdaY=1., sd_y=0., num_context_rangeY=(5, 6), seed=0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.num_samples = num_samples
        self.sdense = sdense
        self.lambdaX = lambdaX
        self.sd_v = sd_v
        self.sd_u = sd_u
        self.sd = sd
        self.rho = rho
        self.scenario=scenario
        self.num_context_range = num_context_range

        self.lambdaY = lambdaY
        self.sd_y = sd_y
        self.num_context_rangeY = num_context_rangeY

        self.seed = seed

        # Generate data
        self.data = []
        self.SparseData = []
        self.OutcomeData = []
        print("Creating dataset...", flush=True)

        np.random.seed(self.seed)

        removed = 0
        for samples in tqdm(range(self.num_samples)):
            # generate x, states_obs, states_true
            # when sd = 0. there's no observation noise, states_obs = states_true
            times, states_true = self.generate_ts(dense=True)
            times = torch.FloatTensor(times)
            states_true = torch.FloatTensor(states_true)

            # states = torch.cat((states,times),dim=-1)
            self.data.append((times, states_true))

            # Generate Outcome
            self.OutcomeData.append(self.Outcome(np.array(times), states_true))
            timseSparse, states_obsSparse, states_trueSparse = self.generate_ts(dense=False)
            timseSparse = torch.FloatTensor(timseSparse)
            states_obsSparse = torch.FloatTensor(states_obsSparse)
            states_trueSparse = torch.FloatTensor(states_trueSparse)

            self.SparseData.append((timseSparse, states_obsSparse, states_trueSparse))

        self.num_samples -= removed

    def generate_ts(self, dense):
        equal_pop = np.random.uniform(1., 1.)
        X_0 = np.array([equal_pop, equal_pop])
        a, b, c = self.alpha, self.beta, self.gamma

        def dX_dt(X, s=0):
            """
            Return the growth rate of fox and rabbit populations
            """
            return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1] / (X[0] + c)),
                             b * X[1] * (1 - X[1] / X[0])])
        def error_cov(sizeX,scenario):
            """
            Define the variance-covariance matrix to generate errors
            sim A:
            self.sd_v != None
            self.sd_u != None
            self.rho = None

            sim B:
            self.sd_v != None
            self.sd_u != None
            self.rho = 0

            sim B2:
            self.sd_v = None
            self.sd_u = None

            sim C:
            self.sd_v !=None
            self.sd_u !=None
            self.rho !=0
            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario=='simA':
                # print("Sim A")
                sigma11=self.sd_v**2*np.identity(sizeX)
                sigma22=self.sd_u**2*np.identity(sizeX)
                sigma12=np.zeros((sizeX,sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B:
            elif scenario=='simB':
                print("Sim B")
                sigma11=self.sd_v**2*np.ones((sizeX,sizeX))
                sigma22=self.sd_u**2*np.ones((sizeX,sizeX))
                sigma12=np.zeros((sizeX,sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B2a:
            elif scenario=='simB2a':
                # print("Sim B2a")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X = np.expand_dims(s, 1)
                sigma11=exponentiated_quadratic(X, X)
                sigma22=exponentiated_quadratic(X, X)
                sigma12=np.zeros((sizeX,sizeX))
                sigma21 =np.zeros((sizeX, sizeX))

            # sim C:
            elif scenario=='simC':
                # print("Sim C")
                sigma11=self.sd_v**2*np.ones((sizeX,sizeX))
                sigma22=self.sd_u**2*np.ones((sizeX,sizeX))
                sigma12=self.rho*self.sd_v*self.sd_u*np.identity(sizeX)
                sigma21=self.rho*self.sd_v*self.sd_u*np.identity(sizeX)

            # sim B2b:
            elif scenario=='simB2b':
                # print("Sim B2b")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X = np.expand_dims(s, 1)
                sigma11=exponentiated_quadratic(X, X)
                sigma22=exponentiated_quadratic(X, X)
                sigma12=self.rho*self.sd_v*self.sd_u*np.identity(sizeX)
                sigma21=self.rho*self.sd_v*self.sd_u*np.identity(sizeX)

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            return sigma

        sizeX = np.random.choice(range(*self.num_context_range), size=1, replace=True)
        if dense == True:
            s = self.sdense
        else:
            s = np.random.exponential(self.lambdaX, sizeX)
            s[0] = 0
            for sik in range(1, len(s)):
                s[sik] = round(s[sik - 1] + s[sik],2)

        X_true = odeint(dX_dt, X_0, s)

        if dense == True:
            return s, X_true
        else:
            sigmaMatrix = error_cov(sizeX=len(X_true),scenario=self.scenario)
            meanVector = np.zeros(2 * len(X_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).reshape((2,-1)).T
            X_obs = X_true + error

            return s, X_obs, X_true

    def Outcome(self, s, X_true):
        a, b, c = self.alpha, self.beta, self.gamma
        sizeY = np.random.choice(range(*self.num_context_rangeY), size=1, replace=True)
        t = np.random.exponential(self.lambdaY, sizeY)
        t[0] = 0
        for tik in range(1, len(t)):
            t[tik] = round(t[tik - 1] + t[tik],2)

        def Beta0(t):
            b0 = 2 * np.exp(-(t - 2.5) ** 2)
            return torch.FloatTensor(np.array(b0))

        def Beta1(t, s):
            b1 = []
            for si in s:
                b1.append(np.cos(t * np.pi / 3) * np.sin(si * np.pi / 5))
            return torch.FloatTensor(np.array(b1))

        def Beta2(t, s):
            b2 = []
            for si in s:
                b2.append(np.sqrt(t * si) / 4.2)
            return torch.FloatTensor(np.array(b2))

        def dX_dt_tensor(X, s=0):
            """
            Return the growth rate of fox and rabbit populations
            """
            return [X[:, 0] * (1 - X[:, 0]) - (a * X[:, 0] * X[:, 1] / (X[:, 0] + c)),
                    b * X[:, 1] * (1 - X[:, 1] / X[:, 0])]

        by = s[1] - s[0]  # the interval length aka delta_X
        lx1 = by * dX_dt_tensor(X_true)[0]
        lx2 = by * dX_dt_tensor(X_true)[1]

        beta0 = Beta0(t)
        beta1 = Beta1(t, s)
        beta2 = Beta2(t, s)

        eta = torch.matmul(lx1, beta1) + torch.matmul(lx2, beta2)
        outcome = beta0 + eta
        outcome = outcome + np.random.normal(0, self.sd_y, outcome.shape)
        return outcome

    def __getitem__(self, index):
        return self.data[index], self.SparseData[index], self.OutcomeData[index]

    def __len__(self):
        return self.num_samples


# if __name__ == "__main__":
#     sdense = np.linspace(0, 15, 100)
# 
#     datasets = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10, sdense=sdense,
#                                               num_samples=5, sd_u=0.3, sd_v=0.3,rho=0.9, scenario='simA', sd=0., sd_y=0.)
# #     # print(datasets[0][1])
#     for i in range(5):
#         timeS, x_obsS, x_trueS = datasets[i][1]
#         time,x_true = datasets[i][0]
#         plt.plot(time.numpy(), x_true.numpy()[:, 0])
#         plt.plot(time.numpy(), x_true.numpy()[:, 1])
#         plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 0])
#         plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 1])
#     plt.show()
