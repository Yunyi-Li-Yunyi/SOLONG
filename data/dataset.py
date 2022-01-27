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


#
# class DeterministicLotkaVolterraData(Dataset):
#     """
#     Dataset of Lotka-Voltera time series.
#         Populations (u,v) evolve according to
#         version 1
#             u' = \alpha u - \beta u v
#             v' = \delta uv - \gamma v
#
#         version 2
#             u' = u*(1-u) -\alpha * u*v/(u+c)
#             v' = \beta*v*(1-(v/u))
#
#         with the dataset sampled either with (u_0,v_0) fixed and (\alpha, \beta,
#         \gamma, \delta) varied, or varying the initial populations for a fixed
#         set of greeks.
#     If initial values for (u,v) are provided then the greeks are sampled from
#         (0.9,0.05,1.25,0.5) to (1.1,0.15,1.75,1.0)
#     If values are provided for the greeks then (u_0 = v_0) is sampled from
#         (0.5) to (2.0)
#     If both are provided, defaults to initial population mode (greeks vary)
#     ---------
#     version : int
#     initial_u   : int
#     fixed initial value for u
#     initial_v   : int
#         fixed initial value for v
#     fixed_alpha : int
#         fixed initial value for \alpha
#     fixed_beta  : int
#         fixed initial value for \beta
#     fixed_gamma : int
#         fixed initial value for \gamme
#     fixed_delta : int
#         fixed initial value for \delta
#     num_samples : int
#         Number of samples of the function contained in dataset.
#     steps : int
#         how many time steps to take from 0 to end_time
#     end_time : float
#         the final time (simulation runs from 0 to end_time)
#     sd : float
#         the standard deviation of random normal noise
#     num_context_range : Tuple[int,int]
#         The range of number of context points.
#
#     end_time_y: float
#         the final time of outcome(simulation runs from 0 to end_time_y)
#     steps_y: int
#             how many time steps to take from 0 to end_time_y
#     sd_y: float
#         the standard deviation of random normal noise
#     """
#
#     def __init__(self, version=1, initial_u=None, initial_v=None,
#                  alpha=None, beta=None, gamma=None, delta=None,
#                  num_samples=1000, steps=100, end_time=15, sd=0., num_context_range=(5, 6),
#                  x1x2Ind=True, sd_v=0., sd_u=0., rho=0.,
#                  end_time_y=10, steps_y=4, sd_y=0.,seed=0):
#
#         if initial_u is None:
#             self.mode = 'greek'
#             self.alpha = alpha
#             self.beta = beta
#             self.gamma = gamma
#             self.delta = delta
#         else:
#             self.mode = 'population'
#             self.initial_u = initial_u
#             self.initial_v = initial_v
#
#         print('Lotka-Voltera is in {self.mode} mode')
#
#         self.version = version
#         self.num_samples = num_samples
#         self.steps = steps
#         self.end_time = end_time
#         self.sd = sd
#         self.sd_v = sd_v
#         self.sd_u = sd_u
#         self.rho = rho
#         self.end_time_y = end_time_y
#         self.steps_y = steps_y
#         self.sd_y = sd_y
#         self.x1x2Ind = x1x2Ind
#         self.seed = seed
#
#         # Generate data
#         self.data = []
#         self.SparseData = []
#         self.OutcomeData = []
#         print("Creating dataset...", flush=True)
#
#         np.random.seed(self.seed)
#
#         removed = 0
#         for samples in tqdm(range(num_samples)):
#             # generate x, states_obs, states_true
#             # when sd = 0. there's no observation noise, states_obs = states_true
#             times, states_obs, states_true = self.generate_ts()
#             times = torch.FloatTensor(times)
#             # times = times.unsqueeze(1)
#             states_obs = torch.FloatTensor(states_obs)
#             states_true = torch.FloatTensor(states_true)
#
#             if self.mode == 'population':
#                 states_obs = states_obs / 100
#                 states_true = states_true / 100
#
#             # states = torch.cat((states,times),dim=-1)
#
#             self.data.append((times, states_obs, states_true))
#
#             # Generate Outcome
#             self.OutcomeData.append(self.Outcome(states_true))
#
#             # Generate Sparse Data
#             points = np.arange(self.steps)
#             initial_loc = np.array([0])
#
#             size = np.random.choice(range(*num_context_range), size=1, replace=True)
#             locations = np.random.choice(points, size=self.steps - size, replace=False)
#             locations = np.concatenate([initial_loc, locations])
#             timseSparse = torch.clone(times)
#             timseSparse[locations] = float('nan')
#             states_obsSparse = torch.clone(states_obs)
#             states_obsSparse[locations] = float('nan')
#             states_trueSparse = torch.clone(states_true)
#             states_trueSparse[locations] = float('nan')
#
#             self.SparseData.append((timseSparse, states_obsSparse, states_trueSparse))
#
#         self.num_samples -= removed
#
#     def generate_ts(self):
#         if self.mode == 'population':
#             X_0 = np.array([self.initial_u, self.initial_v])
#             a = np.random.uniform(0.9, 1.1)
#             b = np.random.uniform(0.05, 0.15)
#             c = np.random.uniform(1.25, 1.75)
#             d = np.random.uniform(0.5, 1.0)
#
#         else:
#             equal_pop = np.random.uniform(1., 1.)
#             if self.version == 1:
#                 X_0 = np.array([2 * equal_pop, equal_pop])
#             if self.version == 2:
#                 X_0 = np.array([equal_pop, equal_pop])
#             a, b, c, d = self.alpha, self.beta, self.gamma, self.delta
#
#         def dX_dt(X, s=0):
#             """
#             Return the growth rate of fox and rabbit populations
#             """
#             if self.version == 1:
#                 return np.array([a * X[0] - b * X[0] * X[0],
#                                  -c * X[1] + d * X[0] * X[1]])
#             if self.version == 2:
#                 return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1] / (X[0] + c)),
#                                  b * X[1] * (1 - X[1] / X[0])])
#
#         s = np.linspace(0, self.end_time, self.steps)
#
#         X_true = odeint(dX_dt, X_0, s)
#         if self.x1x2Ind == True:
#             uv = np.zeros_like(X_true)
#
#             # vi: random errors for X1i
#             vi = np.random.normal(0, self.sd_v, 1)
#             # ui: random errors for X2i
#             ui = np.random.normal(0, self.sd_u, 1)
#
#             uv[:, 0] = vi
#             uv[:, 1] = ui
#
#         # correlation within subject and correlation between X1(t), X2(t)
#         else:
#             uv = np.random.multivariate_normal([0,0],[[1,self.rho],[self.rho,1]],X_true.shape[0])
#
#         X_obs = X_true + uv + np.random.normal(0, self.sd, X_true.shape)
#
#         return s, X_obs, X_true
#
#     def Outcome(self, X_true):
#         a, b, c, d = self.alpha, self.beta, self.gamma, self.delta
#
#         t = np.linspace(0, self.end_time_y, self.steps_y)
#         s = np.linspace(0, self.end_time, self.steps)
#
#         def Beta0(t):
#             b0 = 2 * np.exp(-(t - 2.5) ** 2)
#             return torch.FloatTensor(b0)
#
#         def Beta1(t, s):
#             b1 = []
#             for si in s:
#                 b1.append(np.cos(t * np.pi / 3) * np.sin(si * np.pi / 5))
#             return torch.FloatTensor(b1)
#
#         def Beta2(t, s):
#             b2 = []
#             for si in s:
#                 b2.append(np.sqrt(t * si) / 4.2)
#             return torch.FloatTensor(b2)
#
#         def dX_dt_tensor(X, s=0):
#             """
#             Return the growth rate of fox and rabbit populations
#             """
#             if self.version == 1:
#                 return [a * X[0] - b * X[0] * X[0],
#                         -c * X[:, 1] + d * X[:, 0] * X[:, 1]]
#             if self.version == 2:
#                 return [X[:, 0] * (1 - X[:, 0]) - (a * X[:, 0] * X[:, 1] / (X[:, 0] + c)),
#                         b * X[:, 1] * (1 - X[:, 1] / X[:, 0])]
#
#         by = s[1] - s[0]  # the interval length aka delta_X
#         lx1 = by * dX_dt_tensor(X_true)[0]
#         lx2 = by * dX_dt_tensor(X_true)[1]
#
#         beta0 = Beta0(t)
#         beta1 = Beta1(t, s)
#         beta2 = Beta2(t, s)
#
#         eta = torch.matmul(lx1, beta1) + torch.matmul(lx2, beta2)
#         outcome = beta0 + eta
#         outcome = outcome + np.random.normal(0, self.sd_y, outcome.shape)
#         return outcome
#
#     def __getitem__(self, index):
#         return self.data[index], self.SparseData[index], self.OutcomeData[index]
#
#     def __len__(self):
#         return self.num_samples


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
    end_time : float
        the final time (simulation runs from 0 to end_time)
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

    end_time_y: float
        the final time of outcome(simulation runs from 0 to end_time_y)
    lambdaY: float
        the duration of follow up ~ exp(\lambdaY)
    sd_y: float
        the standard deviation of random normal noise
    """

    def __init__(self,
                 alpha=None, beta=None, gamma=None,
                 num_samples=1000, lambdaX=1., sdense=None, end_time=10, sd=0., num_context_range=(5, 6),
                 x1x2Ind=True, sd_v=0., sd_u=0., rho=0.,
                 end_time_y=10, lambdaY=1., sd_y=0., num_context_rangeY=(5, 6), seed=0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.num_samples = num_samples
        self.sdense = sdense
        self.lambdaX = lambdaX
        self.end_time = end_time
        self.x1x2Ind = x1x2Ind
        self.sd_v = sd_v
        self.sd_u = sd_u
        self.sd = sd
        self.rho = rho
        self.num_context_range = num_context_range

        self.end_time_y = end_time_y
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

        sizeX = np.random.choice(range(*self.num_context_range), size=1, replace=True)
        if dense == True:
            s = self.sdense
        else:
            s = np.random.exponential(self.lambdaX, sizeX)
            s[0] = 0
            for sik in range(1, len(s)):
                s[sik] = round(s[sik - 1] + s[sik],2)
        X_true = odeint(dX_dt, X_0, s)

        if self.x1x2Ind == True:
            uv = np.zeros_like(X_true)

            # vi: random errors for X1i
            vi = np.random.normal(0, self.sd_v, 1)
            # ui: random errors for X2i
            ui = np.random.normal(0, self.sd_u, 1)

            uv[:, 0] = vi
            uv[:, 1] = ui

        # correlation within subject and correlation between X1(t), X2(t)
        else:
            uv = np.zeros_like(X_true)
            uv_ = np.random.multivariate_normal([0, 0], [[self.sd_v**2, self.rho*self.sd_u*self.sd_v], [self.rho*self.sd_u*self.sd_v, self.sd_u**2]],1)
            uv[:, 0] = uv_[0][0]
            uv[:, 1] = uv_[0][1]
        X_obs = X_true + uv + np.random.normal(0, self.sd, X_true.shape)
        if dense == True:
            return s, X_true
        else:
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

#
# if __name__ == "__main__":
#     sdense = np.linspace(0, 15, 100)
#
#     datasets = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10, sdense=sdense,x1x2Ind=False,
#                                               num_samples=5, sd_u=1., sd_v=1.,rho=0.8, sd=0., sd_y=0., end_time=40)
#     # print(datasets[0][1])
#     for i in range(5):
#         timeS, x_obsS, x_trueS = datasets[i][1]
#         time,x_true = datasets[i][0]
#         plt.plot(time.numpy(), x_true.numpy()[:, 0])
#         plt.plot(time.numpy(), x_true.numpy()[:, 1])
#         plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 0])
#         plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 1])
#     plt.show()
