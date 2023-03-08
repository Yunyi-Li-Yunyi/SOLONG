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
class FunctionalData1(Dataset):
    """
     Dataset of functional data series. reranged time scale to be consistent with ODE scenario
         Populations (f1, f2) evolve according to
            f1i(t) = sin(4/6*t)+t/6
            f2i(t) = cos(t/6)+cos(5/6*t)-2

         with the dataset sampled with varying duration of follow-up time t{i,k} ~ exp(\lamda)
         for a fixed set of greeks.
            f1(t) = sin(4/6*t)+t/6+vi(t)
            f2(t) = cos(t/6)+cos(5/6*t)-2+ui(t)
        The correlations between observations controlled by (u,v) ~ MultiNormal()
     ---------
     num_samples : int
         Number of samples of the function contained in dataset.
     lambdaX1 : float
         the duration of follow up for X1 ~ exp(\lambdaX1)
     lambdaX2 : float
         the duration of follow up for X2 ~ exp(\lambdaX2)
     sdense: array
         the time points for dense data
     ts_equal: Bool
         synchronous / asynchronous observations
     num_obs_x1 : int
         The number of X1 observed points
     num_obs_x2 : int
         The number of X2 observed points
     sd_v: float
         the standard deviation of random normal noise for X1
     sd_u: float
         the standard deviation of random normal noise for X2
     rho_w: float
         the covariance within x1(t) and x1(s)
     rho_b: float
         the covariance between x1(t) and x2(t)
     scenario: str
         simulation scenario: A, B, B2
     seed: int
         seed for randomization controlling
     """
    def __init__(self,
                 num_samples=300, lambdaX1=2., lambdaX2=1.,sdense=None,
                 ts_equal = True,
                 num_obs_x1=5,
                 num_obs_x2=7,
                 sd_v=0., sd_u=0., rho_w=0.,rho_b=0.,scenario=None,
                 seed=0):
        self.num_samples = num_samples
        self.lambdaX1 = lambdaX1
        self.lambdaX2 = lambdaX2
        self.sdense = sdense
        self.ts_equal=ts_equal

        self.num_obs_x1 = num_obs_x1
        self.num_obs_x2 = num_obs_x2

        self.sd_v = sd_v
        self.sd_u = sd_u
        self.rho_w = rho_w
        self.rho_b = rho_b
        self.scenario=scenario

        self.seed = seed

        # Generate data
        self.data = [] # True dense data
        self.SparseData = [] # True sparse data
        print("Creating dataset...", flush=True)

        np.random.seed(self.seed)

        for samples in range(self.num_samples):
            # Generate true dense data: synchronous/ asynchronous
            if ts_equal==True:
                times, states_true = self.generate_ts(dense=True)
            else:
                times, states_true = self.generate_ts_2(dense=True)

            times = torch.FloatTensor(times)
            states_true = torch.FloatTensor(states_true)
            self.data.append((times, states_true))

            # Generate SparseData
            if ts_equal==True: # Sparse data for synchronous
                timseSparse, states_obsSparse, states_trueSparse = self.generate_ts(dense=False)
                timseSparse = torch.FloatTensor(timseSparse)
                states_obsSparse = torch.FloatTensor(states_obsSparse)
                states_trueSparse = torch.FloatTensor(states_trueSparse)
                self.SparseData.append((timseSparse, states_obsSparse, states_trueSparse))
            else: # Sparse data for asynchronous
                t1,t2,x1_obs,x2_obs,x1_true,x2_true = self.generate_ts_2(dense=False)
                t1=torch.FloatTensor(t1)
                t2=torch.FloatTensor(t2)
                x1_obs=torch.FloatTensor(x1_obs)
                x2_obs=torch.FloatTensor(x2_obs)
                x1_true=torch.FloatTensor(x1_true)
                x2_true=torch.FloatTensor(x2_true)
                self.SparseData.append((t1, t2, x1_obs,x2_obs,x1_true,x2_true))

    def generate_ts(self, dense):
        # X1(t), X2(t) generated at the same time points
        def x1x2(s):
            """
            Return (X1) and (X2) populations
            """
            return np.array([np.sin(4*s/6)+s/6,
                             np.cos(s/6)+np.cos(5*s/6)-2])

        def error_cov(sizeX, scenario):
            """
            Define the variance-covariance matrix to generate errors
            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario == 'simA': # Independent
                # print("Sim A")
                sigma11 = self.sd_v**2*np.identity(sizeX)
                sigma22 = self.sd_u**2*np.identity(sizeX)
                sigma12 = np.zeros((sizeX, sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B:
            elif scenario == 'simB': # Only Longitudinal Within Variable Correlation
                # print("Sim B")
                sigma11 = self.sd_v**2*np.ones((sizeX, sizeX))
                sigma22 = self.sd_u**2*np.ones((sizeX, sizeX))
                sigma12 = np.zeros((sizeX, sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B2:
            elif scenario == 'simB2': # Between and Within Correlations
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb): # use to calculate Gaussian Process
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X = np.expand_dims(s, 1)
                sigma11 = self.sd_v**2*exponentiated_quadratic(X, X)
                sigma22 = self.sd_u**2*exponentiated_quadratic(X, X)
                sigma12 = self.sd_v*self.sd_u*exponentiated_quadratic(X, X)
                sigma21 = self.sd_u*self.sd_v*exponentiated_quadratic(X, X)

            # sim C:
            elif scenario=='simC':
                # Autoregressive 1 Correlation within variable, between variable correlation exist at each same time point

                # print("Sim C")
                def ar1_corr(n,rho_w):
                    exponent=abs(np.repeat(np.arange(n),n,axis=0).reshape(n,n)-np.arange(n))
                    return rho_w**exponent

                corr11 = ar1_corr(sizeX,self.rho_w)
                sigma11 = self.sd_v**2*corr11
                corr22 = ar1_corr(sizeX,self.rho_w)
                sigma22 = self.sd_u**2*corr22

                sigma12 = self.rho_b*self.sd_v*self.sd_u*np.identity(sizeX)
                sigma21 = self.rho_b*self.sd_v*self.sd_u*np.identity(sizeX)

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            return sigma

        assert self.num_obs_x1 == self.num_obs_x2, "number of observations of X1 and X2 are not equal!"
        # sizeX = np.random.choice(range(*self.num_obs_x1), size=1, replace=True)
        sizeX = self.num_obs_x1

        if dense == True:
            s = self.sdense
            X_true = np.transpose(x1x2(s))
            print(type(X_true))
            return s, X_true
        else:
            s = np.random.exponential(self.lambdaX1, sizeX)
            s[0] = 0
            for sik in range(1, len(s)):
                s[sik] = round(s[sik - 1] + s[sik],2)  # calculate time stamps by accumulating follow-up period
            X_true = np.transpose(x1x2(s))

            sigmaMatrix = error_cov(sizeX=len(X_true),scenario=self.scenario)
            meanVector = np.zeros(2 * len(X_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).reshape((2,-1)).T
            X_obs = X_true + error
            return s, X_obs, X_true

    def generate_ts_2(self, dense):
        assert self.scenario != 'simC', "simC doesn't work for asynchronous scenario!"
        # X1(t), X2(s) are not at the same time point
        def x1x2(s):
            """
            Return (X1) and (X2) populations
            """
            return np.array([np.sin(4*s/6)+s/6,
                             np.cos(s/6)+np.cos(5*s/6)-2])

        def error_cov(sizeX1, sizeX2, scenario):
            """
            Define the variance-covariance matrix to generate errors

            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario == 'simA':
                # print("Sim A")
                sigma11 = self.sd_v ** 2 * np.identity(sizeX1)
                sigma22 = self.sd_u ** 2 * np.identity(sizeX2)
                sigma12 = np.zeros((sizeX1, sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B:
            elif scenario == 'simB':
                # print("Sim B")
                sigma11 = self.sd_v ** 2 * np.ones((sizeX1, sizeX1))
                sigma22 = self.sd_u ** 2 * np.ones((sizeX2, sizeX2))
                sigma12 = np.zeros((sizeX1, sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B2:
            elif scenario == 'simB2':
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X1 = np.expand_dims(s1, 1)
                X2 = np.expand_dims(s2, 1)
                sigma11 = self.sd_v ** 2 * exponentiated_quadratic(X1, X1)
                sigma22 = self.sd_u ** 2 * exponentiated_quadratic(X2, X2)
                sigma12 = self.sd_v * self.sd_u * exponentiated_quadratic(X1, X2)
                sigma21 = self.sd_u * self.sd_v * exponentiated_quadratic(X2, X1)

            # sim C: sim C cannot exist for asynchronous situation

            sigma_top = np.hstack((sigma11, sigma12))
            sigma_bot = np.hstack((sigma21, sigma22))
            sigma = np.vstack((sigma_top, sigma_bot))
            # print(sigma)
            return sigma

        sizeX1 = self.num_obs_x1
        sizeX2 = self.num_obs_x2

        if dense == True:
            s = self.sdense
            X_true = np.transpose(x1x2(s))
            return s, X_true

        else:
            s1 = np.random.exponential(self.lambdaX1, sizeX1)
            s2 = np.random.exponential(self.lambdaX2, sizeX2)

            s1[0] = 0
            for sik in range(1, len(s1)):
                s1[sik] = round(s1[sik - 1] + s1[sik], 2)
            s2[0] = 0
            for sik in range(1, len(s2)):
                s2[sik] = round(s2[sik - 1] + s2[sik], 2)
            X1_true = np.transpose(x1x2(s1))[:, 0]
            X2_true = np.transpose(x1x2(s2))[:, 1]

            sigmaMatrix = error_cov(sizeX1=len(X1_true), sizeX2=len(X2_true), scenario=self.scenario)
            meanVector = np.zeros(len(X1_true) + len(X2_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).squeeze()
            X1_obs = X1_true + error[:len(s1)]
            X2_obs = X2_true + error[len(s1):]

            return s1, s2, np.expand_dims(X1_obs, 1), np.expand_dims(X2_obs, 1), \
                   np.expand_dims(X1_true, 1), np.expand_dims(X2_true, 1)

    def __getitem__(self, index):
        return self.data[index], self.SparseData[index]

    def __len__(self):
        return self.num_samples
class FunctionalDataCov(Dataset):
    """
     Dataset of functional data series. reranged time scale to be consistent with ODE scenario
         Populations (f1, f2) evolve according to
            f1i(t) = sin(4/6*t)+t/6 + beta*c
            f2i(t) = cos(t/6)+cos(5/6*t)-2 + beta*c

         with the dataset sampled with varying duration of follow-up time t{i,k} ~ exp(\lamda)
         for a fixed set of greeks.
            f1(t) = sin(4/6*t)+t/6+beta*c+vi(t)
            f2(t) = cos(t/6)+cos(5/6*t)-2+beta*c+ui(t)
        The correlations between observations controlled by (u,v) ~ MultiNormal()
     ---------
     num_samples : int
         Number of samples of the function contained in dataset.
     beta : float
         true beta for covariate
     lambdaX1 : float
         the duration of follow up for X1 ~ exp(\lambdaX1)
     lambdaX2 : float
         the duration of follow up for X2 ~ exp(\lambdaX2)
     sdense: array
         the time points for dense data
     ts_equal: Bool
         synchronous / asynchronous observations
     num_obs_x1 : int
         The number of X1 observed points
     num_obs_x2 : int
         The number of X2 observed points
     sd_v: float
         the standard deviation of random normal noise for X1
     sd_u: float
         the standard deviation of random normal noise for X2
     rho_w: float
         the covariance within x1(t) and x1(s)
     rho_b: float
         the covariance between x1(t) and x2(t)
     scenario: str
         simulation scenario: A, B, B2
     seed: int
         seed for randomization controlling
     """
    def __init__(self,
                 num_samples=300, beta = 5., lambdaX1=2., lambdaX2=1., sdense=None,
                 ts_equal = True, num_obs_x1=5, num_obs_x2=7,
                 sd_v=0., sd_u=0., rho_w=0., rho_b=0., scenario=None,
                 seed=0):
        self.num_samples = num_samples
        self.beta = beta
        self.lambdaX1 = lambdaX1
        self.lambdaX2 = lambdaX2
        self.sdense = sdense
        self.ts_equal=ts_equal

        self.num_obs_x1 = num_obs_x1
        self.num_obs_x2 = num_obs_x2

        self.sd_v = sd_v
        self.sd_u = sd_u
        self.rho_w = rho_w
        self.rho_b = rho_b
        self.scenario=scenario

        self.seed = seed

        # Generate data
        self.data = [] # True dense data
        self.SparseData = [] # True sparse data
        print("Creating dataset...", flush=True)

        np.random.seed(self.seed)

        for samples in range(self.num_samples):
            # Generate true dense data: synchronous/ asynchronous
            if ts_equal==True:
                times, states_true, c = self.generate_ts(dense=True, seed=samples+self.seed)
            else:
                times, states_true, c = self.generate_ts_2(dense=True, seed= samples+self.seed)

            times = torch.FloatTensor(times)
            states_true = torch.FloatTensor(states_true)
            c = torch.tensor(c)
            self.data.append((times, states_true, c))

            # Generate SparseData
            if ts_equal==True: # Sparse data for synchronous
                timseSparse, states_obsSparse, states_trueSparse,covariate = self.generate_ts(dense=False, seed=samples+self.seed)
                timseSparse = torch.FloatTensor(timseSparse)
                states_obsSparse = torch.FloatTensor(states_obsSparse)
                states_trueSparse = torch.FloatTensor(states_trueSparse)
                covariate = torch.tensor(covariate)

                self.SparseData.append((timseSparse, states_obsSparse, states_trueSparse, covariate))
            else: # Sparse data for asynchronous
                t1,t2,x1_obs,x2_obs,x1_true,x2_true,covariate = self.generate_ts_2(dense=False, seed=samples+self.seed)
                t1=torch.FloatTensor(t1)
                t2=torch.FloatTensor(t2)
                x1_obs=torch.FloatTensor(x1_obs)
                x2_obs=torch.FloatTensor(x2_obs)
                x1_true=torch.FloatTensor(x1_true)
                x2_true=torch.FloatTensor(x2_true)
                covariate = torch.tensor(covariate)
                self.SparseData.append((t1, t2, x1_obs,x2_obs,x1_true,x2_true, covariate))

    def generate_ts(self, dense, seed):
        # X1(t), X2(t) generated at the same time points
        np.random.seed(seed)
        c = np.random.randint(1, 3)  # randomly generate covariate 1 or 2
        def x1x2(s,c):
            """
            Return (X1) and (X2) populations
            """
            return np.array([np.sin(4*s/6)+s/6+self.beta*c,
                             np.cos(s/6)+np.cos(5*s/6)-2+self.beta*c])

        def error_cov(sizeX, scenario):
            """
            Define the variance-covariance matrix to generate errors
            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario == 'simA': # Independent
                # print("Sim A")
                sigma11 = self.sd_v**2*np.identity(sizeX)
                sigma22 = self.sd_u**2*np.identity(sizeX)
                sigma12 = np.zeros((sizeX, sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B:
            elif scenario == 'simB': # Only Longitudinal Within Variable Correlation
                # print("Sim B")
                sigma11 = self.sd_v**2*np.ones((sizeX, sizeX))
                sigma22 = self.sd_u**2*np.ones((sizeX, sizeX))
                sigma12 = np.zeros((sizeX, sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B2:
            elif scenario == 'simB2': # Between and Within Correlations
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb): # use to calculate Gaussian Process
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X = np.expand_dims(s, 1)
                sigma11 = self.sd_v**2*exponentiated_quadratic(X, X)
                sigma22 = self.sd_u**2*exponentiated_quadratic(X, X)
                sigma12 = self.sd_v*self.sd_u*exponentiated_quadratic(X, X)
                sigma21 = self.sd_u*self.sd_v*exponentiated_quadratic(X, X)

            # sim C:
            elif scenario=='simC':
                # Autoregressive 1 Correlation within variable, between variable correlation exist at each same time point

                # print("Sim C")
                def ar1_corr(n,rho_w):
                    exponent=abs(np.repeat(np.arange(n),n,axis=0).reshape(n,n)-np.arange(n))
                    return rho_w**exponent

                corr11 = ar1_corr(sizeX,self.rho_w)
                sigma11 = self.sd_v**2*corr11
                corr22 = ar1_corr(sizeX,self.rho_w)
                sigma22 = self.sd_u**2*corr22

                sigma12 = self.rho_b*self.sd_v*self.sd_u*np.identity(sizeX)
                sigma21 = self.rho_b*self.sd_v*self.sd_u*np.identity(sizeX)

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            return sigma

        assert self.num_obs_x1 == self.num_obs_x2, "number of observations of X1 and X2 are not equal!"
        # sizeX = np.random.choice(range(*self.num_obs_x1), size=1, replace=True)
        sizeX = self.num_obs_x1

        if dense == True:
            s = self.sdense
            X_true = np.transpose(x1x2(s,c))
            print(type(X_true))
            return s, X_true, c
        else:
            s = np.random.exponential(self.lambdaX1, sizeX)
            s[0] = 0
            for sik in range(1, len(s)):
                s[sik] = round(s[sik - 1] + s[sik],2)  # calculate time stamps by accumulating follow-up period
            X_true = np.transpose(x1x2(s,c))
            sigmaMatrix = error_cov(sizeX=len(X_true),scenario=self.scenario)
            meanVector = np.zeros(2 * len(X_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).reshape((2,-1)).T
            X_obs = X_true + error
            return s, X_obs, X_true, c

    def generate_ts_2(self, dense, seed):
        assert self.scenario != 'simC', "simC doesn't work for asynchronous scenario!"
        # X1(t), X2(s) are not at the same time point
        np.random.seed(seed)
        c = np.random.randint(1, 3)  # randomly generate covariate 1 or 2

        def x1x2(s, c):
            """
            Return (X1) and (X2) populations
            """
            return np.array([np.sin(4*s/6)+s/6+self.beta*c,
                             np.cos(s/6)+np.cos(5*s/6)-2+self.beta*c])

        def error_cov(sizeX1, sizeX2, scenario):
            """
            Define the variance-covariance matrix to generate errors

            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario == 'simA':
                # print("Sim A")
                sigma11 = self.sd_v ** 2 * np.identity(sizeX1)
                sigma22 = self.sd_u ** 2 * np.identity(sizeX2)
                sigma12 = np.zeros((sizeX1, sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B:
            elif scenario == 'simB':
                # print("Sim B")
                sigma11 = self.sd_v ** 2 * np.ones((sizeX1, sizeX1))
                sigma22 = self.sd_u ** 2 * np.ones((sizeX2, sizeX2))
                sigma12 = np.zeros((sizeX1, sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B2:
            elif scenario == 'simB2':
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X1 = np.expand_dims(s1, 1)
                X2 = np.expand_dims(s2, 1)
                sigma11 = self.sd_v ** 2 * exponentiated_quadratic(X1, X1)
                sigma22 = self.sd_u ** 2 * exponentiated_quadratic(X2, X2)
                sigma12 = self.sd_v * self.sd_u * exponentiated_quadratic(X1, X2)
                sigma21 = self.sd_u * self.sd_v * exponentiated_quadratic(X2, X1)

            # sim C: sim C cannot exist for asynchronous situation

            sigma_top = np.hstack((sigma11, sigma12))
            sigma_bot = np.hstack((sigma21, sigma22))
            sigma = np.vstack((sigma_top, sigma_bot))
            # print(sigma)
            return sigma

        sizeX1 = self.num_obs_x1
        sizeX2 = self.num_obs_x2

        if dense == True:
            s = self.sdense
            X_true = np.transpose(x1x2(s,c))
            return s, X_true, c

        else:
            s1 = np.random.exponential(self.lambdaX1, sizeX1)
            s2 = np.random.exponential(self.lambdaX2, sizeX2)

            s1[0] = 0
            for sik in range(1, len(s1)):
                s1[sik] = round(s1[sik - 1] + s1[sik], 2)
            s2[0] = 0
            for sik in range(1, len(s2)):
                s2[sik] = round(s2[sik - 1] + s2[sik], 2)
            X1_true = np.transpose(x1x2(s1,c))[:, 0]
            X2_true = np.transpose(x1x2(s2,c))[:, 1]

            sigmaMatrix = error_cov(sizeX1=len(X1_true), sizeX2=len(X2_true), scenario=self.scenario)
            meanVector = np.zeros(len(X1_true) + len(X2_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).squeeze()
            X1_obs = X1_true + error[:len(s1)]
            X2_obs = X2_true + error[len(s1):]

            return s1, s2, np.expand_dims(X1_obs, 1), np.expand_dims(X2_obs, 1), \
                   np.expand_dims(X1_true, 1), np.expand_dims(X2_true, 1), c

    def __getitem__(self, index):
        return self.data[index], self.SparseData[index]

    def __len__(self):
        return self.num_samples
class LotkaVolterraDataAdditive(Dataset):
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
            return np.array([np.cos(X[1])+np.sin(X[0]),
                             X[1]*(1-X[0])])

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
if __name__ == "__main__":
    sdense = np.linspace(0, 100, 100)

    datasets = LotkaVolterraDataAdditive(alpha=1. / 10, beta=1. / 10, gamma=1./10, sdense=sdense,x1x2Ind=False,
                                              num_samples=5, sd_u=0.0, sd_v=0.0,rho=0.8, sd=0., sd_y=0., end_time=40)
    # print(datasets[0][1])
    for i in range(5):
        timeS, x_obsS, x_trueS = datasets[i][1]
        time,x_true = datasets[i][0]
        plt.plot(time.numpy(), x_true.numpy()[:, 0])
        plt.plot(time.numpy(), x_true.numpy()[:, 1])
        plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 0])
        plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 1])
    plt.show()
