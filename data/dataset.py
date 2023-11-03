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
# from tqdm import tqdm
from scipy.integrate import odeint
# from torchdiffeq import odeint
from scipy import ndimage
import scipy.spatial.distance

class DeterministicLotkaVolterraData(Dataset):
    """
    Dataset of Lotka-Volterra time series.
        Populations (u,v) or say (X1, X2) evolve according to

            u' = u*(1-u) -\alpha * u*v/(u+c)
            v' = \beta*v*(1-(v/u))

        with the dataset sampled with varying duration of follow-up time t{i,k} ~ exp(\lamda)
        for a fixed set of greeks.
        X1i(t) = X1(t)+vi(t)
        X2i(t) = X2(t)+ui(t)

        The correlations between observations controlled by (u,v) ~ MultiNormal()
    ---------
    alpha : int
        fixed initial value for \alpha
    beta  : int
        fixed initial value for \beta
    gamma : int
        fixed initial value for \gamme
    initial: float
        fixed true initial value for ODE system
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
    followup: str
        the distribution of follow-up duration
    """
    def __init__(self,
                 alpha=None, beta=None, gamma=None,
                 initial=1.,
                 num_samples=300, lambdaX1=2., lambdaX2=2.,sdense=None,
                 ts_equal = True,
                 num_obs_x1=5,
                 num_obs_x2=7,
                 sd_v=0., sd_u=0., rho_w=0.,rho_b=0.,scenario=None,
                 seed=0, followup=None):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.initial=initial

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
        self.followup=followup

        self.seed = seed

        # Generate data
        self.data = [] # True dense data
        self.SparseData = [] # True sparse data
        print("Creating dataset...", flush=True)

        np.random.seed(self.seed)

        # for samples in tqdm(range(self.num_samples)):
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
        # X1(t), X2(t) generated at the same time points/synchronous
        # equal_pop = np.random.uniform(1., 1.) # True Initial Values X0
        X_0 = np.array([self.initial, self.initial]) # True Initial Values [X10,x20]
        a, b, c = self.alpha, self.beta, self.gamma

        def dX_dt(X, s=0):
            """
            Return the growth rate of fox(X1) and rabbit(X2) populations
            """
            return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1] / (X[0] + c)),
                             b * X[1] * (1 - X[1] / X[0])])
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
            elif scenario == 'simB2': # Between and Within Correlations [Gaussian Process]
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
            X_true = odeint(dX_dt, X_0, s)
            return s, X_true
        else:
            if self.followup=='exponential':
                # generate observation time schedule based on exponential distribution
                s = np.random.exponential(self.lambdaX1, sizeX)
                s[0] = 0
                for sik in range(1, len(s)):
                    s[sik] = round(s[sik - 1] + s[sik],2)  # calculate time stamps by accumulating follow-up period

            elif self.followup=='uniform':
                # generate observations follow uniform distribution
                # s = self.sdense
                # X_true = odeint(dX_dt, X_0, s)
                samp_s_idx = np.random.choice(range(1, self.sdense.size), (self.num_obs_x1-1), replace=False)
                samp_s_idx.sort()
                s = [0]+self.sdense[samp_s_idx].tolist()
                # X_true = X_true[samp_s_idx]
            else:
                print("please specify follow-up distribution")
            X_true = odeint(dX_dt, X_0, s)
            sigmaMatrix = error_cov(sizeX=len(X_true),scenario=self.scenario)
            meanVector = np.zeros(2 * len(X_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).reshape((2,-1)).T
            X_obs = X_true + error

            return s, X_obs, X_true

    def generate_ts_2(self, dense):
        # X1(t), X2(t) asynchronous
        assert self.scenario != 'simC', "simC doesn't work for asynchronous scenario!"
        # X1(t), X2(s) are not at the same time point
        # equal_pop = np.random.uniform(1., 1.)
        X_0 = np.array([self.initial, self.initial])
        a, b, c = self.alpha, self.beta, self.gamma

        def dX_dt(X, s=0):
            """
            Return the growth rate of fox and rabbit populations
            """
            return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1] / (X[0] + c)),
                             b * X[1] * (1 - X[1] / X[0])])

        def error_cov(sizeX1,sizeX2,scenario):
            """
            Define the variance-covariance matrix to generate errors

            :return:
            Return the variance-covariance matrix of errors
            """
            # sim A:
            if scenario == 'simA':
                # print("Sim A")
                sigma11 = self.sd_v**2*np.identity(sizeX1)
                sigma22 = self.sd_u**2*np.identity(sizeX2)
                sigma12 = np.zeros((sizeX1,sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B:
            elif scenario == 'simB':
                # print("Sim B")
                sigma11 = self.sd_v**2*np.ones((sizeX1, sizeX1))
                sigma22 = self.sd_u**2*np.ones((sizeX2, sizeX2))
                sigma12 = np.zeros((sizeX1, sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B2:
            elif scenario=='simB2':
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X1 = np.expand_dims(s1, 1)
                X2 = np.expand_dims(s2,1)
                sigma11 = self.sd_v**2*exponentiated_quadratic(X1, X1)
                sigma22 = self.sd_u**2*exponentiated_quadratic(X2, X2)
                sigma12 = self.sd_v*self.sd_u*exponentiated_quadratic(X1, X2)
                sigma21 = self.sd_u*self.sd_v*exponentiated_quadratic(X2, X1)

            # sim C: sim C cannot exist for asynchronous situation

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            # print(sigma)
            return sigma

        sizeX1 = self.num_obs_x1
        sizeX2 = self.num_obs_x2

        if dense == True:
            s = self.sdense
            X_true = odeint(dX_dt, X_0, s)
            return s, X_true

        else:
            if self.followup=='exponential':
                s1 = np.random.exponential(self.lambdaX1, sizeX1)
                s2 = np.random.exponential(self.lambdaX2, sizeX2)

                s1[0] = 0
                for sik in range(1, len(s1)):
                    s1[sik] = round(s1[sik - 1] + s1[sik],2)
                s2[0] = 0
                for sik in range(1, len(s2)):
                    s2[sik] = round(s2[sik - 1] + s2[sik],2)

            elif self.followup=='uniform':
                samp_s1_idx = np.random.choice(range(1, self.sdense.size), (self.num_obs_x1-1), replace=False)
                samp_s1_idx.sort()
                s1 = [0]+self.sdense[samp_s1_idx].tolist()
                samp_s2_idx = np.random.choice(range(1, self.sdense.size), (self.num_obs_x2-1), replace=False)
                samp_s2_idx.sort()
                s2 = [0]+self.sdense[samp_s2_idx].tolist()
            else:
                print("please specify follow-up distribution")
            X1_true = odeint(dX_dt,X_0,s1)[:,0]
            X2_true = odeint(dX_dt,X_0,s2)[:,1]

            sigmaMatrix = error_cov(sizeX1=len(X1_true),sizeX2=len(X2_true),scenario=self.scenario)
            meanVector = np.zeros(len(X1_true)+len(X2_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).squeeze()
            X1_obs = X1_true + error[:len(s1)]
            X2_obs = X2_true + error[len(s1):]

            return s1, s2, np.expand_dims(X1_obs,1), np.expand_dims(X2_obs,1), \
            np.expand_dims(X1_true,1), np.expand_dims(X2_true,1)

    def __getitem__(self, index):
        return self.data[index], self.SparseData[index]

    def __len__(self):
        return self.num_samples

class FunctionalData(Dataset):
    """
     Dataset of functional data series.
         Populations (f1, f2) evolve according to
            f1i(t) = sin(4t)+t
            f2i(t) = cos(t)+cos(5t)-2

         with the dataset sampled with varying duration of follow-up time t{i,k} ~ exp(\lamda)
         for a fixed set of greeks.
            f1(t) = sin(4t)+t+vi(t)
            f2(t) = cos(t)+cos(5t)-2+ui(t)
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
                 seed=0,followup=None):
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
        self.followup = followup

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
            s=np.array(s)
            """
            Return (X1) and (X2) populations
            """
            return np.array([np.sin(4*s)+s,
                             np.cos(s)+np.cos(5*s)-2])

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
            if self.followup=='exponential':
                s = np.random.exponential(self.lambdaX1, sizeX)
                s[0] = 0
                for sik in range(1, len(s)):
                    s[sik] = round(s[sik - 1] + s[sik],2)  # calculate time stamps by accumulating follow-up period

            elif self.followup=='uniform':
                samp_s_idx = np.random.choice(range(1, self.sdense.size), (sizeX-1), replace=False)
                samp_s_idx.sort()
                s = [0]+self.sdense[samp_s_idx].tolist()

            else: print("please specify follow-up distribution")

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
            return np.array([np.sin(4*s)+s,
                             np.cos(s)+np.cos(5*s)-2])

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
            if self.followup=='exponential':
                s1 = np.random.exponential(self.lambdaX1, sizeX1)
                s2 = np.random.exponential(self.lambdaX2, sizeX2)

                s1[0] = 0
                for sik in range(1, len(s1)):
                    s1[sik] = round(s1[sik - 1] + s1[sik], 2)
                s2[0] = 0
                for sik in range(1, len(s2)):
                    s2[sik] = round(s2[sik - 1] + s2[sik], 2)

            elif self.followup=='uniform':
                samp_s1_idx = np.random.choice(range(1, self.sdense.size), (self.num_obs_x1-1), replace=False)
                samp_s1_idx.sort()
                samp_s2_idx = np.random.choice(range(1, self.sdense.size), (self.num_obs_x2-1), replace=False)
                samp_s2_idx.sort()
                s1 = [0]+ self.sdense[samp_s1_idx].tolist()
                s2 = [0]+ self.sdense[samp_s2_idx].tolist()
            else:
                print("please specify follow-up distribution")
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

if __name__ == "__main__":
    ts_equal = True
    sdense = np.linspace(0, 10, 100)

    # datasets = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10, sdense=sdense,
    #                                           num_samples=1, sd_u=0.1, sd_v=0.1,rho_b=0.3,rho_w=0.5, scenario='simA',
    #                                           initial=1.0,
    #                                           num_obs_x1=5, num_obs_x2=5,ts_equal=ts_equal,seed=1,followup='uniform')

    datasets = FunctionalData(sdense=sdense, num_samples=1, sd_u=2.0, sd_v=2.0,rho_b=0.,rho_w=0., scenario='simA',
                                                  num_obs_x1=5, num_obs_x2=5,ts_equal=ts_equal,seed=1,followup='exponential')
    if ts_equal == True:
        t,x_obs,x_true = datasets[0][1]
        print(t)
        print(x_obs)
        print(x_true)
        time, x_true = datasets[0][0]
        print(x_true)
        for i in range(1):
            time, x_true = datasets[i][0]
            plt.plot(time.numpy(), x_true.numpy()[:, 0],label=r'$\mu_1(t)$')
            plt.plot(time.numpy(), x_true.numpy()[:, 1],label=r'$\mu_2(t)$')
            plt.scatter(t.numpy(), x_obs.numpy()[:, 0])
            plt.scatter(t.numpy(), x_obs.numpy()[:, 1])
        plt.xlabel("t")
        plt.legend()
        # plt.savefig('/N/u/liyuny/Carbonate/Paper1/ODEstudy2')
        plt.show()

    else:
        t1, t2, x1_obs, x2_obs, x1_true, x2_true = datasets[0][1]
        print(t1)
        print(t2)
        print(x1_obs)
        print(x1_true)
        print(x2_obs)
        print(x2_true)
        for i in range(1):
            t1,t2, x1_obs,x2_obs, x1_true,x2_true = datasets[i][1]
            time,x_true = datasets[i][0]
            plt.plot(time.numpy(), x_true.numpy()[:, 0])
            plt.plot(time.numpy(), x_true.numpy()[:, 1])
            plt.scatter(t1.numpy(), x1_obs.numpy()[:])
            plt.scatter(t2.numpy(), x2_obs.numpy()[:])
        plt.show()
# if __name__ == "__main__":
#     ts_equal = True
#     sdense = np.linspace(0, 15, 100)
#
#     datasets = FunctionalData2(sdense=sdense, num_samples=1, sd_u=2.0, sd_v=2.0,rho_b=0.,rho_w=0., scenario='simA',
#                                               num_obs_x1=5, num_obs_x2=5,ts_equal=ts_equal,seed=2)
#     if ts_equal == True:
#         t,x_obs,x_true = datasets[0][1]
#         for i in range(1):
#             time, x_true = datasets[i][0]
#             plt.plot(time.numpy(), x_true.numpy()[:, 0])
#             plt.plot(time.numpy(), x_true.numpy()[:, 1])
#             plt.scatter(t.numpy(), x_obs.numpy()[:, 0])
#             plt.scatter(t.numpy(), x_obs.numpy()[:, 1])
#         plt.show()
#     else:
#         t1, t2, x1_obs, x2_obs, x1_true, x2_true = datasets[0][1]
#         print(x1_obs)
#         print(x2_obs)
#         for i in range(1):
#             t1,t2, x1_obs,x2_obs, x1_true,x2_true = datasets[i][1]
#             time,x_true = datasets[i][0]
#             plt.plot(time.numpy(), x_true.numpy()[:, 0])
#             plt.plot(time.numpy(), x_true.numpy()[:, 1])
#             plt.scatter(t1.numpy(), x1_obs.numpy()[:])
#             plt.scatter(t2.numpy(), x2_obs.numpy()[:])
#         plt.show()
#
# if __name__ == "__main__":
#     ts_equal = True
#     sdense = np.linspace(0, 15, 100)
#
    # dataset = FunctionalDataCov(sdense=sdense, num_samples=3, sd_u=0.2, sd_v=0.2,rho_b=0.,rho_w=0., scenario='simA',
    #                                           num_obs_x1=5, num_obs_x2=5,ts_equal=ts_equal)
#     if ts_equal == True:
#         t,x_obs,x_true,covariate = datasets[0][1]
#         for i in range(1):
#             time, x_true,covariate = datasets[i][0]
#             plt.plot(time.numpy(), x_true.numpy()[:, 0])
#             plt.plot(time.numpy(), x_true.numpy()[:, 1])
#             plt.scatter(t.numpy(), x_obs.numpy()[:, 0])
#             plt.scatter(t.numpy(), x_obs.numpy()[:, 1])
#         plt.show()
#     else:
#         t1, t2, x1_obs, x2_obs, x1_true, x2_true, covariate = datasets[0][1]
#         # print(x1_obs)
#         # print(x2_obs)
#         print(covariate)
#         for i in range(1):
#             t1,t2, x1_obs,x2_obs, x1_true,x2_true, covariate = datasets[i][1]
#             print(covariate)
#             time,x_true, covariate = datasets[i][0]
#             print(covariate)
#             plt.plot(time.numpy(), x_true.numpy()[:, 0])
#             plt.plot(time.numpy(), x_true.numpy()[:, 1])
#             plt.scatter(t1.numpy(), x1_obs.numpy()[:])
#             plt.scatter(t2.numpy(), x2_obs.numpy()[:])
#         plt.show()