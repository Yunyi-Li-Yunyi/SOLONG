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
    num_obs_x1 : Tuple[int,int]
        The range of number of context points.

    lambdaY: float
        the duration of follow up ~ exp(\lambdaY)
    sd_y: float
        the standard deviation of random normal noise
    """

    def __init__(self,
                 alpha=None, beta=None, gamma=None,
                 num_samples=300, lambdaX1=2., lambdaX2=1.,sdense=None,
                 ts_equal = True,
                 num_obs_x1=(5, 6),
                 num_obs_x2=(7,8),
                 sd_v=0., sd_u=0., rho=0.,scenario=None,
                 lambdaY=1., sd_y=0., num_context_rangeY=(5, 6), seed=0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.num_samples = num_samples
        self.sdense = sdense
        self.lambdaX1 = lambdaX1
        self.lambdaX2 = lambdaX2

        self.sd_v = sd_v
        self.sd_u = sd_u
        self.rho = rho
        self.scenario=scenario
        self.ts_equal=ts_equal
        self.num_obs_x1 = num_obs_x1
        self.num_obs_x2 = num_obs_x2

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


        # for samples in tqdm(range(self.num_samples)):
        for samples in range(self.num_samples):
            # generate x, states_obs, states_true
            if ts_equal==True:
                times, states_true = self.generate_ts(dense=True)
            else:
                times, states_true = self.generate_ts_2(dense=True)

            times = torch.FloatTensor(times)
            states_true = torch.FloatTensor(states_true)

            # states = torch.cat((states,times),dim=-1)
            self.data.append((times, states_true))
            # Generate SparseData
            if ts_equal==True:
                timseSparse, states_obsSparse, states_trueSparse = self.generate_ts(dense=False)
                timseSparse = torch.FloatTensor(timseSparse)
                states_obsSparse = torch.FloatTensor(states_obsSparse)
                states_trueSparse = torch.FloatTensor(states_trueSparse)

                self.SparseData.append((timseSparse, states_obsSparse, states_trueSparse))
            else:
                t1,t2,x1_obs,x2_obs,x1_true,x2_true = self.generate_ts_2(dense=False)
                t1=torch.FloatTensor(t1)
                t2=torch.FloatTensor(t2)
                x1_obs=torch.FloatTensor(x1_obs)
                x2_obs=torch.FloatTensor(x2_obs)
                x1_true=torch.FloatTensor(x1_true)
                x2_true=torch.FloatTensor(x2_true)

                self.SparseData.append((t1, t2, x1_obs,x2_obs,x1_true,x2_true))

            # Generate Outcome
            self.OutcomeData.append(self.Outcome(np.array(times), states_true))


    def generate_ts(self, dense):
        # X1(t), X2(t) generated at the same time points
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
                # print("Sim B")
                sigma11=self.sd_v**2*np.ones((sizeX,sizeX))
                sigma22=self.sd_u**2*np.ones((sizeX,sizeX))
                sigma12=np.zeros((sizeX,sizeX))
                sigma21 = np.zeros((sizeX, sizeX))

            # sim B2:
            elif scenario=='simB2':
                # print("Sim B2")
                def exponentiated_quadratic(xa, xb):
                    """Exponentiated quadratic with \sigma=1"""
                    # L2 distance (Squared Euclidian)
                    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
                    return np.exp(sq_norm)

                X = np.expand_dims(s, 1)
                sigma11=self.sd_v**2*exponentiated_quadratic(X, X)
                sigma22=self.sd_u**2*exponentiated_quadratic(X, X)
                sigma12=self.sd_v*self.sd_u*exponentiated_quadratic(X, X)
                sigma21 =self.sd_u*self.sd_v*exponentiated_quadratic(X, X)

            # sim C:
            elif scenario=='simC':
                # print("Sim C")
                def ar1_corr(n,rho):
                    exponent=abs(np.repeat(np.arange(n),n,axis=0).reshape(n,n)-np.arange(n))
                    return rho**exponent

                # sigma11=self.sd_v**2*np.ones((sizeX1,sizeX1))
                # sigma22=self.sd_u**2*np.ones((sizeX2,sizeX2))
                corr11=ar1_corr(sizeX,self.rho)
                sigma11=self.sd_v**2*corr11
                corr22=ar1_corr(sizeX,self.rho)
                sigma22=self.sd_u**2*corr22

                # sigma11=self.sd_v**2*np.ones((sizeX,sizeX))
                # sigma22=self.sd_u**2*np.ones((sizeX,sizeX))
                sigma12=2*self.rho*self.sd_v*self.sd_u*np.identity(sizeX)
                sigma21=2*self.rho*self.sd_v*self.sd_u*np.identity(sizeX)
                # sigma12=np.identity(sizeX)
                # sigma21=np.identity(sizeX)

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            # print(sigma)
            return sigma

        assert self.num_obs_x1==self.num_obs_x2, "number of observations of X1 and X2 are not equal!"
        # sizeX = np.random.choice(range(*self.num_obs_x1), size=1, replace=True)
        sizeX=self.num_obs_x1

        if dense == True:
            s = self.sdense
        else:
            s = np.random.exponential(self.lambdaX1, sizeX)
            s[0] = 0
            for sik in range(1, len(s)):
                s[sik] = round(s[sik - 1] + s[sik],2)
        X_true = odeint(dX_dt, X_0, s)

        if dense == True:
            return s, X_true
        else:
            sigmaMatrix = error_cov(sizeX=len(X_true),scenario=self.scenario)
            meanVector = np.zeros(2 * len(X_true))
            # print(sigmaMatrix)

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).reshape((2,-1)).T
            X_obs = X_true + error

            return s, X_obs, X_true

    def generate_ts_2(self, dense):
        # X1(t), X2(s) are not at the same time point
        equal_pop = np.random.uniform(1., 1.)
        X_0 = np.array([equal_pop, equal_pop])
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
            if scenario=='simA':
                # print("Sim A")
                sigma11=self.sd_v**2*np.identity(sizeX1)
                sigma22=self.sd_u**2*np.identity(sizeX2)
                sigma12=np.zeros((sizeX1,sizeX2))
                sigma21 = np.zeros((sizeX2, sizeX1))

            # sim B:
            elif scenario=='simB':
                # print("Sim B")
                sigma11=self.sd_v**2*np.ones((sizeX1,sizeX1))
                sigma22=self.sd_u**2*np.ones((sizeX2,sizeX2))
                sigma12=np.zeros((sizeX1,sizeX2))
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
                sigma11=self.sd_v**2*exponentiated_quadratic(X1, X1)
                sigma22=self.sd_u**2*exponentiated_quadratic(X2, X2)
                sigma12=self.sd_v*self.sd_u*exponentiated_quadratic(X1, X2)
                sigma21 =self.sd_u*self.sd_v*exponentiated_quadratic(X2, X1)

            # sim C:
            # elif scenario=='simC':
            #     # print("Sim C")
            #     def ar1_corr(n,rho):
            #         exponent=abs(np.repeat(np.arange(n),n,axis=0).reshape(n,n)-np.arange(n))
            #         return rho**exponent
            #
            #     # sigma11=self.sd_v**2*np.ones((sizeX1,sizeX1))
            #     # sigma22=self.sd_u**2*np.ones((sizeX2,sizeX2))
            #     corr11=ar1_corr(sizeX1,self.rho)
            #     sigma11=self.sd_v**2*corr11
            #     print(sigma11)
            #     corr22=ar1_corr(sizeX2,self.rho)
            #     sigma22=self.sd_u**2*corr22
            #     print(sigma22)
            #     sigma12=self.rho*self.sd_v*self.sd_u*np.identity(sizeX1)
            #     print(sigma12)
            #     sigma21=self.rho*self.sd_v*self.sd_u*np.identity(sizeX2)
            #     print(sigma21)

            sigma_top = np.hstack((sigma11,sigma12))
            sigma_bot = np.hstack((sigma21,sigma22))
            sigma = np.vstack((sigma_top,sigma_bot))
            # print(sigma)
            return sigma

        # sizeX1 = np.random.choice(range(*self.num_obs_x1), size=1, replace=True)
        # sizeX2 = np.random.choice(range(*self.num_obs_x2), size=1, replace=True)
        sizeX1=self.num_obs_x1
        sizeX2=self.num_obs_x2

        if dense == True:
            s = self.sdense
            X_true = odeint(dX_dt, X_0, s)
            return s, X_true

        else:
            s1 = np.random.exponential(self.lambdaX1, sizeX1)
            s2 = np.random.exponential(self.lambdaX2, sizeX2)
            # print(s1)
            # print(s2)

            s1[0] = 0
            for sik in range(1, len(s1)):
                s1[sik] = round(s1[sik - 1] + s1[sik],2)
            s2[0] = 0
            for sik in range(1, len(s2)):
                s2[sik] = round(s2[sik - 1] + s2[sik],2)
            X1_true = odeint(dX_dt,X_0,s1)[:,0]
            X2_true = odeint(dX_dt,X_0,s2)[:,1]

            sigmaMatrix = error_cov(sizeX1=len(X1_true),sizeX2=len(X2_true),scenario=self.scenario)
            meanVector = np.zeros(len(X1_true)+len(X2_true))

            error = np.random.multivariate_normal(meanVector, sigmaMatrix, 1).squeeze()
            X1_obs = X1_true + error[:len(s1)]
            X2_obs = X2_true + error[len(s1):]

            return s1,s2, np.expand_dims(X1_obs,1), np.expand_dims(X2_obs,1), \
            np.expand_dims(X1_true,1), np.expand_dims(X2_true,1)
        
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
        return beta0,beta1,beta2,t,outcome

    def __getitem__(self, index):
        return self.data[index], self.SparseData[index], self.OutcomeData[index]

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    sdense = np.linspace(0, 15, 100)

    datasets = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10, sdense=sdense,
                                              num_samples=1, sd_u=1., sd_v=1.,rho=0.2, scenario='simC',
                                              num_obs_x1=4, num_obs_x2=4,ts_equal=True,sd_y=0.)
#     t1, t2, x1_obs, x2_obs, x1_true, x2_true = datasets[0][1]
#     t,x_obs,x_true = datasets[0][1]
#     print(x1_obs)
    # for i in range(1):
    #     t1,t2, x1_obs,x2_obs, x1_true,x2_true = datasets[i][1]
    #     time,x_true = datasets[i][0]
    #     plt.plot(time.numpy(), x_true.numpy()[:, 0])
    #     plt.plot(time.numpy(), x_true.numpy()[:, 1])
    #     plt.scatter(t1.numpy(), x1_obs.numpy()[:])
    #     plt.scatter(t2.numpy(), x2_obs.numpy()[:])
    # plt.show()
