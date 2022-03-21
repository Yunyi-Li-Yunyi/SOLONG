from pygam import LinearGAM,s,te
from data.dataset import DeterministicLotkaVolterraData
# os.environ['R_HOME']='./N/soft/rhel7/r/4.1.1/lib64/R/'

# import rpy2.robjects as robjects
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp


sdense = np.linspace(0, 15, 100)

dataset = DeterministicLotkaVolterraData(alpha=3. / 4, beta=1. / 10, gamma=1. / 10,
                                         num_samples=300, scenario='simB2' ,sd_u=0.3,
                                         sd_v=0.3, rho=0.9, lambdaX=2., sdense=sdense,
                                         num_context_range=(5,6), seed=0)

folder='/N/u/liyuny/Carbonate/cnode_ffr_main/results/deterministic_lv/vnode/sim_b2'

datasets_read = np.load(folder + '/Data_0.npy', allow_pickle=True)
predX = np.load(folder + '/predX.npy', allow_pickle=True)

func = torch.load(osp.join(folder, 'trained_model_0.pth'))

time,x_true = datasets_read[0][0] #true dense values
time_s,x_obs_s,x_true_s = datasets_read[0][1] #Sparse observations

yvec=[]
tvec=[]
b0True = []
b1True = []
b2True = []
for i in range(len(dataset)):
    b0, b1, b2, t, outcome = dataset[i][2]
    yvec.append(outcome.detach().numpy())
    tvec.append(t)
    b0True.append(b0.detach().numpy())
    b1True.append(b1.detach().numpy())
    b2True.append(b2.detach().numpy())

by=time[1]-time[0]

lX = by*func(time,predX[0])
# yvec = np.hstack(yvec)
# tvec=np.hstack(tvec)
#
# np.savetxt(osp.join(folder,'y.csv'),yvec,delimiter=',')
# np.savetxt(osp.join(folder,'t.csv'),tvec,delimiter=',')
# np.savetxt(osp.join(folder,'s.csv'),time,delimiter=',')
# np.savetxt(osp.join(folder,'odeX.csv'),lX.detach().numpy(),delimiter=',')
#
# # importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#
fig = plt.figure()

# syntax for 3-D projection
ax = fig.add_subplot(111,projection='3d')
b1True=np.hstack(b1True)
# print(tvec[:10])
# print(b1True[:10,:10])
# z=b1True.reshape(tvec.shape)
X,Y=np.meshgrid(tvec,time.detach().numpy())
ax.scatter(Y,X,b1True)
plt.show()
# lX = by*predX[0]
# lx1 = by * predX[0][0]
# lx2 = by * predX[0][1]

# lX=lX.detach().numpy()
# # X=np.stack(tvec,)
# print(tvec.shape)
# print(lX.shape)
# print(yvec.shape)
# print(lX.detach().numpy())
# gam=LinearGAM(te(0,0,by=0)+te(0,0)).fit(X,np.stack(yvec))
# print(gam.summary())
# if __name__=='__main__':
#
# t=np.linspace(0, 5, 100)
# s=np.linspace(0, 2, 40)
# X,Y=np.meshgrid(t,s)
# def Beta1(t, s):
#     return np.cos(t * np.pi / 3) * np.sin(s * np.pi / 5)
# zs = np.array(Beta1(np.ravel(X), np.ravel(Y)))
#
# Z = zs.reshape(X.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(X, Y, Z)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()