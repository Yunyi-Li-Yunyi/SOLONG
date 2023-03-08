import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# folder='/N/slate/liyuny/cnode_ffr_main/results/AdniTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE0_finalized'
class ODEplots:

    def __init__(self,begin,end,t):
        super(ODEplots, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.begin = begin
        self.end = end

        self.x = np.arange(self.begin, self.end, 0.1)
        self.y = np.arange(self.begin, self.end, 0.1)
        self.total = np.size(self.x)
        self.tplot = torch.tensor(t)

    def ODE3d(self,folder):
        func = torch.load(osp.join(folder, 'trained_model_0.pth')).to(self.device)
        X,Y = np.meshgrid(self.x,self.y)
        # torchXY = torch.tensor([X,Y]).float()

        def f(x, y):
            torchXY = torch.tensor([x,y]).float()
            return func(self.tplot,torchXY).detach().numpy()

        ZX = []
        ZY = []
        for ix in range(self.total):
            for iy in range(self.total):
                ZX.append(f(self.x[ix],self.y[iy])[0])
                ZY.append(f(self.x[ix],self.y[iy])[1])
        ZX = np.reshape(ZX,(self.total, self.total)).T
        ZY = np.reshape(ZY,(self.total, self.total)).T

        # if __name__ == "__main__":
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, ZX, 50, cmap='binary')
        # ax.plot_wireframe(X, Y, ZX, color='black')
        ax.plot_surface(X, Y, ZX, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Amyloid')
        ax.set_ylabel('Tau')
        ax.set_zlabel('Rate of Change')
        ax.set_title('ODE Values of Amyloid')
        plt.savefig(folder+"/ODEAmyloid_time"+str(self.tplot))
        # plt.show()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, ZX, 50, cmap='binary')
        # ax.plot_wireframe(X, Y, ZX, color='black')
        ax.plot_surface(X, Y, ZY, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Amyloid')
        ax.set_ylabel('Tau')
        ax.set_zlabel('Rate of Change')
        ax.set_title('ODE Values of Tau')
        plt.savefig(folder+"/ODETau_time"+str(self.tplot))
        # plt.show()

    def ODEslide(self,folder,cutPointX,cutpointY):
        func = torch.load(osp.join(folder, 'trained_model_0.pth')).to(self.device)
        def f(x, y):
            torchXY = torch.tensor([x,y]).float()
            return func(self.tplot,torchXY).detach().numpy()
        Zx=[]
        for i in range(self.total):
            Zx.append(f(cutPointX,self.y[i])[0])

        # if __name__ == "__main__":
        fig = plt.figure()
        plt.plot(self.y,Zx)
        plt.xlabel('Tau')
        plt.ylabel('Rate of Change')
        plt.title('ODE Values of Amyloid When Amyloid is 0')
        plt.savefig(folder + "/ODEAmyloid_Amy0_time"+str(self.tplot))
        # plt.show()

        Zy=[]
        for i in range(self.total):
            Zy.append(f(self.x[i],cutpointY)[1])
        fig = plt.figure()
        plt.plot(self.x,Zy)
        plt.xlabel('Amyloid')
        plt.ylabel('Rate of Change')
        plt.title('ODE Values of Tau When Tau is 0')
        plt.savefig(folder + "/ODETau_Tau0_time"+str(self.tplot))
        # plt.show()

    def ODEslideComb(self,cutPointX,cutpointY,folder1,folder2):
        func1 = torch.load(osp.join(folder1, 'trained_model_0.pth')).to(self.device)
        func2 = torch.load(osp.join(folder2, 'trained_model_0.pth')).to(self.device)

        def f(x, y,func):
            torchXY = torch.tensor([x,y]).float()
            return func(self.tplot,torchXY).detach().numpy()

        Zx1=[]
        for i in range(self.total):
            Zx1.append(f(cutPointX,self.y[i],func1)[0])
        Zx2=[]
        for i in range(self.total):
            Zx2.append(f(cutPointX,self.y[i],func2)[0])
        Zy1 = []
        for i in range(self.total):
            Zy1.append(f(self.x[i], cutpointY,func1)[1])
        Zy2 = []
        for i in range(self.total):
            Zy2.append(f(self.x[i], cutpointY,func2)[1])

        fig = plt.figure()
        plt.plot(self.y,Zx1,label="APOE4 non-carrier", c='#FF8000')
        plt.plot(self.y,Zx2,label="APOE4 carrier", c='#FF8000',linestyle='dashed')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('Tau')
        plt.ylabel('Rate of Change')
        plt.title('ODE Values of Amyloid When Amyloid is 0')
        plt.legend()
        plt.savefig(folder1 + "/ODEAmyloid_Amy0_Comb_time"+str(self.tplot))
        # plt.show()

        fig = plt.figure()
        plt.plot(self.y,Zy1,label="APOE4 non-carrier", c='#27408B')
        plt.plot(self.y,Zy2,label="APOE4 carrier", c='#27408B',linestyle='dashed')
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.xlabel('Amyloid')
        plt.ylabel('Rate of Change')
        plt.title('ODE Values of Tau When Tau is 0')
        plt.legend()
        plt.savefig(folder1 + "/ODETau_Tau0_Comb_time"+str(self.tplot))
        # plt.show()

if  __name__ == "__main__":
    folder1 = '/N/slate/liyuny/cnode_ffr_main/results/AdniTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE0_finalized'
    folder2 = '/N/slate/liyuny/cnode_ffr_main/results/AdniTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE12_finalized'
    ODEplots(-10,10,10).ODEslideComb(0, 0,folder1,folder2)
    ODEplots(-10,10,10).ODE3d(folder1)
    ODEplots(-10,10,10).ODE3d(folder2)

    ODEplots(-10,10,15).ODEslideComb(0, 0,folder1,folder2)
    ODEplots(-10,10,15).ODE3d(folder1)
    ODEplots(-10,10,15).ODE3d(folder2)
