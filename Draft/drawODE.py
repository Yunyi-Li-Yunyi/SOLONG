import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
folder='/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_intial1_appfunc_3Relu_decay4_APOE0_finalized_v2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

func = torch.load(osp.join(folder, 'trained_model_0.pth')).to(device)
begin = -10
end = 10

x = np.arange(begin, end,0.1)
y = np.arange(begin, end,0.1)

total = np.size(x)
def ODE3d():
    X,Y = np.meshgrid(x,y)
    # torchXY = torch.tensor([X,Y]).float()

    def f(x, y):
        torchXY = torch.tensor([x,y]).float()
        return func(1,torchXY).detach().numpy()
    ZX = []
    ZY = []
    for ix in range(total):
        for iy in range(total):
            ZX.append(f(x[ix],y[iy])[0])
            ZY.append(f(x[ix],y[iy])[1])
    ZX = np.reshape(ZX,(total, total)).T
    ZY = np.reshape(ZY,(total, total)).T

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
    # plt.savefig(folder+"/ODEAmyloid")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, ZX, 50, cmap='binary')
    # ax.plot_wireframe(X, Y, ZX, color='black')
    ax.plot_surface(X, Y, ZY, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Amyloid')
    ax.set_ylabel('Tau')
    ax.set_zlabel('Rate of Change')
    ax.set_title('ODE Values of Tau')
    # plt.savefig(folder+"/ODETau")
    plt.show()

def ODEslide(cutPointX,cutpointY):
    def f(x, y):
        torchXY = torch.tensor([x,y]).float()
        return func(1,torchXY).detach().numpy()
    Zx=[]
    for i in range(total):
        Zx.append(f(cutPointX,y[i])[0])

    # if __name__ == "__main__":
    fig = plt.figure()
    plt.plot(y,Zx)
    plt.xlabel('Tau')
    plt.ylabel('Rate of Change')
    plt.title('ODE Values of Amyloid When Amyloid is 0')
    plt.savefig(folder + "/ODEAmyloid_Amy0")
    plt.show()

    Zy=[]
    for i in range(total):
        Zy.append(f(x[i],cutpointY)[1])
    fig = plt.figure()
    plt.plot(x,Zy)
    plt.xlabel('Amyloid')
    plt.ylabel('Rate of Change')
    plt.title('ODE Values of Tau When Tau is 0')
    plt.savefig(folder + "/ODETau_Tau0")
    plt.show()

def ODEslideComb(cutPointX,cutpointY,folder1,folder2):
    func1 = torch.load(osp.join(folder1, 'trained_model_0.pth')).to(device)
    func2 = torch.load(osp.join(folder2, 'trained_model_0.pth')).to(device)

    def f(x, y,func):
        torchXY = torch.tensor([x,y]).float()
        return func(1,torchXY).detach().numpy()

    Zx1=[]
    for i in range(total):
        Zx1.append(f(cutPointX,y[i],func1)[0])
    Zx2=[]
    for i in range(total):
        Zx2.append(f(cutPointX,y[i],func2)[0])
    Zy1 = []
    for i in range(total):
        Zy1.append(f(x[i], cutpointY,func1)[1])
    Zy2 = []
    for i in range(total):
        Zy2.append(f(x[i], cutpointY,func2)[1])

    fig = plt.figure()
    plt.plot(y,Zx1,label="APOE4 non-carrier", c='#FF8000')
    plt.plot(y,Zx2,label="APOE4 carrier", c='#FF8000',linestyle='dashed')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Tau')
    plt.ylabel('Rate of Change')
    plt.title('ODE Values of Amyloid When Amyloid is 0')
    plt.legend()
    plt.savefig(folder1 + "/ODEAmyloid_Amy0_Comb")
    # plt.show()

    fig = plt.figure()
    plt.plot(y,Zy1,label="APOE4 non-carrier", c='#27408B')
    plt.plot(y,Zy2,label="APOE4 carrier", c='#27408B',linestyle='dashed')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Amyloid')
    plt.ylabel('Rate of Change')
    plt.title('ODE Values of Tau When Tau is 0')
    plt.legend()
    plt.savefig(folder1 + "/ODETau_Tau0_Comb")
    # plt.show()

if  __name__ == "__main__":
    folder1 = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_intial1_appfunc_3Relu_decay4_APOE0_finalized_v2'
    folder2 = '/N/slate/liyuny/cnode_ffr_main/results/meta_whole_lr4_intial1_appfunc_3Relu_decay4_APOE12_finalized'
    ODEslideComb(0, 0,folder1,folder2)
