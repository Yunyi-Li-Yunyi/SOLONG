import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
import numpy as np

from torch.distributions import Bernoulli,Normal

class VanillaODEFunc(nn.Module):
    """
    Maps times X to predictions y.
    The decoder is an ODEsolver, using torchdiffeq.

    Parameters
    ----------
    X_dim: int
        Dimension of x values. Currently only works for dimension of 1.
    h_dim: int
        Dimension of hidden layer in odefunc.
    y_dim: int
        Dimension of y values.
    """

    def __init__(self, t_dim, h_dim, y_dim, exclude_time=True):
        super(VanillaODEFunc,self).__init__()

        # X is always time.
        assert t_dim == 1

        self.exclude_time = exclude_time
        self.t_dim = t_dim # must be 1
        self.h_dim = h_dim
        self.y_dim = y_dim

        inp_dim = y_dim if exclude_time else y_dim+t_dim

        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.ReLU(),
                      # nn.Linear(h_dim,h_dim),
                      # nn.Sigmoid(),
                      nn.Linear(h_dim,h_dim),
                      nn.ReLU(),
                      nn.Linear(h_dim,y_dim)]

        self.net = nn.Sequential(*ode_layers)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.1)
                nn.init.constant_(m.bias,val=0)

    def forward(self,t,y):
        return self.net(y)

class ODEFuncTimeVariate(nn.Module):
    """
    Maps times X to predictions y.
    The decoder is an ODEsolver, using torchdiffeq.

    Parameters
    ----------
    X_dim: int
        Dimension of x values. Currently only works for dimension of 1.
    h_dim: int
        Dimension of hidden layer in odefunc.
    y_dim: int
        Dimension of y values.
    """

    def __init__(self, t_dim, h_dim, y_dim, exclude_time=False):
        super(ODEFuncTimeVariate,self).__init__()

        # X is always time.
        assert t_dim == 1

        self.exclude_time = exclude_time
        self.t_dim = t_dim # must be 1
        self.h_dim = h_dim
        self.y_dim = y_dim

        inp_dim = y_dim if exclude_time else y_dim+t_dim

        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.ReLU(),
                      # nn.Linear(h_dim,h_dim),
                      # nn.Sigmoid(),
                      nn.Linear(h_dim,h_dim),
                      nn.ReLU(),
                      nn.Linear(h_dim,y_dim)]

        self.net = nn.Sequential(*ode_layers)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.1)
                nn.init.constant_(m.bias,val=0)

    def forward(self,t,y):
        # time = t.view(1, 1,1).repeat(y.size(0), 1)
        time = t.view(1)
        yt = torch.cat((y, time), dim=0)
        return self.net(yt)

class ApplicationODEFunc(nn.Module):
    """
    Maps times X to predictions y.
    The decoder is an ODEsolver, using torchdiffeq.

    Parameters
    ----------
    t_dim: int
        Dimension of t values. Currently only works for dimension of 1.
    h_dim: int
        Dimension of hidden layer in odefunc.
    y_dim: int
        Dimension of y values.
    """

    def __init__(self, t_dim, h_dim, y_dim, exclude_time):
        super(ApplicationODEFunc,self).__init__()

        # X is always time.
        assert t_dim == 1

        self.exclude_time = exclude_time
        self.t_dim = t_dim # must be 1
        self.h_dim = h_dim
        self.y_dim = y_dim

        inp_dim = y_dim if exclude_time else y_dim+t_dim

        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.ReLU(),
                      # nn.Linear(h_dim,h_dim),
                      # nn.Sigmoid(),
                      nn.Linear(h_dim,h_dim),
                      nn.ReLU(),
                      nn.Linear(h_dim, h_dim),
                      nn.ReLU(),
                      nn.Linear(h_dim,y_dim)]


        self.net = nn.Sequential(*ode_layers)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.1)
                nn.init.constant_(m.bias,val=0)

    def forward(self,t,y):
        if self.exclude_time:
            return self.net(y)
        else:
            time = t.view(1)
            yt = torch.cat((y, time), dim=0)
            return self.net(yt)

