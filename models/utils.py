import torch
import numpy as np
from torchdiffeq import odeint


# def SparseData(times, states_obs, states_true, num_context_range=(5, 10), locations=None):
#     """
#     Given input time x and their value y, return random
#     sparse observe time points and obsevations
#
#     Parameters
#     ----------
#     times: torch.Tensor
#         Shape(batch_size, num_points, x_dim)
#
#     states_obs: torch.Tensor
#         Shape(batch_size, num_points, y_dim)
#
#     states_true: torch.Tensor
#         Shape(batch_size, num_points, y_dim)
#
#     num_context_range: Tuple[int,int]
#         The range of number of context points.
#
#     locations: array
#         Specify locations of context points
#     """
#     num_points = times.shape[1]
#     num_samples = times.shape[0]
#
#     # Sample locations of context and target points
#     maskX = torch.zeros(times.shape)
#     maskY = torch.zeros(states_obs.shape)
#     if locations is None:
#         points = np.arange(num_points)
#         initial_loc = np.array([])
#
#         for subj in range(num_samples):
#             size = np.random.choice(range(*num_context_range), size=1, replace=True)
#             locations = np.random.choice(points, size=size, replace=False)
#             locations = np.concatenate([initial_loc, locations])
#             maskX[subj, locations, :] = 1
#             maskY[subj, locations, :] = 1
#
#     else:
#         maskX[:,locations,:]=1
#         maskY[:,locations,:]=1
#
#     maskX = maskX.ge(1)
#     maskY = maskY.ge(1)
#
#     TimesSparse = torch.clone(times)
#     TimesSparse[maskX] = float('nan')
#
#     StatesObsSparse = torch.clone(states_obs)
#     StatesObsSparse[maskY] = float('nan')
#
#     StatesTrueSparse = torch.clone(states_true)
#     StatesTrueSparse[maskY] = float('nan')
#
#     return TimesSparse,StatesObsSparse,StatesTrueSparse

def ObservedData(TimesSparse, StatesObsSparse, StatesTrueSparse):
    """
    Given Sparse inputs x and y (with nans) return combined x and y

    Parameters
    ----------
    TimesSparse: torch.Tensor
        Shape(batch_size, num_points, x_dim)
    StatesObsSparse: torch.Tensor
        Shape(batch_size, num_points, y_dim)
    StatesTrueSparse: torch.Tensor
        Shape(batch_size, num_points, y_dim)

    return: combined torch.Tensor obsX, obsY
    """
    # maskX = torch.tensor(torch.isnan(TimesSparse) == 0)
    maskX = (torch.isnan(TimesSparse) == 0).clone().detach()

    # maskY = torch.tensor(torch.isnan(StatesObsSparse) == 0)
    maskY = (torch.isnan(StatesObsSparse) == 0).clone().detach()

    obsX = torch.masked_select(TimesSparse, maskX)

    obsY = torch.masked_select(StatesObsSparse, maskY)
    obsY = torch.reshape(obsY, (-1, StatesObsSparse.shape[2]))

    trueY = torch.masked_select(StatesTrueSparse, maskY)
    trueY = torch.reshape(trueY, (-1, StatesTrueSparse.shape[2]))

    # sortX = torch.sort(obsX)[0]
    # index = torch.sort(obsX)[1]
    # print(obsX)
    # print(obsY)
    sortX, counts = torch.unique(obsX, sorted=True, return_counts=True)
    sortYobs = torch.empty((len(sortX), StatesObsSparse.shape[2]))
    sortYtrue = torch.empty((len(sortX), StatesTrueSparse.shape[2]))
    # print(counts)
    for i in range(sortX.size(0)):
        groupYobs = obsY[obsX == sortX[i]]
        groupYtrue = trueY[obsX == sortX[i]]
        sortYobs[i] = torch.mean(groupYobs, 0)
        sortYtrue[i] = torch.mean(groupYtrue, 0)

    # print(obsX)
    # print(sortX)
    # print(obsY)
    # print(sortYobs)

    return sortX, sortYobs, sortYtrue


def PredData(device,ODEFunc, initial_u, initial_v, timepoint=None):
    """
    ODEFunc: nn.Module
        deep learning ODE function
    initial_u: int
        initial value of u
    initial_v: int
        initial value of v
    timepoint: torch.tensor
        timepoints at which predictions will be calculated
    steps : float
        length of interval
    end_time : float
        the final time (simulation runs from 0 to end_time)

    :return:
    timepoint, pred
    """
    # if timepoint == None:
    #     x = np.arange(0, end_time, by)
    #     x = torch.tensor(x)
    # else:
    s = torch.tensor(timepoint).to(device)
    x0 = torch.tensor([initial_u, initial_v]).to(device)
    predX = odeint(ODEFunc, x0, s).to(device)
    return s, predX

# if __name__=='__main__':
#     x = torch.randn(3, 5,1)
#     y = torch.randn(3, 5,2)
#     sparseX,sparseYobs, sparseYtrue = SparseData(x,y,y,(3,4))
#
#     sortX, sortYobs, sortYtrue = ObservedData(sparseX, sparseYobs,sparseYtrue)
#     print(sortX)
#     print(sortYobs)
