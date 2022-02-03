import numpy as np
import matplotlib.pyplot as plt
import torch

# folder = './results/deterministic_lv/vnode/test_exp/'
#
# datasets = np.load(folder+'trueData_0.npy',allow_pickle=True)
# for i,item in enumerate(datasets):
#     true,obs,y = item
#
# true,obs,y=datasets[:]
# timeS, x_obsS, x_trueS = datasets[0][1]
# time,x_true = datasets[0][0]
# plt.plot(time.numpy(), x_true.numpy()[:, 0])
# plt.plot(time.numpy(), x_true.numpy()[:, 1])
# plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 0])
# plt.scatter(timeS.numpy(), x_obsS.numpy()[:, 1])
# plt.show()

from joblib import Parallel, delayed

def process(i):
    return i * i

results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
print(results)
