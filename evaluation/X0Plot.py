import numpy as np
import matplotlib.pyplot as plt
import torch
# Plot Application Study
def appPlot(folder):
    # folder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE0_7var_earlystop5e4_block_random/1'

    data = np.load(folder+'/x0Record.npy',allow_pickle=True)
    color = ['#FF8000','#8B8378','red','black','green','blue','pink']
    legends = ["Amyloid PET","Total Tau PET","ADAS13","CSF TAU","CSF PTAU","FDG","AV45"]

    data_temp = [data[:][i].detach().numpy() for i in range(data.shape[0])]
    plt.figure()
    for dim in range(7):
        plt.plot(range(data.shape[0]),np.array(data_temp)[:,dim],c=color[dim],label=legends[dim])
        # plt.show()
    plt.title("The Change of Initial Value Along the Training Process")
    plt.xlabel("Training Epochs")
    plt.ylabel("Initial Value")
    plt.legend()
    plt.savefig(folder+'/X0')

# Plot Simulation Study
def simPlot(folder):
    # folder = '/N/slate/liyuny/cnode_ffr_main/results/Train_Init/simA/test_100_0.3_10_False_0.2_0.1_2.0'

    data = torch.load(folder+'/x0Record.pt')
    color = ['orange','c']
    legends = ["X1 Initial Value Estimate","X2 Initial Value Estimate"]

    data_temp = [data[:][i].detach().numpy() for i in range(len(data))]
    plt.figure()
    for dim in range(2):
        plt.plot(range(len(data)),np.array(data_temp)[:,dim],c=color[dim],label=legends[dim])
        # plt.show()
    plt.axhline(y=1, color='r', linestyle='-',label="True Initial Value")
    plt.title("The Change of Initial Value Along the Training Process")
    plt.xlabel("Training Epochs")
    plt.ylabel("Initial Value")
    plt.legend()
    plt.savefig(folder+'/X0')

if __name__=='__main__':
    folder = '/Users/yunyili/Library/CloudStorage/Dropbox/IN/Dissertation/Paper1/github/cnode_ffr_main/results/deterministic_lv/simA/104_0.3_5_False_0.3_0.1_2.0'
    simPlot(folder)