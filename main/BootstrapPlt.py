from models.NODEmodels import *
import matplotlib.pyplot as plt


def BootstrapPlt(var,color,label):
    folder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE0_7var_earlystop5e4_block_random/1'
    # plot the fitted curve
    time = np.load(folder + '/timeX_.npy')
    predX = np.load(folder + '/predX_9999_299000.npy', allow_pickle=True)
    minT=min(time)
    maxT=max(time)
    Xfull = torch.tensor(np.linspace(minT, maxT, 100))
    # time = np.linspace(minT,30,100)
    # func = torch.load(folder+'/trained_model_0.pth')
    # predAb = odeint(func, torch.tensor(predX[0,:]), torch.tensor(time))

    plt.figure()
    # plt.plot(time, predAb[:, var].detach().numpy(), c=color, linewidth=1.5, alpha=1.0,label=label)
    plt.plot(Xfull, predX[:, var], c=color, linewidth=1.5, alpha=1.0,label=label)
    # plt.plot(Xfull, predX[:, 1], c='#8B8378', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: Total Tau PET")
    # plt.plot(Xfull, predX[:, 2], c='red', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: ADAS13")
    # plt.plot(Xfull, predX[:, 3], c='black', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF TAU")
    # plt.plot(Xfull, predX[:, 4], c='green', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF PTAU")
    # plt.plot(Xfull, predX[:, 5], c='blue', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: FDG")
    # plt.plot(Xfull, predX[:, 6], c='pink', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: AV45")
    plt.ylim(-4,4)
    # plot bootstrap predicted X values
    Nboots = 100
    Iterfolder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE0_7var_early5e4_bootstrap/'
    for i in range(Nboots):
        time = np.load(Iterfolder +str(i+1)+ '/timeX_.npy')
        predX = np.load(Iterfolder + str(i+1)+'/predX_9999_299000.npy', allow_pickle=True)
        minT = min(time)
        # maxT = max(time)
        Xfull = torch.tensor(np.linspace(minT, maxT, 100))
        # time = np.linspace(minT, 30, 100)
        funcboot = torch.load(Iterfolder +str(i+1)+ '/trained_model_0.pth')
        predboot = odeint(funcboot, torch.tensor(predX[0, :]), Xfull)
        plt.plot(Xfull,predboot[:, var].detach().numpy(),c=color,linewidth=0.3,alpha=0.4)
        # plt.plot(Xfull,predX[:, var],c=color,linewidth=0.3,alpha=0.4)
        # plt.plot(Xfull,predX[:, 1],c='#8B8378',linewidth=0.3,alpha=0.4)
        # plt.plot(Xfull, predX[:, 2], c='red', linewidth=0.3, alpha=0.4)
        # plt.plot(Xfull, predX[:, 3], c='black', linewidth=0.3, alpha=0.4)
        # plt.plot(Xfull, predX[:, 4], c='green', linewidth=0.3, alpha=0.4)
        # plt.plot(Xfull, predX[:, 5], c='blue', linewidth=0.3, alpha=0.4)
        # plt.plot(Xfull, predX[:, 6], c='pink', linewidth=0.3, alpha=0.4)
    plt.title("100 Bootstrap Fitted Curves")
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    plt.legend()
    # plt.show()
    plt.savefig(Iterfolder + "/BootstrapPlt_v1_"+str(var))

def BootstrapPltCI(var,color,label):
    folder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE12_7var/1'
    # plot the fitted curve
    time = np.load(folder + '/timeX_.npy')
    predX = np.load(folder + '/predX_5000.npy', allow_pickle=True)
    minT=min(time)
    maxT=max(time)
    Xfull = torch.tensor(np.linspace(minT, maxT, 100))
    time = np.linspace(minT,maxT,100)
    func = torch.load(folder+'/trained_model_0.pth')
    predAb = odeint(func, torch.tensor(predX[0,:]), torch.tensor(time))

    plt.figure()
    plt.plot(time, predAb[:, var].detach().numpy(), c=color, linewidth=1.5, alpha=1.0,label=label)
    # plt.plot(Xfull, predX[:, var], c=color, linewidth=1.5, alpha=1.0,label=label)
    # plt.plot(Xfull, predX[:, 1], c='#8B8378', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: Total Tau PET")
    # plt.plot(Xfull, predX[:, 2], c='red', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: ADAS13")
    # plt.plot(Xfull, predX[:, 3], c='black', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF TAU")
    # plt.plot(Xfull, predX[:, 4], c='green', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF PTAU")
    # plt.plot(Xfull, predX[:, 5], c='blue', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: FDG")
    # plt.plot(Xfull, predX[:, 6], c='pink', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: AV45")
    plt.ylim(-2.5,7)
    # plot bootstrap predicted X values
    Nboots = 100
    Iterfolder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE0_finalized_7var_bootstrap/'
    predbootAll = []
    for i in range(Nboots):
        time = np.load(Iterfolder +str(i+1)+ '/timeX_.npy')
        predX = np.load(Iterfolder + str(i+1)+'/predX_5000.npy', allow_pickle=True)
        minT = min(time)
        # maxT = max(time)
        Xfull = torch.tensor(np.linspace(minT, maxT, 100))
        time = np.linspace(minT, maxT, 100)
        funcboot = torch.load(Iterfolder +str(i+1)+ '/trained_model_0.pth')
        predboot = odeint(funcboot, torch.tensor(predX[0, :]), torch.tensor(time))
        predbootAll.append(predboot)

    predAll = torch.stack(predbootAll,0)
    upper = torch.quantile(predAll,0.975,dim=0)
    lower = torch.quantile(predAll,0.025,dim=0)

    plt.plot(time,lower[:, var].detach().numpy(),c=color,linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, var].detach().numpy(),c=color,linewidth=1.5,linestyle='dashed')
    # plt.plot(Xfull,predX[:, var],c=color,linewidth=0.3,alpha=0.4)
    # plt.plot(Xfull,predX[:, 1],c='#8B8378',linewidth=0.3,alpha=0.4)
    # plt.plot(Xfull, predX[:, 2], c='red', linewidth=0.3, alpha=0.4)
    # plt.plot(Xfull, predX[:, 3], c='black', linewidth=0.3, alpha=0.4)
    # plt.plot(Xfull, predX[:, 4], c='green', linewidth=0.3, alpha=0.4)
    # plt.plot(Xfull, predX[:, 5], c='blue', linewidth=0.3, alpha=0.4)
    # plt.plot(Xfull, predX[:, 6], c='pink', linewidth=0.3, alpha=0.4)
    plt.title("100 Bootstrap Fitted Curves")
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    # plt.legend()
    # plt.show()
    plt.savefig(Iterfolder + "/BootstrapPlt_inter_ci_"+str(var))

def BootstrapPltCIall():
    folder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE12_7var/1/1'
    # plot the fitted curve
    time = np.load(folder + '/timeX_.npy')
    predX = np.load(folder + '/predX_5000.npy', allow_pickle=True)
    minT=min(time)
    maxT=max(time)
    Xfull = torch.tensor(np.linspace(minT, maxT, 100))
    time = np.linspace(minT,maxT,100)
    func = torch.load(folder+'/trained_model_0.pth')
    predAb = odeint(func, torch.tensor(predX[0,:]), torch.tensor(time))

    plt.figure()
    # plt.plot(time, predAb[:, var].detach().numpy(), c=color, linewidth=1.5, alpha=1.0,label=label)
    # plt.plot(Xfull, predX[:, var], c=color, linewidth=1.5, alpha=1.0,label=label)
    plt.plot(time, predAb[:, 0].detach().numpy(), c='#FF8000', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: Amyloid PET")
    plt.plot(time, predAb[:, 1].detach().numpy(), c='#8B8378', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: Total Tau PET")
    plt.plot(time, predAb[:, 2].detach().numpy(), c='red', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: ADAS13")
    plt.plot(time, predAb[:, 3].detach().numpy(), c='black', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF TAU")
    plt.plot(time, predAb[:, 4].detach().numpy(), c='green', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: CSF PTAU")
    plt.plot(time, predAb[:, 5].detach().numpy(), c='blue', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: FDG")
    plt.plot(time, predAb[:, 6].detach().numpy(), c='pink', linewidth=1.5, alpha=1.0,label="APOE4 non-carrier: AV45")
    plt.ylim(-2.5,7)
    # plot bootstrap predicted X values
    Nboots = 100
    Iterfolder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/meta_whole_lr4_intial1_appfunc_3Relu_hdim32_decay3_APOE0_finalized_7var_bootstrap/'
    predbootAll = []
    for i in range(Nboots):
        time = np.load(Iterfolder +str(i+1)+ '/timeX_.npy')
        predX = np.load(Iterfolder + str(i+1)+'/predX_5000.npy', allow_pickle=True)
        minT = min(time)
        # maxT = max(time)
        Xfull = torch.tensor(np.linspace(minT, maxT, 100))
        time = np.linspace(minT, maxT, 100)
        funcboot = torch.load(Iterfolder +str(i+1)+ '/trained_model_0.pth')
        predboot = odeint(funcboot, torch.tensor(predX[0, :]), torch.tensor(time))
        predbootAll.append(predboot)

    predAll = torch.stack(predbootAll,0)
    upper = torch.quantile(predAll,0.975,dim=0)
    lower = torch.quantile(predAll,0.025,dim=0)

    plt.plot(time,lower[:, 0].detach().numpy(),c='#FF8000',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 0].detach().numpy(),c='#FF8000',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 1].detach().numpy(),c='#8B8378',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 1].detach().numpy(),c='#8B8378',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 2].detach().numpy(),c='red',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 2].detach().numpy(),c='red',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 3].detach().numpy(),c='black',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 3].detach().numpy(),c='black',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 4].detach().numpy(),c='green',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 4].detach().numpy(),c='green',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 5].detach().numpy(),c='blue',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 5].detach().numpy(),c='blue',linewidth=1.5,linestyle='dashed')
    plt.plot(time,lower[:, 6].detach().numpy(),c='pink',linewidth=1.5,linestyle='dashed')
    plt.plot(time,upper[:, 6].detach().numpy(),c='pink',linewidth=1.5,linestyle='dashed')

    plt.title("100 Bootstrap Fitted Curves")
    plt.xlabel("Years From Onset")
    plt.ylabel("Linear Transformed Values")
    # plt.legend()
    # plt.show()
    plt.savefig(Iterfolder + "/BootstrapPlt_inter_ci_plt")

if __name__ == "__main__":
    # Iterfolder = '/N/slate/liyuny/cnode_ffr_main/results/AdniCogNoTime/APOE12_7var_early5e4_bootstrap/'
    var = range(7)
    color = ['#FF8000','#8B8378','red','black','green','blue','pink']
    label = ["APOE4 non carrier: Amyloid PET","APOE4 non carrier: Total Tau PET","APOE4 non carrier: ADAS13",
             "APOE4 non carrier: CSF TAU","APOE4 non carrier: CSF PTAU","APOE4 non carrier: FDG","APOE4 non carrier: AV45"]
    for i in var:
    # i=0
        BootstrapPlt(i,color[i],label[i])
    #     BootstrapPltCIall()
