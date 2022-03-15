import matplotlib.pyplot as plt
import os
import os.path as osp
import torch

folder='/N/slate/liyuny/cnode_ffr_main/results/formal_sim/simA/300_0.3_5_True_0.2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

func = torch.load(osp.join(folder, 'trained_model_0.pth')).to(device)

