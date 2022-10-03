import argparse
import os
import os.path as osp
import json
import numpy as np
from evaluation.eval import final_eval_range
from models.utils import obsPercent as op

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name', type=str,required=True)
parser.add_argument('--newrange', type=int)
parser.add_argument('--scenario', type=str, choices=['simA','simB','simC','simB2'],default='simA') # simulation scenario
parser.add_argument('--outdir',type=str,default='/N/slate/liyuny/cnode_ffr_main/results/formal_sim/lambda_4.0')
argsOut = parser.parse_args()

if __name__ == "__main__":
    folder = osp.join(argsOut.outdir, argsOut.scenario, argsOut.exp_name)
    outdir = osp.join(argsOut.outdir, argsOut.scenario, argsOut.exp_name,'eval'+str(argsOut.newrange))

    with open(osp.join(folder, 'args.txt')) as f:
        args = json.load(f)
    sdense = np.linspace(0, argsOut.newrange, 100)
    op(folder,args['iter_end'],args['ts_equal'],argsOut.newrange,folder)

