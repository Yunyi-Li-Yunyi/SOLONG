#!/bin/bash
# Written by Yunyi Li under MIT license: https://github.com/Yunyi-learner/master/LICENSE.md
module load python/3.6.8
#ROOTDIR=`readlink -f $0 | xargs dirname`/
ROOTDIR='/N/u/liyuny/Carbonate/cnode_ffr_main/'
OUTDIR='/N/u/liyuny/Carbonate/cnode_ffr_main'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

module load python/3.6.8

echo Run simulation
python -m main.main \
  --exp_name test_sim_b2_0.3_5_5 \
  --scenario simB2 \
  --ts_equal False \
  --num_obs_x1 5 \
  --lambdaX1 2. \
  --num_obs_x2 5 \
  --lambdaX2 2. \
  --sd_u 0.3 \
  --sd_v 0.3 \
  --rho 0.9 \
  --h_dim 32 \
  --epochs 501 \
  --rep 1 \
  --outdir $OUTDIR \


