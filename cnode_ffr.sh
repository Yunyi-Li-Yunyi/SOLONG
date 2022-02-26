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
  --exp_name 100_0.3_10_10\
  --num_samples 100 \
  --scenario simA \
  --ts_equal True \
  --num_obs_x1 10 \
  --lambdaX1 2. \
  --num_obs_x2 10 \
  --lambdaX2 2. \
  --sd_u 0.3 \
  --sd_v 0.3 \
  --rho 0.9 \
  --h_dim 32 \
  --epochs 501 \
  --rep 1000 \
  --outdir $OUTDIR \


