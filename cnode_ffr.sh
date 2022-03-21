#!/bin/bash
# Written by Yunyi Li under MIT license: https://github.com/Yunyi-learner/master/LICENSE.md
module load python
#ROOTDIR=`readlink -f $0 | xargs dirname`/
ROOTDIR='/N/u/liyuny/Quartz/cnode_ffr_main/'
OUTDIR='/N/slate/liyuny/test'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

echo Run simulation
echo $PWD
python -m main.main1 \
  --exp_name test_100_1.0_5_5\
  --scenario simC \
  --ts_equal True \
  --num_samples 100 \
  --num_obs_x1 5 \
  --lambdaX1 2. \
  --num_obs_x2 5 \
  --lambdaX2 2. \
  --sd_u 1.0 \
  --sd_v 1.0 \
  --rho_b 0.1 \
  --rho_w 0.3 \
  --h_dim 32 \
  --epochs 501 \
  --iter_start 0 \
  --iter_end 1 \
  --outdir $OUTDIR


