#!/bin/bash
# Written by Yunyi Li under MIT license: https://github.com/Yunyi-learner/master/LICENSE.md
ROOTDIR=`readlink -f $0 | xargs dirname`/
#ROOTDIR='/N/u/liyuny/Carbonate/cnode_ffr-main/'
OUTDIR='/N/u/liyuny/Carbonate/cnode_ffr_main'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

module load python/3.6.8

echo Run simulation
python -m main.main \
  --exp_name test_b \
  --scenario simA \
  --sd_u 0.3 \
  --sd_v 0.3 \
  --rho 0.9 \
  --h_dim 32 \
  --epochs 2000 \
  --rep 1 \
  --outdir $OUTDIR \


