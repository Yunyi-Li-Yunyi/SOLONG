#!/bin/bash
# Written by Yunyi Li under MIT license: https://github.com/Yunyi-learner/master/LICENSE.md
#ROOTDIR=`readlink -f $0 | xargs dirname`/
ROOTDIR='/N/u/liyuny/Carbonate/thindrives/Dissertation/node_ffr-main'
OUTDIR='/N/u/liyuny/Carbonate/cnode_ffr_main'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

#echo $PYTHONPATH

echo Run simulation
python -m main.main \
  --exp_name test_a \
  --x1x2Ind True \
  --sd 0.5 \
  --sd_u 0.3 \
  --sd_v 0.3 \
  --rho 0.9 \
  --h_dim 32 \
  --epochs 2000 \
  --rep 25 \
  --outdir $OUTDIR \


