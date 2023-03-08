#!/bin/bash

#Load any modules that your program needs
module load python/3.8.5
#ROOTDIR=`readlink -f $0 | xargs dirname`/
ROOTDIR='/N/u/liyuny/Quartz/cnode_ffr_main/'
OUTDIR='/N/slate/liyuny/cnode_ffr_main/results/testCov'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR
echo $PWD

scenarioIndex=(simC)
num_samplesIndex=(10)
num_obs_xIndex=(20)
sd_uvIndex=(0.3)
rhowIndex=(0.3)
rhobIndex=(0.1)
lambdaIndex=(2.0)
ts_equalIndex=(True)

for scenario in ${scenarioIndex[*]}; do
 for num_samples in ${num_samplesIndex[*]}; do
  for sd_uv in ${sd_uvIndex[*]}; do
   for num_obs_x in ${num_obs_xIndex[*]}; do
    for ts_equal in ${ts_equalIndex[*]}; do
     for rho_w in ${rhowIndex[*]}; do
      for lambda in ${lambdaIndex[*]}; do
        for rho_b in ${rhobIndex[*]}; do
        echo Run simulation
        python -m main.mainCov \
        --exp_name $num_samples'_'$sd_uv'_'$num_obs_x'_'$ts_equal'_'$rho_w'_'$rho_b'_'$lambda\
        --data 'functionalCov' \
        --model 'timevnode' \
        --scenario $scenario \
        --ts_equal $ts_equal \
        --num_samples $num_samples \
        --num_obs_x1 $num_obs_x \
        --lambdaX1 $lambda \
        --num_obs_x2 $num_obs_x \
        --lambdaX2 $lambda \
        --sd_u $sd_uv \
        --sd_v $sd_uv \
        --rho_w $rho_w \
        --rho_b $rho_b \
        --h_dim 32 \
        --epochs 5000 \
        --ifplot False \
        --iter_start 0 \
        --iter_end 1 \
        --outdir $OUTDIR
        done
      done
     done
    done
   done
  done
 done
done
