#!/bin/bash

#SBATCH -J simA
#SBATCH -p debug
#SBATCH -o sim_%j.txt
#SBATCH -e error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liyuny@iu.edu
#SBATCH --nodes=1
#SBATCH --exclude=c2
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=128
#SBATCH -A r00330

#Load any modules that your program needs
module load python
#ROOTDIR=`readlink -f $0 | xargs dirname`/
ROOTDIR='/N/slate/liyuny/Paper1/'
OUTDIR='/N/slate/liyuny/Paper1/results/notrefine/exponential'
export PYTHONPATH=$PYTHONPATH:$ROOTDIR
echo $PWD

scenarioIndex=(simA)
num_samplesIndex=(100)
num_obs_xIndex=(5)
sd_uvIndex=(0.3 1.0)
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
	srun python -m main.main \
	--exp_name $num_samples'_'$sd_uv'_'$num_obs_x'_'$ts_equal'_'$rho_w'_'$rho_b'_'$lambda\
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
	--rangeMax 15 \
	--h_dim 5 \
	--n_hiddenly 3 \
	--initialrefine False \
	--Bepochs 1 \
	--epochs 5001 \
	--iter_start 0 \
	--iter_end 100 \
	--ifplot True \
	--outdir $OUTDIR
	done
      done
     done
    done
   done
  done
 done
done
