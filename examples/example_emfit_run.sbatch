#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=00:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --job-name=emfit-run
###SBATCH --array=1-10
#SBATCH --output=%x-%a.out
#SBATCH --error=%x-%a.err
###SBATCH --mail-type=begin,end,fail
###SBATCH --mail-user=email_address

./example_emfit_run.sh ${SLURM_ARRAY_TASK_ID} &
wait