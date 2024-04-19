#!/bin/bash

#SBATCH --nodes=1                            ## Number of nodes allotted for the Job
#SBATCH --ntasks=1                           ## Number of tasks per node
#SBATCH --cpus-per-task=256                  ## Number of CPUs to allot per task
#SBATCH --time=00:30:00                      ## Time for the task
#SBATCH --constraint=cpu                     ## Constraint for the Perlmutter
#SBATCH --qos=debug                          ## Job Mode: debug|interactive|regular
#SBATCH --job-name=emfit-run                 ## Job Name
###SBATCH --array=1-10                       ## Array numbers related to the job
#SBATCH --output=%x-%a.out                   ## Name of the job output file
#SBATCH --error=%x-%a.err                    ## Name of the job error file
###SBATCH --mail-type=begin,end,fail         ## Reasons to Email
###SBATCH --mail-user=email_address          ## Add your email address

./example_emfit_run.sh ${SLURM_ARRAY_TASK_ID} &
wait