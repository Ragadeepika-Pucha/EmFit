#!/bin/bash

cd
source .bashrc
## Change $PATH_TO_WORKING_DIR --> Path to the directory where the EmFit is being run
cd $PATH_TO_WORKING_DIR          

source /global/cfs/cdirs/desi/software/desi_environment.sh 22.5

## Change $PATH_TO_EMFIT_DIR --> Path to the EmFit repository in your system
python $PATH_TO_EMFIT_DIR/py/run_desi_emfit.py INPUT_FILE_NAME OUTPUT_FILE_NAME       