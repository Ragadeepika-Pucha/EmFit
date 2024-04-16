#!/bin/bash

cd
source .bashrc
cd $PATH_TO_WORKING_DIR

source /global/cfs/cdirs/desi/software/desi_environment.sh 22.5

python $PATH_TO_EMFIT_DIR/py/run_desi_emfit.py INPUT_FILE_NAME OUTPUT_FILE_NAME