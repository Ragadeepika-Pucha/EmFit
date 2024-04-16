# Running EmFit

You can run EmFit on your table of sources through the following steps:
* Clone the EmFit Repo to the directory of your choice.
* EmFit works only with DESI 22.5 kernel at the moment.
* The following columns are required for running EmFit:
    - TARGETID
    - SPECPROD
    - SURVEY
    - PROGRAM
    - HEALPIX
    - Z
 
## From Terminal
```
cd
source .bashrc
cd $WORKING_DIRECTORY

source /global/cfs/cdirs/desi/software/desi_environment.sh 22.5

python $PATH_TO_EMFIT_DIR/py/run_desi_emfit.py INPUT_FILE_NAME OUTPUT_FILE_NAME
```

## From Jupyter Notebook
You can run the code from a Jupyter Notebook for one source at a time via the following code. 
Change the kernel to DESI 22.5
```
import sys
sys.path.append('$PATH_TO_EMFIT_DIR/py/')
import emline_fitting as emfit

t_params = emfit.fit_spectra(specprod, survey, program, healpix, targetid, z)
```

## Using SBATCH Files
This directory contains example files for running EmFit using SLURM:
    - `example_emfit_run.sh`
    - `example_emfit_run.sbatch`
If the tables has ~5000 sources, the code can be run in `debug` mode, which takes about 15-20 minutes.
For larger tables, use `regular` mode.

Author : Ragadeepika Pucha \
Version : 2024, April 16

    
    
    


