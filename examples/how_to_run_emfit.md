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
* For running the code from Jupyter Notebook (DESI 22.5 kernel) for one source, get the above-mentioned required column data.

  ```
    import sys
    sys.path.append('$path_to_repo_directory$/py/')
    import emline_fitting as emfit
    t_params = emfit.fit_spectra(specprod, survey, program, healpix, targetid, z)
  ```
    
    
    


