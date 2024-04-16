## Running EmFit

You can run EmFit on your table of sources through the following steps:
* Clone the EmFit Repo to the directory of your choice.
* EmFit works only with DESI 22.5 kernel at the moment.
* Create the list of sources as an Astropy Table with the following compulsory columns:
    - TARGETID
    - SPECPROD
    - SURVEY
    - PROGRAM
    - HEALPIX
    - Z
* For running the code from terminal:
    `path_to_repo_directory$/py/run_desi_emfit.py input_file output_file`
* For running the code from Jupyter Notebook (DESI 22.5 kernel) for one source, get the above-mentioned required column data.
    import sys
    sys.path.append('$path_to_repo_directory$/py/')
    import emline_fitting as emfit
    t_params = emfit.fit_spectra(specprod, survey, program, healpix, targetid, z)
    
    
    


