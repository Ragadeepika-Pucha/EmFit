"""
This script is for running the DESI EmFit Code for a given table of sources.
It requires input and output filenames

Author : Ragadeepika Pucha
Version : 2025 April 03
"""
####################################################################################################
import sys
sys.path.append('/global/cfs/cdirs/desi/users/raga19/repos/EmFit/py/')
sys.path.append('/global/cfs/cdirs/desi/users/raga19/repos/DESI_Project/py/')

import numpy as np
import emline_fitting as emfit
import spec_utils
from astropy.table import Table, vstack

import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool
import time
####################################################################################################

## Input and Output Filenames
filename = str(sys.argv[1])
outfile = str(sys.argv[2])

## Starting timer
start = time.time()

t = Table.read(filename)

t_outs = []

## Empty arrays for collecting data related to galaxies
t_ins = []
t_fmeta = []
models_arr = np.ndarray((0,3,7781))
lam_arr = []
flam_arr = []
ivar_arr = []
ebv_arr = []
tgt_arr = []
rsigma_arr = []
res_arr = []

## Group by SPECPROD-SURVEY-PROGRAM-HEALPIX
specprods = np.unique(t['SPECPROD'].astype(str))

## Group by specprod
for specprod in specprods:
    t_spec = t[t['SPECPROD'].astype(str) == specprod]

    ## Group by surveys
    surveys = np.unique(t_spec['SURVEY'].astype(str))
    for survey in surveys:
        t_surv = t_spec[t_spec['SURVEY'].astype(str) == survey]

        ## Group by programs
        programs = np.unique(t_surv['PROGRAM'].astype(str))
        for program in programs:
            t_prog = t_surv[t_surv['PROGRAM'].astype(str) == program]

            ## Group by healpix
            healpix = np.unique(t_prog['HEALPIX'])
            for hpx in healpix:
                t_hpx = t_prog[t_prog['HEALPIX'] == hpx]

                print (f'------------------------------ {survey} {program} {hpx} ----------------------------- {len(t_hpx)} galaxies -------------------------------')

                ## List of all targets
                targets = t_hpx['TARGETID'].data

                ## Coadded-Spectra, FastSpecFit files
                coadd_spec, fmeta, models = spec_utils.get_spectra_fastspec_data(specprod, survey,\
                                                                                 program, hpx,\
                                                                                 targets)

                ## Required parameters
                lam = coadd_spec.wave['brz']
                flams = coadd_spec.flux['brz']
                ivars = coadd_spec.ivar['brz']
                ebvs = coadd_spec.fibermap['EBV'].data
                tgt_ids = coadd_spec.target_ids().data
                
                rsigmas = spec_utils.compute_resolution_sigma(coadd_spec)
                res_matrices = coadd_spec.R['brz']

                ## Collecting the data
                t_ins.append(t_hpx)
                t_fmeta.append(fmeta)
                models_arr = np.concatenate([models_arr, models])
                flam_arr.extend(flams)
                ivar_arr.extend(ivars)
                ebv_arr.extend(ebvs)
                tgt_arr.extend(tgt_ids)
                rsigma_arr.extend(rsigmas)
                res_arr.extend(res_matrices)

                if ((len(flam_arr) >= 250)|((specprod == specprods[-1])&(survey == surveys[-1])&(program == programs[-1])&(hpx == healpix[-1]))):
                    ## If the number of sources crosses 250 then run the code
                    run_start = time.time()
                    t_in = vstack(t_ins)
                    fmeta = vstack(t_fmeta)

                    ## Sort the t_in, fmeta, and models_arr to match the tgt_arr array
                    ## Create a dictionary for fast lookup
                    id_to_index = {tgt : ii for ii, tgt in enumerate(tgt_arr)}
                    t_in_sorted = t_in[np.argsort([id_to_index[tgt] for tgt in t_in['TARGETID'].data])]

                    ## Sorted indices for fmeta
                    sorted_indices = np.argsort([id_to_index[tgt] for tgt in fmeta['TARGETID'].data])
                    models_arr_sorted = models_arr[sorted_indices, :, :]
                    
                    print (f'==================================== Fitting {len(t_in)} galaxies ==============================================================')

                    ## Multiprocessing 
                    args = [(t_in[kk], models_arr[kk], tgt_arr[kk], lam, flam_arr[kk], \
                             ivar_arr[kk], ebv_arr[kk], rsigma_arr[kk], res_arr[kk]) for kk in range(len(t_in))]
                    pool = Pool(processes = 256)
                    t_fit = vstack(pool.starmap(emfit.fit_single_spectrum, args))
                    pool.close()
                    pool.join()
            
                    t_outs.append(t_fit)
            
                    ## Reset the arrays
                    t_ins = []
                    t_fmeta = []
                    models_arr = np.ndarray((0,3,7781))
                    lam_arr = []
                    flam_arr = []
                    ivar_arr = []
                    ebv_arr = []
                    tgt_arr = []
                    rsigma_arr = []
                    res_arr = []

                    run_end = time.time()
                    print (f'==================================== Fitting Time: {round(run_end - run_start, 2)} sec =========================================')

## Join all tables
t_final = vstack(t_outs)
print (f'================================== Total ================================== {len(t_final)} galaxies ================================================')        

t_final.write(outfile, overwrite = True)

end = time.time()
print ('Time taken: ', round(end-start, 2), 'sec')

####################################################################################################


