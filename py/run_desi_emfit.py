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

## Empty arrays for collecting data related to galaxies
targets_list, t_ins = [], []
flam_arr, ivar_arr, ebv_arr, tgt_arr, rsigma_arr, res_arr = [], [], [], [], [], []
t_fmeta, models_arr = [], np.ndarray((0,3,7781))
t_outs = []
count = 0

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
                count = count+len(t_hpx)

                targets = t_hpx['TARGETID'].data
    
                ## Collect targets
                targets_list.append((specprod, survey, program, hpx, targets))
                t_ins.append(t_hpx)

                if ((count >= 5120)|((specprod == specprods[-1])&(survey == surveys[-1])&(program == programs[-1])&(hpx == healpix[-1]))):
                    print (f'================================= Fetching the Spectra of {count:4d} galaxies from {len(targets_list):3d} files ============================================')

                    spec_start = time.time()

                    pool = Pool(processes = 128)
                    spec_results = pool.starmap(spec_utils.get_spectra_fastspec_data, targets_list)

                    spec_end = time.time()
                    print (f'==================================== Fetching Spectra Time: {round(spec_end - spec_start, 2):4.2f} sec =====================================================')

                    ## Collect all the spectra information
                    for result in spec_results:
                        coadd_spec, fmeta, models = result

                        lam = coadd_spec.wave['brz']
                        flams = coadd_spec.flux['brz']
                        ivars = coadd_spec.ivar['brz']
                        ebvs = coadd_spec.fibermap['EBV'].data
                        tgt_ids = coadd_spec.target_ids().data
                        rsigmas = spec_utils.compute_resolution_sigma(coadd_spec)
                        res_matrices = coadd_spec.R['brz']

                        ## Collect data
                        t_fmeta.append(fmeta)
                        models_arr = np.concatenate([models_arr, models])
                        flam_arr.extend(flams)
                        ivar_arr.extend(ivars)
                        ebv_arr.extend(ebvs)
                        tgt_arr.extend(tgt_ids)
                        rsigma_arr.extend(rsigmas)
                        res_arr.extend(res_matrices)

                    ## Sort the t_in, fmeta, and models_arr to match the tgt_arr array
                    t_in = vstack(t_ins)
                    fmeta = vstack(t_fmeta)

                    id_to_index = {tgt: ii for ii, tgt in enumerate(tgt_arr)}
                    t_in_sorted = t_in[np.argsort([id_to_index[tgt] for tgt in t_in['TARGETID'].data])]

                    ## Sorted indices for fmeta
                    sorted_indices = np.argsort([id_to_index[tgt] for tgt in fmeta['TARGETID'].data])
                    models_arr_sorted = models_arr[sorted_indices, :, :]

                    run_start = time.time()

                    print (f'==================================== Fitting {len(t_in_sorted):4d} galaxies ==============================================================')

                    ## Multiprocessing 
                    args = [(t_in_sorted[kk], models_arr_sorted[kk], lam, flam_arr[kk], \
                             ivar_arr[kk], ebv_arr[kk], rsigma_arr[kk], res_arr[kk]) for kk in range(len(t_in_sorted))]
                    pool = Pool(processes = 128)
                    t_fit = vstack(pool.starmap(emfit.fit_single_spectrum, args))
                    pool.close()
                    pool.join()
            
                    t_outs.append(t_fit)

                    ## Reset arrays and numbers
                    targets_list, t_ins = [], []
                    flam_arr, ivar_arr, ebv_arr, tgt_arr, rsigma_arr, res_arr = [], [], [], [], [], []
                    t_fmeta, models_arr = [], np.ndarray((0,3,7781))

                    count = 0

                    run_end = time.time()
                    print (f'==================================== Fitting Time: {round(run_end - run_start, 2):4.2f} sec ============================================================')
                    print ('=========================================================================================================================')

## Join all tables
t_final = vstack(t_outs)
print (f'================================== Total ========== {len(t_final):5d} galaxies ======================================================')       

t_final.write(outfile, overwrite = True)

end = time.time()
print ('Time taken: ', round(end-start, 2), 'sec')

####################################################################################################


