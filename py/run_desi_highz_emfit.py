"""
This script is for running the DESI EmFit High-z code for a given table of sources.
Only works for Hb for now. It requires input and output filenames

Author : Ragadeepika Pucha
Version : 2024 August 30
"""
####################################################################################################
import sys
sys.path.append('/global/cfs/cdirs/desi/users/raga19/repos/EmFit/py/')
sys.path.append('/global/cfs/cdirs/desi/users/raga19/repos/DESI_Project/py/')

import numpy as np
import emline_fitting_highz as emfit_highz
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

pool = Pool(processes = 128)
inputs = [(obj['SPECPROD'], obj['SURVEY'], obj['PROGRAM'], obj['HEALPIX'],\
           obj['TARGETID'], obj['Z']) for obj in t]
t_final = vstack(pool.starmap(emfit_highz.fit_highz_spectra_fixed_hb.fit_highz_spectra, inputs))
pool.close()
pool.join()

t_final.write(outfile, overwrite = True)

end = time.time()
print ('Time taken: ', round(end-start, 2), 'sec')

####################################################################################################