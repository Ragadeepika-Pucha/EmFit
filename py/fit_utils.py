"""
This script consists of utility functions for emission-line fitting related stuff.

Author : Ragadeepika Pucha
Version : 2023, March 24
"""

###################################################################################################

import numpy as np

from astropy.table import Table
import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

import matplotlib.pyplot as plt

###################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':22,
    'axes.linewidth':2.0,
    'xtick.major.size':6.0,
    'xtick.minor.size':4.0,
    'xtick.major.width':2.0,
    'xtick.minor.width':1.5,
    'xtick.direction':'in', 
    'xtick.minor.visible':True,
    'xtick.top':True,
    'ytick.major.size':6.0,
    'ytick.minor.size':4.0,
    'ytick.major.width':2.0,
    'ytick.minor.width':1.5,
    'ytick.direction':'in', 
    'ytick.minor.visible':True,
    'ytick.right':True
}

plt.rcParams.update(**settings)

###################################################################################################
    
def compute_aon_emline(lam_rest, flam_rest, ivar_rest, model, emline):
    
    if (emline == 'hb'):
        noise_lam = ((lam_rest >= 4700) & (lam_rest <= 4800))|((lam_rest >= 4920)&(lam_rest <= 4935))
    elif (emline == 'sii'):
        noise_lam = ((lam_rest >= 6650)&(lam_rest <= 6690))|((lam_rest >= 6760)&(lam_rest <= 6800))
    elif (emline == 'oiii'):
        noise_lam = ((lam_rest >= 4900)&(lam_rest <= 4935))|((lam_rest >= 5050)&(lam_rest <= 5100))
    elif (emline == 'nii_ha'):
        noise_lam = ((lam_rest >= 6330)&(lam_rest <= 6450))|((lam_rest >= 6650)&(lam_rest <= 6690))
        
    lam_region = lam_rest[noise_lam]
    flam_region = flam_rest[noise_lam]
    model_region = model(lam_region)
    
    res = flam_region - model_region
    noise = np.std(res)
    
    n_models = model.n_submodels
    
    if (n_models > 1):
        names_models = model.submodel_names
        aon_vals = dict()
        for name in names_models:
            aon = model[name].amplitude/noise
            aon_vals[name] = aon
    else:
        name = model.name
        
        aon_vals = dict()
        aon = model.amplitude/noise
        aon_vals[name] = aon
    
   
    return (aon_vals)
        
####################################################################################################

def get_params(gfit):
    mean = gfit.mean.value
    stddev = gfit.stddev.value
    amplitude = gfit.amplitude.value
    flux = compute_emline_flux(amplitude, stddev)
    sigma = lamspace_to_velspace(stddev, mean)
    
    return (amplitude, mean, stddev, sigma, flux)

####################################################################################################

    

