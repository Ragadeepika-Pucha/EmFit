"""
This script consists of functions for computing the parameters of the emission-line fits.
It consists of the following functions:
    1) get_parameters(gfit, models)
    2) get_allfit_params.normal_fit(fits, lam, flam)
    3) get_allfit_params.extreme_fit(fits, lam, flam)

Author : Ragadeepika Pucha
Version : 2024, March 10
"""

###################################################################################################
from astropy.table import Table
import numpy as np

import measure_fits as mfit
import spec_utils
from astropy.stats import sigma_clipped_stats
###################################################################################################

def get_parameters(gfit, models):
    """
    Function to get amplitude, mean, standard deviation, sigma, and flux for each of 
    model components in a given emission-line model.
    
    Parameters
    ----------
    gfit : Astropy model
        Compound model for the emission-line
        
    models : list
        List of total submodels expected from a given emission-line fitting.
        
    Returns
    -------
    params : dict
        Dictionary with the parameter values
    """
    
    params = {}
    n = gfit.n_submodels
    
    if (n > 1):
        names = gfit.submodel_names
    else:
        names = gfit.name
    
    for model in models:
        if (model in names):
            if (n == 1):
                amp, mean, std = gfit.parameters
            else:
                amp, mean, std = gfit[model].parameters
            sig = mfit.lamspace_to_velspace(std, mean)
            flux = mfit.compute_emline_flux(amp, std)
            
            params[f'{model}_amplitude'] = [amp]
            params[f'{model}_mean'] = [mean]
            params[f'{model}_std'] = [std]
            params[f'{model}_sigma'] = [sig]
            params[f'{model}_flux'] = [flux]
        else:
            params[f'{model}_amplitude'] = [0.0]
            params[f'{model}_mean'] = [0.0]
            params[f'{model}_std'] = [0.0]
            params[f'{model}_sigma'] = [0.0]
            params[f'{model}_flux'] = [0.0]
            
    return (params)
    
###################################################################################################
###################################################################################################

class get_allfit_params:
    """
    Functions to get all the parameters together.
        1) normal_fit(fits, lam, flam)
        2) extreme_fit(fits, lam, flam)
    """
    
    def normal_fit(fits, lam, flam):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] fits.
        This is for the normal source fitting method.
        
        Parameters
        ----------
        fits : list
            List of [Hb, [OIII], [NII]+Ha, [SII]] bestfits
            
        lam : array
            Wavelength array of the spectra
            
        flam : array
            Flux array of the spectra
            
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb fit, 
            followed by continuum and noise measurements.
            
        oiii_params : dict
            Gaussian parameters of the [OIII] fit, 
            followed by continuum and noise measurements.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha fit, 
            followed by continuum and noise measurements.
            
        sii_params : dict
            Gaussian parameters of the [SII] fit,
            followed by continuum and noise measurements.
        """
        
        
        gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii = fits

        ## Parameters for the fit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_parameters(gfit_hb, hb_models)
        oiii_params = get_parameters(gfit_oiii, oiii_models)
        nii_ha_params = get_parameters(gfit_nii_ha, nii_ha_models)
        sii_params = get_parameters(gfit_sii, sii_models)

        ## Continuum
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha['nii_ha_cont'].amplitude.value]
        sii_params['sii_continuum'] = [gfit_sii['sii_cont'].amplitude.value]

        ## NOISE
        hb_noise = mfit.compute_noise_emline(lam, flam, 'hb')
        oiii_noise = mfit.compute_noise_emline(lam, flam, 'oiii')
        nii_ha_noise = mfit.compute_noise_emline(lam, flam, 'nii_ha')
        sii_noise = mfit.compute_noise_emline(lam, flam, 'sii')

        hb_params['hb_noise'] = [hb_noise]
        oiii_params['oiii_noise'] = [oiii_noise]
        nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
        sii_params['sii_noise'] = [sii_noise]

        return (hb_params, oiii_params, nii_ha_params, sii_params)
    
###################################################################################################

    def extreme_fit(fits, lam, flam):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] fits.
        This is for the extreme broad-line source fitting.
        
        Parameters
        ----------
        fits : list
            List of [Hb+[OIII], [NII]+Ha+[SII]] bestfits
            
        lam : array
            Wavelength array of the spectra
            
        flam : array
            Flux array of the spectra
            
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb fit, 
            followed by continuum and noise measurements.
            
        oiii_params : dict
            Gaussian parameters of the [OIII] fit, 
            followed by continuum and noise measurements.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha fit, 
            followed by continuum and noise measurements.
            
        sii_params : dict
            Gaussian parameters of the [SII] fit,
            followed by continuum and noise measurements.
        """
        
        gfit_hb_oiii, gfit_nii_ha_sii = fits

        ## Parameters for the fit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_parameters(gfit_hb_oiii, hb_models)
        oiii_params = get_parameters(gfit_hb_oiii, oiii_models)
        nii_ha_params = get_parameters(gfit_nii_ha_sii, nii_ha_models)
        sii_params = get_parameters(gfit_nii_ha_sii, sii_models)

        ## Continuum
        hb_params['hb_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
        oiii_params['oiii_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
        nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]
        sii_params['sii_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]

        ## NOISE
        hb_noise = mfit.compute_noise_emline(lam, flam, 'hb')
        oiii_noise = mfit.compute_noise_emline(lam, flam, 'oiii')
        nii_ha_noise = mfit.compute_noise_emline(lam, flam, 'nii_ha')
        sii_noise = mfit.compute_noise_emline(lam, flam, 'sii')

        hb_params['hb_noise'] = [hb_noise]
        oiii_params['oiii_noise'] = [oiii_noise]
        nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
        sii_params['sii_noise'] = [sii_noise]

        return (hb_params, oiii_params, nii_ha_params, sii_params)

###################################################################################################
###################################################################################################
    
    
    


        
    
                     
        
        
        
    
        
        
        
        
    
    
    
    
        


    
    
    
        
        
        
