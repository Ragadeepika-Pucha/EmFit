"""
This script consists of functions related to fitting the emission line spectra, 
and plotting the models and residuals.

Author : Ragadeepika Pucha
Version : 2023, March 14
"""

####################################################################################################

import numpy as np

import fit_utils
import fit_lines

import matplotlib.pyplot as plt

####################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':18,
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

####################################################################################################


def fit_emline_spectra(lam_rest, flam_rest, ivar_rest):
    """
    Fit [SII], Hb, [OIII], [NII]+Ha emission lines for a given emission line spectra.
    
    Parameters
    ----------
    lam_rest : numpy array
        Restframe wavelength array of the emission-line spectra
        
    flam_rest : numpy array
        Restframe flux array of the emission-line spectra
        
    ivar_rest : numpy array
        Restframe inverse variance array of the emission-line spectra
        
    Returns
    -------
    fits : list
        List of astropy model fits in the order of increasing wavelength.
        Hb fit, [OIII] fit, Ha+[NII] fit, [SII] fit
        
    rchi2 : list
        List of reduced chi2 values for the fits in the order of increasing wavelength.
        rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii
    
    """
    
    ## Select Hb region for fitting
    lam_hb, flam_hb, ivar_hb = fit_utils.get_fit_window(lam_rest, flam_rest, \
                                                        ivar_rest, em_line = 'hb')
    
    ## Select [OIII] region for fitting
    lam_oiii, flam_oiii, ivar_oiii = fit_utils.get_fit_window(lam_rest, flam_rest, \
                                                              ivar_rest, em_line = 'oiii')
    
    ## Select [NII] + Ha region for fitting
    lam_nii_ha, flam_nii_ha, ivar_nii_ha = fit_utils.get_fit_window(lam_rest, flam_rest, \
                                                                    ivar_rest, em_line = 'nii_ha')
    
    ## Select [SII] region for fitting
    lam_sii, flam_sii, ivar_sii = fit_utils.get_fit_window(lam_rest, flam_rest, \
                                                           ivar_rest, em_line = 'sii')
    
    ## Fit [SII] emission-lines
    sii_fit, rchi2_sii = fit_lines.fit_sii_lines(lam_sii, flam_sii, ivar_sii)
    
    ## Fit [OIII] emission-lines
    oiii_fit, rchi2_oiii = fit_lines.fit_oiii_lines(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Fit Hb 
    hb_fit, rchi2_hb = fit_lines.fit_hb_line(lam_hb, flam_hb, ivar_hb)#, sii_fit['sii6716'])
    
    ## Fit [NII]+Ha
    nii_ha_fit, rchi2_nii_ha = fit_lines.fit_nii_ha_lines(lam_nii_ha, flam_nii_ha, ivar_nii_ha, \
                                                              hb_fit, sii_fit)
    
    ## All the fits and rchi2 values in increasing order of wavelength
    fits = [hb_fit, oiii_fit, nii_ha_fit, sii_fit]
    rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    return (fits, rchi2s)

####################################################################################################
    
def emline_fit_spectra(specprod, survey, program, healpix, targetid, z):
    
    
    ## Get rest-frame emission-line spectra
    lam_rest, flam_rest, ivar_rest = fit_utils.get_emline_spectra(specprod, survey, \
                                                                  program, healpix, targetid,\
                                                                  z, rest_frame = True)
    
    ## Get the fits
    fits, rchi2s = fit_emline_spectra(lam_rest, flam_rest, ivar_rest)
    
    ## Plot the fits
    plot_spectra_fits(targetid, lam_rest, flam_rest, fits, rchi2s)
    
    return (fits, rchi2s)
    
####################################################################################################

    
    
    
    