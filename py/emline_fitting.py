"""
This script consists of functions related to fitting the emission line spectra, 
and plotting the models and residuals.

Author : Ragadeepika Pucha
Version : 2023, March 29
"""

####################################################################################################

import numpy as np

import fit_utils, spec_utils
import fit_lines
import measure_fits as mfit
import emline_params as emp

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


def fit_emline_spectra(specprod, survey, program, healpix, targetid, z):
    """
    Fit [SII], Hb, [OIII], [NII]+Ha emission lines for a given emission line spectra.
    
    Parameters
    ----------
    pecprod : str
        Spectral Production Pipeline name fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : str
        Healpix number of the target
        
    targetid : int64
        The unique TARGETID associated with the target
        
    z : float
        Redshift of the target
        
    Returns
    -------
    fits : list
        List of astropy model fits in the order of increasing wavelength.
        Hb fit, [OIII] fit, Ha+[NII] fit, [SII] fit
        
    rchi2 : list
        List of reduced chi2 values for the fits in the order of increasing wavelength.
        rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii
    
    """
    
    lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                                   healpix, targetid, z, \
                                                                   rest_frame = True, \
                                                                   plot_continuum = False)
    
    lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                         ivar_rest, em_line = 'hb')
    lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                               ivar_rest, em_line = 'oiii')
    lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                     ivar_rest, em_line = 'nii_ha')
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                            ivar_rest, em_line = 'sii')
    
    fitter_sii, gfit_sii, rchi2_sii = fit_lines.fit_sii_lines(lam_sii, flam_sii, ivar_sii)
    fitter_oiii, gfit_oiii, rchi2_oiii = fit_lines.fit_oiii_lines(lam_oiii, flam_oiii, ivar_oiii)
    fitter_hb, gfit_hb, rchi2_hb = fit_lines.fit_hb_line(lam_hb, flam_hb, ivar_hb)
    
    if (gfit_sii.n_submodels == 2):
        fitter_nii_ha,\
        gfit_nii_ha,\
        rchi2_nii_ha = fit_lines.fit_nii_ha_lines_template(lam_nii_ha, \
                                                           flam_nii_ha, \
                                                           ivar_nii_ha, \
                                                           temp_fit = gfit_sii['sii6716'], \
                                                           frac_temp = 100)
    elif (gfit_sii.n_submodels == 4):
        fitter_nii_ha, \
        gfit_nii_ha, \
        rchi2_nii_ha = fit_lines.fit_nii_ha_lines_template(lam_nii_ha, \
                                                           flam_nii_ha, \
                                                           ivar_nii_ha, \
                                                           temp_fit = gfit_sii['sii6716'], \
                                                           temp_out_fit = gfit_sii['sii6716_out'], \
                                                           frac_temp = 100)
    
    hb_params = emp.get_hb_params(fitter_hb, gfit_hb)
    oiii_params = emp.get_oiii_params(fitter_oiii, gfit_oiii)
    nii_ha_params = emp.get_nii_ha_params_template(fitter_nii_ha, gfit_nii_ha)
    sii_params = emp.get_sii_params(fitter_sii, gfit_sii)
    
    hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                         gfit_hb, em_line = 'hb')
    oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                           gfit_oiii, em_line = 'oiii')
    nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                             gfit_nii_ha, em_line = 'nii_ha')
    sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                          gfit_sii, em_line = 'sii')
    
    tgts = [targetid, specprod, survey, program, healpix, z]
    
    row = tgts+hb_params+[hb_noise, rchi2_hb]+\
    oiii_params+[oiii_noise, rchi2_oiii]+\
    nii_ha_params+[nii_ha_noise, rchi2_nii_ha]+\
    sii_params+[sii_noise, rchi2_sii]
        
    ## All the fits and rchi2 values in increasing order of wavelength
    fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
    rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    return (fits, rchi2s, row)

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

    
    
    
    