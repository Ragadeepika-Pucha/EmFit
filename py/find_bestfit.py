"""
This script consists of functions for fitting emission-lines. 
The different functions are divided into different classes for different emission lines.

Author : Ragadeepika Pucha
Version : 2023, April 25
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import fit_utils
import measure_fits as mfit
import fit_lines as fl

###################################################################################################

def find_sii_best_fit(lam_sii, flam_sii, ivar_sii):
    """
    Find the best fit for [SII]6716,6731 doublet.
    The code fits both one-component and two-component fits and picks the best version.
    The two-component fit needs to be >20% better to be picked.

    Parameters
    ----------
    lam_sii : numpy array
        Wavelength array of the [SII] region where the fits need to be performed.

    flam_sii : numpy array
        Flux array of the spectra in the [SII] region.

    ivar_sii : numpy array
        Inverse variance array of the spectra in the [SII] region.

    Returns
    -------
    gfit : Astropy model
        Best-fit 1 component or 2 component model

    rchi2: float
        Reduced chi2 of the best-fit
    """

    ## Single-component fits
    gfit_1comp, rchi2_1comp = fl.fit_sii_lines.fit_one_component(lam_sii, flam_sii, ivar_sii)

    ## Two-component fits
    gfit_2comp, rchi2_2comp = fl.fit_sii_lines.fit_two_components(lam_sii, flam_sii, ivar_sii)

    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100

    ## Also Amp ([SII]) > Amp ([SII]; out)
    amp_sii6716 = gfit_2comp['sii6716'].amplitude.value
    amp_sii6716_out = gfit_2comp['sii6716_out'].amplitude.value
    
    ## Also Sig ([SII]; out) > Sig ([SII])
    sig_sii = mfit.lamspace_to_velspace(gfit_2comp['sii6716'].stddev.value, \
                                        gfit_2comp['sii6716'].mean.value)
    sig_sii_out = mfit.lamspace_to_velspace(gfit_2comp['sii6716_out'].stddev.value, \
                                            gfit_2comp['sii6716_out'].stddev.value)

    if ((del_rchi2 >= 20)&(sig_sii_out > sig_sii)&(amp_sii6716 > amp_sii6716_out)):
        return (gfit_2comp, rchi2_2comp)
    else:
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################
####################################################################################################

def find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii):
    """
    Find the best fit for [OIII]4959,5007 doublet.
    The code fits both one-component and two-component fits and picks the best version.
    The two-component fit needs to be >20% better to be picked.

    Parameters
    ----------
    lam_oiii : numpy array
        Wavelength array of the [OIII] region where the fits need to be performed.

    flam_oiii : numpy array
        Flux array of the spectra in the [OIII] region.

    ivar_oiii : numpy array
        Inverse variance array of the spectra in the [OIII] region.

    Returns
    -------
    gfit : Astropy model
        Best-fit 1 component or 2 component model

    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    ## Single component fit
    gfit_1comp, rchi2_1comp = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Two-component fit
    gfit_2comp, rchi2_2comp = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Select the best fit based on rchi2
    ## Rchi2 of the 2-componen is improved by 20%, then the 2-component fit is picked
    ## Otherwise, 1-component fit is the best fit
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100
    
    ## Extra criterion - 
    ## Amp ([OIII]) > Amp([OIII]; out)
    ## Sigma ([OIII]) < Sigma ([OIII]; out)
    amp_oiii5007 = gfit_2comp['oiii5007'].amplitude.value
    amp_oiii5007_out = gfit_2comp['oiii5007_out'].amplitude.value
    
    sig_oiii = mfit.lamspace_to_velspace(gfit_2comp['oiii5007'].stddev.value, \
                                         gfit_2comp['oiii5007'].mean.value)
    sig_oiii_out = mfit.lamspace_to_velspace(gfit_2comp['oiii5007_out'].stddev.value, \
                                             gfit_2comp['oiii5007_out'].mean.value)
    
    if ((del_rchi2 >= 20)&(sig_oiii_out > sig_oiii)&(amp_oiii5007 > amp_oiii5007_out)):
        return (gfit_2comp, rchi2_2comp)
    else:
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################
####################################################################################################

def find_hb_best_fit(lam_hb, flam_hb, ivar_hb, sii_bestfit):
    """
    Function to find the best fit for Hbeta, with or without broad-lines
    
    Parameters
    ----------
    lam_hb : numpy array
        Wavelength array of the Hbeta region where the fit needs to be performed.
        
    flam_hb : numpy array
        Flux array of the spectra in the Hbeta region
        
    ivar_hb : numpy array
        Inverse variance array of the spectra in the Hbeta region
        
    sii_bestfit : Astropy model
        Best fit model for the [SII] emission-lines
        
    Returns
    -------
    gfit : Astropy model
        Best-fit "with" or "without" broad-line model
        
    rchi2 : float
        Reduced chi2 of the best-fit
    """

    n_sii = sii_bestfit.n_submodels
    
    ## If n_sii == 2, first try free-fit model, otherwise fix the width of Hbeta to [SII]
    ## If n_sii == 4, fix the width of narrow and outflow components to [SII]
    
    if (n_sii == 2):
        gfit_free, rchi2_free = fl.fit_hb_line.fit_free_one_component(lam_hb, flam_hb, ivar_hb, \
                                                                     sii_bestfit, frac_temp = 100.)
        
        sig_sii = mfit.lamspace_to_velspace(sii_bestfit['sii6716'].stddev.value, \
                                           sii_bestfit['sii6716'].mean.value)
        
        n_hb = gfit_free.n_submodels
        
        if (n_hb == 1):
            sig_hb = mfit.lamspace_to_velspace(gfit_free.stddev.value, \
                                              gfit_free.mean.value)
        else:
            sig_hb = mfit.lamspace_to_velspace(gfit_free['hb_n'].stddev.value, \
                                              gfit_free['hb_n'].mean.value)
            
        per_diff = (sig_sii - sig_hb)*100/sig_sii
                
        if ((per_diff <= -30)|(per_diff >= 30)):
            gfit_hb, rchi2_hb = fl.fit_hb_line.fit_fixed_one_component(lam_hb, flam_hb, \
                                                                      ivar_hb, sii_bestfit)
        else:
            gfit_hb, rchi2_hb = gfit_free, rchi2_free
            
    else:
        gfit_free, rchi2_free = fl.fit_hb_line.fit_free_two_components(lam_hb, flam_hb, \
                                                                      ivar_hb, sii_bestfit, \
                                                                      frac_temp = 100.)
        
        sig_sii = mfit.lamspace_to_velspace(sii_bestfit['sii6716'].stddev.value, \
                                           sii_bestfit['sii6716'].mean.value)
        sig_sii_out = mfit.lamspace_to_velspace(sii_bestfit['sii6716_out'].stddev.value, \
                                               sii_bestfit['sii6716_out'].mean.value)
        
        sig_hb = mfit.lamspace_to_velspace(gfit_free['hb_n'].stddev.value, \
                                          gfit_free['hb_n'].mean.value)
        sig_hb_out = mfit.lamspace_to_velspace(gfit_free['hb_out'].stddev.value, \
                                              gfit_free['hb_out'].mean.value)
        
        per_diff_n = (sig_sii - sig_hb)*100/sig_sii
        per_diff_out = (sig_sii_out - sig_hb_out)*100/sig_sii_out
        
        if (((per_diff_n <= -30)|(per_diff_n >= 30))|((per_diff_out <= -30)|(per_diff_out >= 30))):
            gfit_hb, rchi2_hb = fl.fit_hb_line.fit_fixed_two_components(lam_hb, flam_hb, \
                                                                       ivar_hb, sii_bestfit)
        else:
            gfit_hb, rchi2_hb = gfit_free, rchi2_free
            
        
    return (gfit_hb, rchi2_hb)
    
####################################################################################################
####################################################################################################

def find_nii_ha_best_fit(lam_nii, flam_nii, ivar_nii, sii_bestfit):
    """
    Function to find the best fit for [NII]+Ha, with or without broad-lines
    
    Parameters
    ----------
    lam_nii : numpy array
        Wavelength array of the [NII]+Ha region where the fit needs to be performed.
        
    flam_nii : numpy array
        Flux array of the spectra in the [NII]+Ha region
        
    ivar_nii : numpy array
        Inverse variance array of the spectra in the [NII]+Ha region
        
    sii_bestfit : Astropy model
        Best fit model for the [SII] emission-lines
        
    Returns
    -------
    gfit : Astropy model
        Best-fit "with" or "without" broad-line model
        
    rchi2 : float
        Reduced chi2 of the best-fit
    """
    n_sii = sii_bestfit.n_submodels
    
    ## first try fixing [NII] to [SII] and letting Ha free
    ## If Ha fit doesn't converge, fix Ha to [SII] as well
    
    if (n_sii == 2):
        gfit_free, rchi2_free = fl.fit_nii_ha_lines.fit_free_ha_one_component(lam_nii, flam_nii, ivar_nii, \
                                                                              sii_bestfit, frac_temp = 100.)

        sig_sii = mfit.lamspace_to_velspace(sii_bestfit['sii6716'].stddev.value, \
                                           sii_bestfit['sii6716'].mean.value)

        sig_ha = mfit.lamspace_to_velspace(gfit_free['ha_n'].stddev.value, \
                                          gfit_free['ha_n'].mean.value)

        per_diff = (sig_sii - sig_ha)*100/sig_sii
    
        if ((per_diff <= -30)|(per_diff >= 30)):
            gfit_nii_ha, rchi2_nii_ha = fl.fit_nii_ha_lines.fit_fixed_one_component(lam_nii, flam_nii, \
                                                                                    ivar_nii, sii_bestfit)
            
        else:
            gfit_nii_ha, rchi2_nii_ha = gfit_free, rchi2_free
            
    else:
        gfit_free, rchi2_free = fl.fit_nii_ha_lines.fit_free_ha_two_components(lam_nii, flam_nii, ivar_nii, \
                                                                               sii_bestfit, frac_temp = 100.)
        
        sig_sii = mfit.lamspace_to_velspace(sii_bestfit['sii6716'].stddev.value, \
                                           sii_bestfit['sii6716'].mean.value)
        sig_sii_out = mfit.lamspace_to_velspace(sii_bestfit['sii6716_out'].stddev.value, \
                                               sii_bestfit['sii6716_out'].mean.value)
        
        sig_ha = mfit.lamspace_to_velspace(gfit_free['ha_n'].stddev.value, \
                                          gfit_free['ha_n'].mean.value)
        sig_ha_out = mfit.lamspace_to_velspace(gfit_free['ha_out'].stddev.value, \
                                              gfit_free['ha_out'].mean.value)
        
        per_diff_n = (sig_sii - sig_ha)*100/sig_sii
        per_diff_out = (sig_sii_out - sig_ha_out)*100/sig_sii_out
        
        
        if (((per_diff_n <= -30)|(per_diff_n >= 30))|((per_diff_out <= -30)|(per_diff_out >= 30))):
            gfit_nii_ha, rchi2_nii_ha = fl.fit_nii_ha_lines.fit_fixed_two_components(lam_nii, flam_nii, \
                                                                                ivar_nii, sii_bestfit)
        else:
            gfit_nii_ha, rchi2_nii_ha = gfit_free, rchi2_free
        
    
    return (gfit_nii_ha, rchi2_nii_ha)

####################################################################################################
####################################################################################################
    
        
    
    