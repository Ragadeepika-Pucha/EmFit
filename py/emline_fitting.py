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
    
    ## Select [OIII] region for fitting
    oiii_lam = (lam_rest >= 4900)&(lam_rest <= 5050)
    lam_oiii = lam_rest[oiii_lam]
    flam_oiii = flam_rest[oiii_lam]
    ivar_oiii = ivar_rest[oiii_lam]
    
    ## Select [NII] + Ha region for fitting
    nii_lam = (lam_rest >= 6400)&(lam_rest <= 6700)
    lam_nii = lam_rest[nii_lam]
    flam_nii = flam_rest[nii_lam]
    ivar_nii = ivar_rest[nii_lam]
    
    ## Select Hb region for fitting
    hb_lam = (lam_rest >= 4800)&(lam_rest <= 4930)
    lam_hb = lam_rest[hb_lam]
    flam_hb = flam_rest[hb_lam]
    ivar_hb = ivar_rest[hb_lam]
    
    ## Select [SII] region for fitting
    sii_lam = (lam_rest >= 6650)&(lam_rest <= 6800)
    lam_sii = lam_rest[sii_lam]
    flam_sii = flam_rest[sii_lam]
    ivar_sii = ivar_rest[sii_lam]
    
    ## Fit [SII] emission-lines
    sii_fit, rchi2_sii = fit_lines.fit_sii_lines(lam_sii, flam_sii, ivar_sii)
    
    ## Fit [OIII] emission-lines
    oiii_fit, rchi2_oiii = fit_lines.fit_oiii_lines(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Fit Hb 
    hb_fit, rchi2_hb = fit_lines.fit_hb_line(lam_hb, flam_hb, ivar_hb)
    
    ## Fit [NII]+Ha
    nii_ha_fit, rchi2_nii_ha = fit_lines.fit_nii_ha_lines(lam_nii, flam_nii, ivar_nii, \
                                                              hb_fit, sii_fit)
    
    ## All the fits and rchi2 values in increasing order of wavelength
    fits = [hb_fit, oiii_fit, nii_ha_fit, sii_fit]
    rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    return (fits, rchi2s)

####################################################################################################
    
def plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s):
    """
    Plot spectra, fits and residuals in Hb, [OIII], Ha+[NII], and [SII] regions.
    
    Parameters
    ----------
    lam_rest : numpy array
        Restframe wavelength array of the emission-line spectra
        
    flam_rest : numpy array
        Restframe flux array of the emission-line spectra
        
    fits : list
        List of astropy model fits in the order of increasing wavelength.
        Hb fit, [OIII] fit, Ha+[NII] fit, [SII] fit
        
    rchi2 : list
        List of reduced chi2 values for the fits in the order of increasing wavelength.
        rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii
    
    Returns
    -------
        None.
    """
    
    ## Select [OIII] region 
    oiii_lam = (lam_rest >= 4900)&(lam_rest <= 5050)
    lam_oiii = lam_rest[oiii_lam]
    flam_oiii = flam_rest[oiii_lam]
    
    ## Select [NII] + Ha region 
    nii_lam = (lam_rest >= 6400)&(lam_rest <= 6700)
    lam_nii = lam_rest[nii_lam]
    flam_nii = flam_rest[nii_lam]
    
    ## Select Hb region 
    hb_lam = (lam_rest >= 4800)&(lam_rest <= 4930)
    lam_hb = lam_rest[hb_lam]
    flam_hb = flam_rest[hb_lam]
    
    ## Select [SII] region 
    sii_lam = (lam_rest >= 6650)&(lam_rest <= 6800)
    lam_sii = lam_rest[sii_lam]
    flam_sii = flam_rest[sii_lam]
    
    hb_fit, oiii_fit, nii_ha_fit, sii_fit = fits
    rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii = rchi2s
    
    fig = plt.figure(figsize = (30, 7))
    gs = fig.add_gridspec(5, 16)
    gs.update(hspace = 0.0)
    
    ## Subplots for Hb
    hb = fig.add_subplot(gs[0:4, 0:4])
    hb_res = fig.add_subplot(gs[4:, 0:4], sharex = hb)
    
    ## Subplots for [OIII]
    oiii = fig.add_subplot(gs[0:4, 4:8])
    oiii_res = fig.add_subplot(gs[4:, 4:8])
    
    ## Subplots for [NII]+Ha
    ha = fig.add_subplot(gs[0:4, 8:12])
    ha_res = fig.add_subplot(gs[4:, 8:12])
    
    ## Subplots for [SII]
    sii = fig.add_subplot(gs[0:4, 12:])
    sii_res = fig.add_subplot(gs[4:, 12:])
    
    #########################################################################
    #### Hb spectra + models
    
    hb.plot(lam_hb, flam_hb, color = 'k')
    hb.plot(lam_hb, hb_fit(lam_hb), color = 'r', lw = 3.0, ls = '--')
    hb.set(ylabel = '$F_{\lambda}$')
    plt.setp(hb.get_xticklabels(), visible = False)
    
    ## Hb fits
    n_hb = hb_fit.n_submodels
    
    if (n_hb == 1):
        hb.plot(lam_hb, hb_fit(lam_hb), color = 'orange')
    else:
        hb.plot(lam_hb, hb_fit['hb_n'](lam_hb), color = 'orange')
        hb.plot(lam_hb, hb_fit['hb_b'](lam_hb), color = 'blue')
        
    ## Rchi2 value
    hb.annotate('$H\\beta$', xy = (4810, 0.9), xycoords = hb.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
    hb.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_hb, 2)), xy = (4810, 0.8), xycoords = hb.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
                
        
    ## Hb fit residuals
    res_hb = (flam_hb - hb_fit(lam_hb))
    hb_res.scatter(lam_hb, res_hb, color = 'r', marker = '.')
    hb_res.axhline(0.0, color = 'k', ls = ':')
    hb_res.set(xlabel = '$\lambda$')
    hb_res.set_ylabel('Data - Model', fontsize = 14)
    
    #########################################################################
    #### [OIII] spectra + models
    
    oiii.plot(lam_oiii, flam_oiii, color = 'k')
    oiii.plot(lam_oiii, oiii_fit(lam_oiii), color = 'r', lw = 3.0, ls = '--')
    oiii.set(ylabel = '$F_{\lambda}$')
    plt.setp(oiii.get_xticklabels(), visible = False)
    
    ## [OIII] fits
    n_oiii = oiii_fit.n_submodels
    names_oiii = oiii_fit.submodel_names
    
    for ii in range(n_oiii):
        oiii.plot(lam_oiii, oiii_fit[names_oiii[ii]](lam_oiii), color = 'orange')
        
    ## Rchi2 value
    oiii.annotate('[OIII]4959,5007', xy = (4910, 0.9), xycoords = oiii.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
    oiii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_oiii, 2)), xy = (4910, 0.8), xycoords = oiii.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
    
    ## [OIII] fit residuals
    res_oiii = (flam_oiii - oiii_fit(lam_oiii))
    oiii_res.scatter(lam_oiii, res_oiii, color = 'r', marker = '.')
    oiii_res.axhline(0.0, color = 'k', ls = ':')
    oiii_res.set(xlabel = '$\lambda$')
    oiii_res.set_ylabel('Data - Model', fontsize = 14)
    
    #########################################################################
    #### [NII] + Ha spectra + models
    
    ha.plot(lam_nii, flam_nii, color = 'k')
    ha.plot(lam_nii, nii_ha_fit(lam_nii), color = 'r', lw = 3.0, ls = '--')
    ha.set(ylabel = '$F_{\lambda}$')
    plt.setp(ha.get_xticklabels(), visible = False)
    
    ## Ha + [NII] fits
    n_ha = nii_ha_fit.n_submodels
    names_ha = nii_ha_fit.submodel_names
    
    if ('ha_b' in names_ha):
        for ii in range(n_ha-1):
            ha.plot(lam_nii, nii_ha_fit[names_ha[ii]](lam_nii), color = 'orange')
        ha.plot(lam_nii, nii_ha_fit['ha_b'](lam_nii), color = 'blue')
    else:
        for ii in range(n_ha):
            ha.plot(lam_nii, nii_ha_fit[names_ha[ii]](lam_nii), color = 'orange')
            
    ## Rchi2 value
    ha.annotate('$H\\alpha$ + [NII]6548,6583', xy = (6410, 0.9), xycoords = ha.get_xaxis_transform(), \
                                      fontsize = 16, color = 'k')
    ha.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_nii_ha, 2)), xy = (6410, 0.8), xycoords = ha.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
        
    ## [NII]+Ha fit residuals
    res_nii = (flam_nii - nii_ha_fit(lam_nii))
    ha_res.scatter(lam_nii, res_nii, color = 'r', marker = '.')
    ha_res.axhline(0.0, color = 'k', ls = ':')
    ha_res.set(xlabel = '$\lambda$')
    ha_res.set_ylabel('Data - Model', fontsize = 14)
    
    #########################################################################
    #### [SII] spectra + models
    
    sii.plot(lam_sii, flam_sii, color = 'k')
    sii.plot(lam_sii, sii_fit(lam_sii), color = 'r', lw = 3.0, ls = '--')
    sii.set(ylabel = '$F_{\lambda}$')
    plt.setp(sii.get_xticklabels(), visible = False)
    
    ## [SII] fits
    n_sii = sii_fit.n_submodels
    names_sii = sii_fit.submodel_names
    
    for ii in range(n_sii):
        sii.plot(lam_sii, sii_fit[names_sii[ii]](lam_sii), color = 'orange')
        
    ## Rchi2 value
    sii.annotate('[SII]6716, 6731', xy = (6660, 0.9), xycoords = sii.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
    sii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_sii, 2)), xy = (6660, 0.8), xycoords = sii.get_xaxis_transform(), \
                                    fontsize = 16, color = 'k')
        
    ## [SII] fit residuals
    res_sii = (flam_sii - sii_fit(lam_sii))
    sii_res.scatter(lam_sii, res_sii, color = 'r', marker = '.')
    sii_res.axhline(0.0, color = 'k', ls = ':')
    sii_res.set(xlabel = '$\lambda$')
    sii_res.set_ylabel('Data - Model', fontsize = 14)
    
    plt.tight_layout()  
    
####################################################################################################

def emline_fit_spectra(specprod, survey, program, healpix, targetid, z):
    
    
    ## Get rest-frame emission-line spectra
    lam_rest, flam_rest, ivar_rest = fit_utils.get_emline_spectra(specprod, survey, \
                                                                  program, healpix, targetid,\
                                                                  z, rest_frame = True)
    
    ## Get the fits
    fits, rchi2s = fit_emline_spectra(lam_rest, flam_rest, ivar_rest)
    
    ## Plot the fits
    plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s)
    
####################################################################################################

    
    
    
    