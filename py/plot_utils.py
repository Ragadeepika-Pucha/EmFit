"""
This script consists of plotting-related functions.
The following functions are available:
    1) plot_spectra_continuum(lam, flam, total_cont, axs = None)
    2) plot_spectra_fits(targetid, lam_rest, flam_rest, fits, rchi2s)
    3) plot_emline_fit(lam_win, flam_win, emfit, narrow_components = None, \
                        broad_component = None)

Author : Ragadeepika Pucha
Version : 2023, March 27

"""

###################################################################################################

import numpy as np

from astropy.table import Table
import fitsio
import matplotlib.pyplot as plt

import measure_fits as mfit

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

def plot_spectra_continuum(lam, flam, total_cont, axs = None):
    """
    This function overplots the stellar continuum on the spectra.
    
    Parameters
    ----------
    lam : numpy array
        Wavelength array of the spectra. 
        
    flam : numpy array
        Flux array of the spectra
        
    total_cont : numpy array
        Total stellar continuum of the spectra
        
    axs : axis object
        Axes where the plot needs to be. Default is None.
        
    Returns
    -------
        None
    """
    
    if (axs == None):
        plt.figure(figsize = (24, 8))
        axs = plt.gca()
    ## Plotting spectra
    axs.plot(lam, flam, color = 'grey', alpha = 0.8, label = 'Spectra')
    ## Overplotting the total continuum 
    axs.plot(lam, total_cont, color = 'r', lw = 2.0, label = 'Total continuum')
    axs.set(xlabel = '$\lambda$', ylabel = '$F_{\lambda}$')
    axs.legend(fontsize = 16, loc = 'best')
    
###################################################################################################

def plot_spectra_fits(targetid, lam_rest, flam_rest, fits, rchi2s):
    """
    Plot spectra, fits and residuals in Hb, [OIII], Ha+[NII], and [SII] regions.
    
    Parameters
    ----------
    targetid : int64
        TARGETID of the object
    
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
        Figure
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
    
    ## Separate the fits and rchi2 values
    hb_fit, oiii_fit, nii_ha_fit, sii_fit = fits
    rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii = rchi2s
    
    fig = plt.figure(figsize = (30, 7))
    plt.suptitle(f'https://www.legacysurvey.org/viewer-desi/desi-spectrum/daily/targetid{targetid}',\
                 fontsize = 16)
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
    
    ################################################################################################
    ############################## Hb spectra + models #############################################
    
    hb.plot(lam_hb, flam_hb, color = 'k')
    hb.plot(lam_hb, hb_fit(lam_hb), color = 'r', lw = 3.0, ls = '--')
    hb.set(ylabel = '$F_{\lambda}$')
    plt.setp(hb.get_xticklabels(), visible = False)
    
    ## Hb fits
    n_hb = hb_fit.n_submodels
    
    if (n_hb == 1):
        hb.plot(lam_hb, hb_fit(lam_hb), color = 'orange')
        sig_hb_n = mfit.lamspace_to_velspace(hb_fit.stddev.value, \
                                                  hb_fit.mean.value)
        hb.annotate('$\sigma (H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s',\
                    xy = (4870, 0.9), xycoords = hb.get_xaxis_transform(),\
                    fontsize = 16, color = 'k')
    else:
        hb.plot(lam_hb, hb_fit['hb_n'](lam_hb), color = 'orange')
        hb.plot(lam_hb, hb_fit['hb_b'](lam_hb), color = 'blue')
        
        sig_hb_n = mfit.lamspace_to_velspace(hb_fit['hb_n'].stddev.value,\
                                                  hb_fit['hb_n'].mean.value)
        sig_hb_b = mfit.lamspace_to_velspace(hb_fit['hb_b'].stddev.value,\
                                                  hb_fit['hb_b'].mean.value)
        hb.annotate('$\sigma (H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', xy = (4870, 0.9),\
                    xycoords = hb.get_xaxis_transform(), fontsize = 16, color = 'k')
        hb.annotate('$\sigma (H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s', xy = (4870, 0.85),\
                    xycoords = hb.get_xaxis_transform(), fontsize = 16, color = 'k')
        
    ## Rchi2 value
    hb.annotate('$H\\beta$', xy = (4805, 0.9), xycoords = hb.get_xaxis_transform(),\
                                    fontsize = 16, color = 'k')
    hb.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_hb, 2)),\
                xy = (4805, 0.85), xycoords = hb.get_xaxis_transform(),\
                fontsize = 16, color = 'k')
                
    ## Hb fit residuals
    res_hb = (flam_hb - hb_fit(lam_hb))
    hb_res.scatter(lam_hb, res_hb, color = 'r', marker = '.')
    hb_res.axhline(0.0, color = 'k', ls = ':')
    hb_res.set(xlabel = '$\lambda$')
    hb_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [OIII] spectra + models #########################################
    
    oiii.plot(lam_oiii, flam_oiii, color = 'k')
    oiii.plot(lam_oiii, oiii_fit(lam_oiii), color = 'r', lw = 3.0, ls = '--')
    oiii.set(ylabel = '$F_{\lambda}$')
    plt.setp(oiii.get_xticklabels(), visible = False)
    
    ## [OIII] fits
    n_oiii = oiii_fit.n_submodels
    names_oiii = oiii_fit.submodel_names
    
    for ii in range(n_oiii):
        oiii.plot(lam_oiii, oiii_fit[names_oiii[ii]](lam_oiii), color = 'orange')
        sig_val = mfit.lamspace_to_velspace(oiii_fit[names_oiii[ii]].stddev.value, \
                                                 oiii_fit[names_oiii[ii]].mean.value)
        oiii.annotate(f'$\sigma$ ({names_oiii[ii]}) = {round(sig_val, 1)} km/s', \
                      xy = (4905, 0.8-(ii*0.05)), xycoords = oiii.get_xaxis_transform(),\
                      fontsize = 16, color = 'k')
        
    ## Rchi2 value
    oiii.annotate('[OIII]4959,5007', xy = (4905, 0.9), xycoords = oiii.get_xaxis_transform(),\
                  fontsize = 16, color = 'k')
    oiii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_oiii, 2)), xy = (4905, 0.85),\
                  xycoords = oiii.get_xaxis_transform(),\
                  fontsize = 16, color = 'k')
    
    ## [OIII] fit residuals
    res_oiii = (flam_oiii - oiii_fit(lam_oiii))
    oiii_res.scatter(lam_oiii, res_oiii, color = 'r', marker = '.')
    oiii_res.axhline(0.0, color = 'k', ls = ':')
    oiii_res.set(xlabel = '$\lambda$')
    oiii_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [NII]+Ha spectra + models #######################################
    
    ha.plot(lam_nii, flam_nii, color = 'k')
    ha.plot(lam_nii, nii_ha_fit(lam_nii), color = 'r', lw = 3.0, ls = '--')
    ha.set(ylabel = '$F_{\lambda}$')
    plt.setp(ha.get_xticklabels(), visible = False)
    
    ## Ha + [NII] fits
    n_ha = nii_ha_fit.n_submodels
    names_ha = nii_ha_fit.submodel_names
    
    if ('ha_b' in names_ha):
        for ii in range(n_ha-2):
            ha.plot(lam_nii, nii_ha_fit[names_ha[ii]](lam_nii), color = 'orange')
            sig_val = mfit.lamspace_to_velspace(nii_ha_fit[names_ha[ii]].stddev.value,\
                                                     nii_ha_fit[names_ha[ii]].mean.value)
            ha.annotate(f'$\sigma$ ({names_ha[ii]}) = \n{round(sig_val, 1)} km/s',\
                        xy = (6600, 0.9-(ii*0.1)), xycoords = ha.get_xaxis_transform(),\
                        fontsize = 16, color = 'k')    
        ha.plot(lam_nii, nii_ha_fit['ha_n'](lam_nii), color = 'orange')
        ha.plot(lam_nii, nii_ha_fit['ha_b'](lam_nii), color = 'blue')
        
        sig_ha_n = mfit.lamspace_to_velspace(nii_ha_fit['ha_n'].stddev.value,\
                                                  nii_ha_fit['ha_n'].mean.value)
        sig_ha_b = mfit.lamspace_to_velspace(nii_ha_fit['ha_b'].stddev.value,\
                                                  nii_ha_fit['ha_b'].mean.value)
        ha.annotate('$\sigma (H\\alpha;n)$ = '+str(round(sig_ha_n,1))+' km/s',\
                    xy = (6405, 0.8),xycoords = ha.get_xaxis_transform(),\
                    fontsize = 16, color = 'k')
        ha.annotate('$\sigma (H\\alpha;b)$ = '+str(round(sig_ha_b,1))+' km/s',\
                    xy = (6405, 0.75), xycoords = ha.get_xaxis_transform(),\
                    fontsize = 16, color = 'k')      
    else:
        for ii in range(n_ha-1):
            ha.plot(lam_nii, nii_ha_fit[names_ha[ii]](lam_nii), color = 'orange')
            sig_val = mfit.lamspace_to_velspace(nii_ha_fit[names_ha[ii]].stddev.value,\
                                                     nii_ha_fit[names_ha[ii]].mean.value)
            ha.annotate(f'$\sigma$ ({names_ha[ii]}) = \n{round(sig_val, 1)} km/s',\
                        xy = (6600, 0.9-(ii*0.1)), xycoords = ha.get_xaxis_transform(),\
                        fontsize = 16, color = 'k')
        ha.plot(lam_nii, nii_ha_fit['ha_n'](lam_nii), color = 'orange')
        
        sig_ha_n = mfit.lamspace_to_velspace(nii_ha_fit['ha_n'].stddev.value,\
                                                  nii_ha_fit['ha_n'].mean.value)
        ha.annotate('$\sigma (H\\alpha;n)$ = '+str(round(sig_ha_n, 1))+' km/s',\
                    xy = (6405, 0.8),xycoords = ha.get_xaxis_transform(),\
                    fontsize = 16, color = 'k')
            
            
    ## Rchi2 value
    ha.annotate('$H\\alpha$ + [NII]6548,6583', xy = (6405, 0.9),\
                xycoords = ha.get_xaxis_transform(),\
                fontsize = 16, color = 'k')
    ha.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_nii_ha, 2)),\
                xy = (6405, 0.85), xycoords = ha.get_xaxis_transform(),\
                fontsize = 16, color = 'k')
        
    ## [NII]+Ha fit residuals
    res_nii = (flam_nii - nii_ha_fit(lam_nii))
    ha_res.scatter(lam_nii, res_nii, color = 'r', marker = '.')
    ha_res.axhline(0.0, color = 'k', ls = ':')
    ha_res.set(xlabel = '$\lambda$')
    ha_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [SII] spectra + models ##########################################
    
    sii.plot(lam_sii, flam_sii, color = 'k')
    sii.plot(lam_sii, sii_fit(lam_sii), color = 'r', lw = 3.0, ls = '--')
    sii.set(ylabel = '$F_{\lambda}$')
    plt.setp(sii.get_xticklabels(), visible = False)
    
    ## [SII] fits
    n_sii = sii_fit.n_submodels
    names_sii = sii_fit.submodel_names
    
    for ii in range(n_sii):
        sii.plot(lam_sii, sii_fit[names_sii[ii]](lam_sii), color = 'orange')
        sig_val = mfit.lamspace_to_velspace(sii_fit[names_sii[ii]].stddev.value,\
                                                 sii_fit[names_sii[ii]].mean.value)
        sii.annotate(f'$\sigma$ ({names_sii[ii]}) = \n{round(sig_val, 1)} km/s',\
                     xy = (6655, 0.75-(ii*0.1)), xycoords = sii.get_xaxis_transform(),\
                     fontsize = 16, color = 'k')
        
    ## Rchi2 value
    sii.annotate('[SII]6716, 6731', xy = (6655, 0.9),\
                 xycoords = sii.get_xaxis_transform(),\
                 fontsize = 16, color = 'k')
    sii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_sii, 2)),\
                 xy = (6655, 0.85), xycoords = sii.get_xaxis_transform(),\
                 fontsize = 16, color = 'k')
        
    ## [SII] fit residuals
    res_sii = (flam_sii - sii_fit(lam_sii))
    sii_res.scatter(lam_sii, res_sii, color = 'r', marker = '.')
    sii_res.axhline(0.0, color = 'k', ls = ':')
    sii_res.set(xlabel = '$\lambda$')
    sii_res.set_ylabel('Data - Model', fontsize = 14)
    
    plt.tight_layout()
    plt.close()
    
    return (fig)
    
####################################################################################################

def plot_emline_fit(lam_win, flam_win, emfit, narrow_components = None, \
                    broad_component = None):
    """
    Plot the emission-line fit for a specific emission-line.
    
    Parameters
    ----------
    lam_win : numpy array
        Wavelength array for the fit window
        
    flam_win : numpy array
        Flux array of the spectra in the fit window
        
    emfit : Astropy model
        Astropy model in the fit window
        
    narrow_components : list
        List of astropy narrow component models
        Default is None
        
    broad_component : list
        List of astropy broad component model
        Default is None
        
    Returns
    -------
        Figure
    """
    
    
    fig = plt.figure(figsize = (8,7))
    gs = fig.add_gridspec(5, 4)
    gs.update(hspace = 0.0)
    
    ax1 = fig.add_subplot(gs[0:4, :])
    ax2 = fig.add_subplot(gs[4:, :], sharex = ax1)
    
    ax1.plot(lam_win, flam_win, color = 'k')
    ax1.plot(lam_win, emfit(lam_win), color = 'r', lw = 3.0, ls = '--')
    ax1.set(ylabel = '$F_{\lambda}$')
    plt.setp(ax1.get_xticklabels(), visible = False)
    
    if (narrow_components is not None):
        for comp in narrow_components:
            ax1.plot(lam_win, comp(lam_win), color = 'orange')
        
    if (broad_component is not None):
        ax1.plot(lam_win, broad_component(lam_win), color = 'blue')
            
    res = (flam_win - emfit(lam_win))
    ax2.scatter(lam_win, res, color = 'r', marker = '.')
    ax2.axhline(0.0, color = 'k', ls = ':')
    ax2.set(xlabel = '$\lambda$')
    ax2.set_ylabel('Data - Model', fontsize = 14)
    
    plt.close()
    
    return (fig)
    
####################################################################################################