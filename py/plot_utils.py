"""
This script consists of plotting-related functions.
The following functions are available:
    1) plot_spectra_continuum(lam, flam, total_cont, axs = None)
    2) plot_spectra_fits(targetid, lam_rest, flam_rest, fits, rchi2s)
    3) plot_emline_fit(lam_win, flam_win, emfit, narrow_components = None, \
                        broad_component = None)
    4) plot_from_params(lam_rest, flam_rest, t, index, rchi2s)
    5) plot_spectra_fastspec_model(lam_rest, flam_rest, model_rest, fspec_row, title = None)
    6) compare_different_fits(table, index, t_em)

Author : Ragadeepika Pucha
Version : 2023, October 3rd

"""

###################################################################################################

import numpy as np

from astropy.table import Table
from astropy.modeling.models import Gaussian1D, Const1D

import fitsio
import matplotlib.pyplot as plt
import spec_utils
import emline_fitting as emfit

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


def plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s, title = None):
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
        
    title : str
        Title for the plot
    
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
    plt.suptitle(title, fontsize = 16)
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
    
    ## Hb continuum
    hb_cont = hb_fit['hb_cont'].amplitude.value

    if (n_hb == 1):
        hb.plot(lam_hb, hb_fit(lam_hb), color = 'orange')
        sig_hb_n = mfit.lamspace_to_velspace(hb_fit.stddev.value, \
                                             hb_fit.mean.value)
        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                    xy = (4870, 0.9), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')
    else:
        hb.plot(lam_hb, hb_fit['hb_n'](lam_hb) + hb_cont, color = 'orange')
        sig_hb_n = mfit.lamspace_to_velspace(hb_fit['hb_n'].stddev.value, \
                                             hb_fit['hb_n'].mean.value)
        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                    xy = (4870, 0.9), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')
        
        if ('hb_out' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_out'](lam_hb) + hb_cont, color = 'orange')
            sig_hb_out = mfit.lamspace_to_velspace(hb_fit['hb_out'].stddev.value, \
                                                   hb_fit['hb_out'].mean.value)
            
            hb.annotate('$\sigma \\rm(H\\beta;out)$ = '+str(round(sig_hb_out, 1))+' km/s', \
                        xy = (4870, 0.8), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 16, color = 'k')
            
        if ('hb_b' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_b'](lam_hb) + hb_cont, color = 'blue')
            sig_hb_b = mfit.lamspace_to_velspace(hb_fit['hb_b'].stddev.value,\
                                                      hb_fit['hb_b'].mean.value)

            hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s',\
                        xy = (4870, 0.7), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 16, color = 'k')
        
    ## Rchi2 value
    hb.annotate('$H\\beta$', xy = (4800, 0.9), \
                xycoords = hb.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    hb.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_hb, 2)), \
                xy = (4800, 0.8), xycoords = hb.get_xaxis_transform(), \
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
    
    n_oiii = oiii_fit.n_submodels
    names_oiii = oiii_fit.submodel_names
    
    ### [OIII] continuum
    oiii_cont = oiii_fit['oiii_cont'].amplitude.value
    
    for ii in range(1, n_oiii):
        oiii.plot(lam_oiii, oiii_fit[names_oiii[ii]](lam_oiii) + oiii_cont, color = 'orange')
                
    sig_oiii = mfit.lamspace_to_velspace(oiii_fit['oiii5007'].stddev.value, \
                                         oiii_fit['oiii5007'].mean.value)
    oiii.annotate('$\sigma$ ([OIII]) = '+str(round(sig_oiii, 1))+' km/s', \
                  xy = (4900, 0.7), xycoords = oiii.get_xaxis_transform(), \
                  fontsize = 16, color = 'k')
    
    if ('oiii4959_out' in names_oiii):
        sig_oiii_out = mfit.lamspace_to_velspace(oiii_fit['oiii5007_out'].stddev.value, \
                                                 oiii_fit['oiii5007_out'].mean.value)
        
        oiii.annotate('$\sigma$ ([OIII];out) = '+str(round(sig_oiii_out, 1))+ ' km/s', \
                      xy = (4900, 0.6), xycoords = oiii.get_xaxis_transform(), \
                      fontsize = 16, color = 'k')
    ## Rchi2 value
    oiii.annotate('[OIII]4959,5007', xy = (4900, 0.9), \
                  xycoords = oiii.get_xaxis_transform(),\
                  fontsize = 16, color = 'k')
    oiii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_oiii, 2)), \
                  xy = (4900, 0.8), xycoords = oiii.get_xaxis_transform(), \
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
    
    ## [NII] + Ha continuum
    nii_ha_cont = nii_ha_fit['nii_ha_cont'].amplitude.value
    
    ## Plot narrow components
    ha.plot(lam_nii, nii_ha_fit['nii6548'](lam_nii) + nii_ha_cont, color = 'orange')
    ha.plot(lam_nii, nii_ha_fit['nii6583'](lam_nii) + nii_ha_cont, color = 'orange')
    ha.plot(lam_nii, nii_ha_fit['ha_n'](lam_nii) + nii_ha_cont, color = 'orange')
    
    sig_nii = mfit.lamspace_to_velspace(nii_ha_fit['nii6548'].stddev.value, \
                                        nii_ha_fit['nii6548'].mean.value)
    ha.annotate('$\sigma$ ([NII]) = \n'+str(round(sig_nii, 1))+' km/s', \
                xy = (6600, 0.9), xycoords = ha.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    
    sig_ha = mfit.lamspace_to_velspace(nii_ha_fit['ha_n'].stddev.value, \
                                       nii_ha_fit['ha_n'].mean.value)
    ha.annotate('$\sigma \\rm(H\\alpha)$ = \n'+str(round(sig_ha, 1))+' km/s', \
                xy = (6600, 0.8), xycoords = ha.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    
    ## Outflow components
    if ('nii6548_out' in nii_ha_fit.submodel_names):
        ha.plot(lam_nii, nii_ha_fit['nii6548_out'](lam_nii) + nii_ha_cont, color = 'orange')
        ha.plot(lam_nii, nii_ha_fit['nii6583_out'](lam_nii) + nii_ha_cont, color = 'orange')
        ha.plot(lam_nii, nii_ha_fit['ha_out'](lam_nii) + nii_ha_cont, color = 'orange')
        
        sig_nii_out = mfit.lamspace_to_velspace(nii_ha_fit['nii6548_out'].stddev.value, \
                                                nii_ha_fit['nii6548_out'].mean.value)
        
        ha.annotate('$\sigma$ ([NII];out) = \n'+str(round(sig_nii_out, 1))+' km/s', \
                    xy = (6600, 0.7), xycoords = ha.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')
        
        sig_ha_out = mfit.lamspace_to_velspace(nii_ha_fit['ha_out'].stddev.value, \
                                               nii_ha_fit['ha_out'].mean.value)
        
        ha.annotate('$\sigma \\rm(H\\alpha;out)$ = \n'+str(round(sig_ha_out, 1))+' km/s', \
                    xy = (6600, 0.6), xycoords = ha.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')
        
    ## Broad component
    if ('ha_b' in nii_ha_fit.submodel_names):
        ha.plot(lam_nii, nii_ha_fit['ha_b'](lam_nii) + nii_ha_cont, color = 'blue')
        
        sig_ha_b = mfit.lamspace_to_velspace(nii_ha_fit['ha_b'].stddev.value, \
                                             nii_ha_fit['ha_b'].mean.value)
        
        fwhm_ha_b = 2.355*sig_ha_b
        
        ha.annotate('FWHM $\\rm(H\\alpha;b)$ = '+str(round(fwhm_ha_b, 1))+' km/s', \
                    xy = (6400, 0.7), xycoords = ha.get_xaxis_transform(), \
                    fontsize = 14, color = 'k')
            
            
    ## Rchi2 value
    ha.annotate('$H\\alpha$ + [NII]6548,6583', xy = (6400, 0.9),\
                xycoords = ha.get_xaxis_transform(),\
                fontsize = 16, color = 'k')
    ha.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_nii_ha, 2)),\
                xy = (6400, 0.8), xycoords = ha.get_xaxis_transform(),\
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
    
    ## [SII] continuum
    sii_cont = sii_fit['sii_cont'].amplitude.value
    
    for ii in range(1, n_sii):
        sii.plot(lam_sii, sii_fit[names_sii[ii]](lam_sii) + sii_cont, color = 'orange')
        
    sig_sii = mfit.lamspace_to_velspace(sii_fit['sii6716'].stddev.value, \
                                        sii_fit['sii6716'].mean.value)
    
    sii.annotate('$\sigma$ ([SII]) = '+str(round(sig_sii, 1))+' km/s', \
                 xy = (6650, 0.7), xycoords = sii.get_xaxis_transform(), \
                 fontsize = 16, color = 'k')
    
    if ('sii6716_out' in names_sii):
        sig_sii_out = mfit.lamspace_to_velspace(sii_fit['sii6716_out'].stddev.value, \
                                                sii_fit['sii6716_out'].mean.value)
        
        sii.annotate('$\sigma$ ([SII];out) = '+str(round(sig_sii_out, 1))+' km/s', \
                     xy = (6650, 0.6), xycoords = sii.get_xaxis_transform(), \
                     fontsize = 16, color = 'k')
        
    ## Rchi2 value
    sii.annotate('[SII]6716, 6731', xy = (6650, 0.9),\
                 xycoords = sii.get_xaxis_transform(),\
                 fontsize = 16, color = 'k')
    sii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_sii, 2)),\
                 xy = (6650, 0.8), xycoords = sii.get_xaxis_transform(),\
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

def plot_from_params(table, index, title = None):
    """
    Function to plot the spectra+fits from the table of parameters.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame wavelength array of the spectra
    
    flam_rest : numpy array
        Rest-frame flux array of the spectra
        
    t : Astropy table
        Table with parameters
    
    index : int
        Index of the source in the table
        
    rchi2s : list
        List of reduced chi2 values from the fit in the order:
        [Hbeta, [OIII], [NII] + Ha, [SII]] 
        
    title : str
        Title of the plot. Default is None.
        
    Returns
    -------
    fig : Figure
        Figure with spectra + best-fit model
    """

    specprod = table['SPECPROD'].astype(str).data[index]
    survey = table['SURVEY'].astype(str).data[index]
    program = table['PROGRAM'].astype(str).data[index]
    healpix = table['HEALPIX'].data[index]
    targetid = table['TARGETID'].data[index]
    z = table['Z'].data[index]
    
    lam_rest, flam_rest, ivar_rest, _ = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                                     healpix, targetid, z, \
                                                                     rest_frame = True)
    
    fits = emfit.construct_fits(table, index)
    
    hb_rchi2 = table['HB_RCHI2'].data[index]
    oiii_rchi2 = table['OIII_RCHI2'].data[index]
    nii_ha_rchi2 = table['NII_HA_RCHI2'].data[index]
    sii_rchi2 = table['SII_RCHI2'].data[index]
    
    rchi2s = [hb_rchi2, oiii_rchi2, nii_ha_rchi2, sii_rchi2]
    if (title == None):
        title = f'TARGETID: {targetid}; z : {round(z, 2)}'

    fig = plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s, title = title)
    
    return (fig)

####################################################################################################

# def plot_spectra_fastspec_model(lam, flam, model, axs = None):
#     """
#     This function overplots the fastspecfit model on the spectra.
    
#     Parameters
#     ----------
#     lam : numpy array
#         Wavelength array for the spectra
        
#     flam : numpy array
#         Flux array of the spectra
        
#     model : numpy array
#         Model array of the spectra
        
#     axs : axis object
#         Axes where the plot needs to be. Default is None
        
#     Returns
#     -------
#         None
#     """
    
#     if (axs == None):
#         plt.figure(figsize = (24, 8))
#         axs = plt.gca()
        
#     ## Plotting spectra
#     axs.plot(lam, flam, color = 'grey', label = 'Spectra')
#     ## Overplotting the model
#     axs.plot(lam, model, color = 'k', lw = 2.0, label = 'Fastspec Model')
#     axs.set(xlabel = '$\lambda$', ylabel = '$F_{\lambda}$')
#     axs.legend(fontsize = 16, loc = 'best')
    
def plot_spectra_fastspec_model(lam_rest, flam_rest, model_rest, fspec_row, title = None):
    """
    This function overplots the fastspecfit model on the spectra, focusing on 
    Hb, [OIII], [NII]+Ha, and [SII] regions.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame wavelength array
        
    flam_rest : numpy array
        Rest-frame flux array
        
    model_rest : numpy array
        Rest-frame fastspecfit model array
        
    fspec_row : Astropy Row
        Fastspec measurements
        
    title : str
        Title for the entire plot. Default is None.
        
    Returns
    -------
        Figure
        
    """
    
    ## Select [OIII] region
    oiii_lam = (lam_rest >= 4900)&(lam_rest <= 5050)
    lam_oiii = lam_rest[oiii_lam]
    flam_oiii = flam_rest[oiii_lam]
    model_oiii = model_rest[oiii_lam]
    
    ## Select [NII]+Ha region
    nii_lam = (lam_rest >= 6400)&(lam_rest <= 6700)
    lam_nii = lam_rest[nii_lam]
    flam_nii = flam_rest[nii_lam]
    model_nii = model_rest[nii_lam]
    
    ## Select Hb region
    hb_lam = (lam_rest >= 4800)&(lam_rest <= 4930)
    lam_hb = lam_rest[hb_lam]
    flam_hb = flam_rest[hb_lam]
    model_hb = model_rest[hb_lam]
    
    ## Select [SII] region
    sii_lam = (lam_rest >= 6650)&(lam_rest <= 6800)
    lam_sii = lam_rest[sii_lam]
    flam_sii = flam_rest[sii_lam]
    model_sii = model_rest[sii_lam]
    
    fig = plt.figure(figsize = (30,7))
    plt.suptitle(title, fontsize = 16)
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
    hb.plot(lam_hb, model_hb, color = 'r', lw = 3.0, ls = '--')
    hb.set(ylabel = '$F_{\lambda}$')
    plt.setp(hb.get_xticklabels(), visible = False)
    
    sig_hb_n = fspec_row['HBETA_SIGMA'].data[0]
    sig_hb_b = fspec_row['HBETA_BROAD_SIGMA'].data[0]
    
    hb.annotate('$H\\beta$', xy = (4800, 0.9), \
                xycoords = hb.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    
    hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                    xy = (4870, 0.9), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')
    
    hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s',\
                        xy = (4870, 0.7), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 16, color = 'k')
    
    
    ## Hb fit residuals
    res_hb = (flam_hb - model_hb)
    hb_res.scatter(lam_hb, res_hb, color = 'r', marker = '.')
    hb_res.axhline(0.0, color = 'k', ls = ':')
    hb_res.set(xlabel = '$\lambda$')
    hb_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [OIII] spectra + models #########################################
    
    oiii.plot(lam_oiii, flam_oiii, color = 'k')
    oiii.plot(lam_oiii, model_oiii, color = 'r', lw = 3.0, ls = '--')
    oiii.set(ylabel = '$F_{\lambda}$')
    plt.setp(oiii.get_xticklabels(), visible = False)
    
    sig_oiii = fspec_row['OIII_5007_SIGMA'].data[0]
    
    oiii.annotate('[OIII]4959,5007', xy = (4900, 0.9), \
                  xycoords = oiii.get_xaxis_transform(),\
                  fontsize = 16, color = 'k')
    
    oiii.annotate('$\sigma$ ([OIII]) = '+str(round(sig_oiii, 1))+' km/s', \
                  xy = (4900, 0.7), xycoords = oiii.get_xaxis_transform(), \
                  fontsize = 16, color = 'k')
    
    ## [OIII] fit residuals
    res_oiii = (flam_oiii - model_oiii)
    oiii_res.scatter(lam_oiii, res_oiii, color = 'r', marker = '.')
    oiii_res.axhline(0.0, color = 'k', ls = ':')
    oiii_res.set(xlabel = '$\lambda$')
    oiii_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [NII]+Ha spectra + models #######################################
    
    ha.plot(lam_nii, flam_nii, color = 'k')
    ha.plot(lam_nii, model_nii, color = 'r', lw = 3.0, ls = '--')
    ha.set(ylabel = '$F_{\lambda}$')
    plt.setp(ha.get_xticklabels(), visible = False)
    
    sig_nii = fspec_row['NII_6584_SIGMA'].data[0]
    sig_ha = fspec_row['HALPHA_SIGMA'].data[0]
    sig_ha_b = fspec_row['HALPHA_BROAD_SIGMA'].data[0]
    fwhm_ha_b = 2.355*sig_ha_b

    ha.annotate('$H\\alpha$ + [NII]6548,6583', xy = (6400, 0.9),\
                xycoords = ha.get_xaxis_transform(),\
                fontsize = 16, color = 'k')
    
    ha.annotate('$\sigma$ ([NII]) = \n'+str(round(sig_nii, 1))+' km/s', \
                xy = (6600, 0.9), xycoords = ha.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    
    ha.annotate('$\sigma \\rm(H\\alpha)$ = \n'+str(round(sig_ha, 1))+' km/s', \
                xy = (6600, 0.8), xycoords = ha.get_xaxis_transform(), \
                fontsize = 16, color = 'k')
    
    ha.annotate('FWHM $\\rm(H\\alpha;b)$ = '+str(round(fwhm_ha_b, 1))+' km/s', \
                    xy = (6400, 0.7), xycoords = ha.get_xaxis_transform(), \
                    fontsize = 14, color = 'k')
    
    ## [NII]+Ha fit residuals
    res_nii = (flam_nii - model_nii)
    ha_res.scatter(lam_nii, res_nii, color = 'r', marker = '.')
    ha_res.axhline(0.0, color = 'k', ls = ':')
    ha_res.set(xlabel = '$\lambda$')
    ha_res.set_ylabel('Data - Model', fontsize = 14)
    
    ################################################################################################
    ############################## [SII] spectra + models ##########################################
    
    sii.plot(lam_sii, flam_sii, color = 'k')
    sii.plot(lam_sii, model_sii, color = 'r', lw = 3.0, ls = '--')
    sii.set(ylabel = '$F_{\lambda}$')
    plt.setp(sii.get_xticklabels(), visible = False)
    
    sig_sii = fspec_row['SII_6731_SIGMA'].data[0]
    
    sii.annotate('[SII]6716, 6731', xy = (6650, 0.9),\
                 xycoords = sii.get_xaxis_transform(),\
                 fontsize = 16, color = 'k')
    
    sii.annotate('$\sigma$ ([SII]) = '+str(round(sig_sii, 1))+' km/s', \
                 xy = (6650, 0.7), xycoords = sii.get_xaxis_transform(), \
                 fontsize = 16, color = 'k')
    
    ## [SII] fit residuals
    res_sii = (flam_sii - model_sii)
    sii_res.scatter(lam_sii, res_sii, color = 'r', marker = '.')
    sii_res.axhline(0.0, color = 'k', ls = ':')
    sii_res.set(xlabel = '$\lambda$')
    sii_res.set_ylabel('Data - Model', fontsize = 14)
    
    plt.tight_layout()
    plt.close()
    
    return (fig)

###################################################################################################

def compare_different_fits(table, index, t_em):
    """
    Function to compare emfit and Fastspecfit V2 and V3.
    
    Parameters 
    ----------
    table : Astropy Table
        Table of Sources
        
    index : int
        Row number of the interested target
        
    t_em : Astropy Table
        Emfit Results Table
        
    Returns
    -------
    fig_em : Figure
        EmFit fits figure
        
    fig_v2 : Figure
        Fastspecfit V2.0 figure
        
    fig_v3 : Figure
        Fastspecfit V3.0 figure
    
    """
    
    
    ## Variables
    specprod = table['SPECPROD'].astype(str).data[index]
    survey = table['SURVEY'].astype(str).data[index]
    program = table['PROGRAM'].astype(str).data[index]
    healpix = table['HEALPIX'].data[index]
    targetid = table['TARGETID'].data[index]
    z = table['Z'].data[index]
    logmass = table['logM'].data[index]
    
    ### Get Spectra
    coadd_spec = spec_utils.find_coadded_spectra(specprod, survey, program, \
                                                healpix, targetid)
    _, cont_v2, em_model_v2, fspec_row_v2 = spec_utils.find_fastspec_models(specprod, survey, \
                                                                            program, healpix, \
                                                                            targetid, \
                                                                            ver = 'v2.0', \
                                                                            fspec = True)
    _, cont_v3, em_model_v3, fspec_row_v3 = spec_utils.find_fastspec_models(specprod, survey, \
                                                                            program, healpix, \
                                                                            targetid, \
                                                                            ver = 'v3.0', \
                                                                            fspec = True)

    ## Total fastspecfit model
    fast_model_v2 = cont_v2 + em_model_v2
    fast_model_v3 = cont_v3 + em_model_v3
    
    ## Convert to rest-frame values
    lam_rest = coadd_spec.wave['brz']/(1+z)
    flam_rest = coadd_spec.flux['brz'].flatten()*(1+z)
    fmodel_rest_v2 = fast_model_v2*(1+z)
    fmodel_rest_v3 = fast_model_v3*(1+z)
    
    ## Emfit Information
    sel = np.nonzero((t_em['TARGETID'].data == targetid))[0][0]
    fits = emfit.construct_fits(t_em, sel)
    
    hb_rchi2 = t_em['HB_RCHI2'].data[sel]
    oiii_rchi2 = t_em['OIII_RCHI2'].data[sel]
    nii_ha_rchi2 = t_em['NII_HA_RCHI2'].data[sel]
    sii_rchi2 = t_em['SII_RCHI2'].data[sel]
    
    rchi2s = [hb_rchi2, oiii_rchi2, nii_ha_rchi2, sii_rchi2]
    
    ## Create the title
    title = f'TARGETID: {targetid}; z : {round(z, 3)}; log M*: {round(logmass, 1)}\n'+\
        f'https://www.legacysurvey.org/viewer-desi/desi-spectrum/daily/targetid{targetid}\n'
    
    
    ## Figures
    fig_em = plot_spectra_fits(lam_rest, flam_rest - (cont_v2*(1+z)), fits, \
                               rchi2s, title = title)
    fig_v2 = plot_spectra_fastspec_model(lam_rest, flam_rest, fmodel_rest_v2, \
                                         fspec_row_v2, title = title)
    fig_v3 = plot_spectra_fastspec_model(lam_rest, flam_rest, fmodel_rest_v3, \
                                         fspec_row_v3, title = title)
    
    return (fig_em, fig_v2, fig_v3)

###################################################################################################