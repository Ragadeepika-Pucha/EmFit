"""
This script consists of plotting-related functions.
The following functions are available:
    1) plot_spectra_continuum(lam, flam, total_cont, axs = None)
    2) plot_emline_fit(lam_win, flam_win, emfit, narrow_components = None, \
                        broad_component = None)
    3) plot_spectra_fits.normal_fit(lam_rest, flam_rest, fits, rchi2s, title = None)
    4) plot_spectra_fits.extreme_fit(lam_rest, flam_rest, fits, rchi2s, title = None)
    5) plot_fits_from_table.normal_fit(table, index, title = None)
    6) plot_fits_from_table.extreme_fit(table, index, title = None)
    7) plot_from_params(table, index, title = None)
    8) plot_fits(table, index, title = None, plot_smooth_cont = False)

Author : Ragadeepika Pucha
Version : 2024, April 25

"""

###################################################################################################

import numpy as np

from astropy.table import Table
from astropy.modeling.models import Gaussian1D, Const1D

from desiutil.dust import dust_transmission

import fitsio
import matplotlib.pyplot as plt
import spec_utils
import emline_fitting as emfit

import measure_fits as mfit

###################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':28,
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
    
####################################################################################################
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
        First submodel should be the continuum model
        
    narrow_components : list
        List of astropy narrow component models
        Default is None
        
    broad_component : Astropy model
        Astropy broad component model
        Default is None
        
    continuum : Astropy model
        Astropy continuum model
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
    
    ## Continuum
    cont = emfit[0].amplitude.value
    
    if (narrow_components is not None):
        for comp in narrow_components:
            ax1.plot(lam_win, comp(lam_win)+cont, color = 'orange')
        
    if (broad_component is not None):
        ax1.plot(lam_win, broad_component(lam_win)+cont, color = 'blue')
            
    res = (flam_win - emfit(lam_win))
    ax2.scatter(lam_win, res, color = 'r', marker = '.')
    ax2.axhline(0.0, color = 'k', ls = ':')
    ax2.set(xlabel = '$\lambda$')
    ax2.set_ylabel('Data - Model', fontsize = 14)
    
    plt.close()
    
    return (fig)
    
####################################################################################################
####################################################################################################

class plot_spectra_fits:
    """
    Functions to plot spectra, fits, and residuals.
        1) normal_fit(lam_rest, flam_rest, fits, rchi2s, title = None)
        2) extreme_fit(lam_rest, flam_rest, fits, rchi2s, title = None)
    """

    def normal_fit(lam_rest, flam_rest, fits, rchi2s, title = None, \
                   smooth_cont = None, plot_smooth_cont = False):
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
            
        smooth_cont : numpy array
            Restframe smooth continuum array of the spectra.
            Only needed if plot_smooth_cont = True
            Default is None.
            
        plot_smooth_cont : bool
            Whether or not to overplot smooth continuum from fastspecfit

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
        
        ## If plot_smooth_cont = True
        ## Divide into different windows
        if (plot_smooth_cont == True):
            cont_hb = smooth_cont[hb_lam]
            cont_oiii = smooth_cont[oiii_lam]
            cont_nii = smooth_cont[nii_lam]
            cont_sii = smooth_cont[sii_lam]

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

        ############################################################################################
        ############################## Hb spectra + models #########################################

        hb.plot(lam_hb, flam_hb, color = 'k')
        hb.plot(lam_hb, hb_fit(lam_hb), color = 'r', lw = 3.0, ls = '--')
        hb.set(ylabel = '$F_{\lambda}$')
        plt.setp(hb.get_xticklabels(), visible = False)

        ## Hb fits
        n_hb = hb_fit.n_submodels

        ## Hb continuum
        hb_cont = hb_fit['hb_cont'].amplitude.value

        ## Narrow component of Hb
        hb.plot(lam_hb, hb_fit['hb_n'](lam_hb) + hb_cont, color = 'orange')
        sig_hb_n = mfit.lamspace_to_velspace(hb_fit['hb_n'].stddev.value, \
                                             hb_fit['hb_n'].mean.value)
        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                    xy = (4870, 0.9), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 16, color = 'k')

        ## Outflow component of Hb, if available 
        if ('hb_out' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_out'](lam_hb) + hb_cont, color = 'orange')
            sig_hb_out = mfit.lamspace_to_velspace(hb_fit['hb_out'].stddev.value, \
                                                   hb_fit['hb_out'].mean.value)

            hb.annotate('$\sigma \\rm(H\\beta;out)$ = '+str(round(sig_hb_out, 1))+' km/s', \
                        xy = (4870, 0.8), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 16, color = 'k')

        ## Broad component of Hb, if available
        if ('hb_b' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_b'](lam_hb) + hb_cont, color = 'blue')
            sig_hb_b = mfit.lamspace_to_velspace(hb_fit['hb_b'].stddev.value,\
                                                      hb_fit['hb_b'].mean.value)

            hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s',\
                        xy = (4870, 0.7), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 16, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            hb.plot(lam_hb, cont_hb, color = 'grey')

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
        hb_res.set(xlabel = '$\lambda_{rest}$')
        hb_res.set_ylabel('Data - Model', fontsize = 14)
        

        ############################################################################################
        ############################## [OIII] spectra + models #####################################

        oiii.plot(lam_oiii, flam_oiii, color = 'k')
        oiii.plot(lam_oiii, oiii_fit(lam_oiii), color = 'r', lw = 3.0, ls = '--')
        oiii.set(ylabel = '$F_{\lambda}$')
        plt.setp(oiii.get_xticklabels(), visible = False)

        n_oiii = oiii_fit.n_submodels
        names_oiii = oiii_fit.submodel_names

        ### [OIII] continuum
        oiii_cont = oiii_fit['oiii_cont'].amplitude.value

        ## [OIII]4959,5007 components, including outflow components, if available
        for ii in range(1, n_oiii):
            oiii.plot(lam_oiii, oiii_fit[names_oiii[ii]](lam_oiii) + oiii_cont, color = 'orange')

        sig_oiii = mfit.lamspace_to_velspace(oiii_fit['oiii5007'].stddev.value, \
                                             oiii_fit['oiii5007'].mean.value)
        oiii.annotate('$\sigma$ ([OIII]) = '+str(round(sig_oiii, 1))+' km/s', \
                      xy = (4900, 0.7), xycoords = oiii.get_xaxis_transform(), \
                      fontsize = 16, color = 'k')

        ## Outflow component sigma values if available
        if ('oiii4959_out' in names_oiii):
            sig_oiii_out = mfit.lamspace_to_velspace(oiii_fit['oiii5007_out'].stddev.value, \
                                                     oiii_fit['oiii5007_out'].mean.value)

            oiii.annotate('$\sigma$ ([OIII];out) = '+str(round(sig_oiii_out, 1))+ ' km/s', \
                          xy = (4900, 0.6), xycoords = oiii.get_xaxis_transform(), \
                          fontsize = 16, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            oiii.plot(lam_oiii, cont_oiii, color = 'grey')
            
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
        oiii_res.set(xlabel = '$\lambda_{rest}$')
        oiii_res.set_ylabel('Data - Model', fontsize = 14)

        ############################################################################################
        ############################## [NII]+Ha spectra + models ###################################

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

        ## Outflow components, if available
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

        ## Broad component, if available
        if ('ha_b' in nii_ha_fit.submodel_names):
            ha.plot(lam_nii, nii_ha_fit['ha_b'](lam_nii) + nii_ha_cont, color = 'blue')

            sig_ha_b = mfit.lamspace_to_velspace(nii_ha_fit['ha_b'].stddev.value, \
                                                 nii_ha_fit['ha_b'].mean.value)

            fwhm_ha_b = 2.355*sig_ha_b

            ha.annotate('FWHM $\\rm(H\\alpha;b)$ = '+str(round(fwhm_ha_b, 1))+' km/s', \
                        xy = (6400, 0.7), xycoords = ha.get_xaxis_transform(), \
                        fontsize = 14, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            ha.plot(lam_nii, cont_nii, color = 'grey')

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
        ha_res.set(xlabel = '$\lambda_{rest}$')
        ha_res.set_ylabel('Data - Model', fontsize = 14)

        ############################################################################################
        ############################## [SII] spectra + models ######################################

        sii.plot(lam_sii, flam_sii, color = 'k')
        sii.plot(lam_sii, sii_fit(lam_sii), color = 'r', lw = 3.0, ls = '--')
        sii.set(ylabel = '$F_{\lambda}$')
        plt.setp(sii.get_xticklabels(), visible = False)

        ## [SII] fits
        n_sii = sii_fit.n_submodels
        names_sii = sii_fit.submodel_names

        ## [SII] continuum
        sii_cont = sii_fit['sii_cont'].amplitude.value

        ## All components of [SII], outflow components included, if available
        for ii in range(1, n_sii):
            sii.plot(lam_sii, sii_fit[names_sii[ii]](lam_sii) + sii_cont, color = 'orange')

        sig_sii = mfit.lamspace_to_velspace(sii_fit['sii6716'].stddev.value, \
                                            sii_fit['sii6716'].mean.value)

        sii.annotate('$\sigma$ ([SII]) = '+str(round(sig_sii, 1))+' km/s', \
                     xy = (6650, 0.7), xycoords = sii.get_xaxis_transform(), \
                     fontsize = 16, color = 'k')

        ## Outflow sigma values, if available
        if ('sii6716_out' in names_sii):
            sig_sii_out = mfit.lamspace_to_velspace(sii_fit['sii6716_out'].stddev.value, \
                                                    sii_fit['sii6716_out'].mean.value)

            sii.annotate('$\sigma$ ([SII];out) = '+str(round(sig_sii_out, 1))+' km/s', \
                         xy = (6650, 0.6), xycoords = sii.get_xaxis_transform(), \
                         fontsize = 16, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            sii.plot(lam_sii, cont_sii, color = 'grey')

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
        sii_res.set(xlabel = '$\lambda_{rest}$')
        sii_res.set_ylabel('Data - Model', fontsize = 14)

        plt.tight_layout()
        plt.close()

        return (fig)
    
####################################################################################################

    def extreme_fit(lam_rest, flam_rest, fits, rchi2s, title = None, \
                   smooth_cont = None, plot_smooth_cont = False):
        """
        Plot spectra, fits and residuals in Hb+[OIII] and [NII]+Ha+[SII] regions.

       Parameters
        ----------    
        lam_rest : numpy array
            Restframe wavelength array of the emission-line spectra

        flam_rest : numpy array
            Restframe flux array of the emission-line spectra

        fits : list
            List of astropy model fits in the order of increasing wavelength.
            Hb+[OIII] fit, [NII]+Ha+[SII] fit

        rchi2 : list
            List of reduced chi2 values for the fits in the order of increasing wavelength.
            rchi2_hb_oiii, rchi2_nii_ha_sii

        title : str
            Title for the plot
            
        smooth_cont : numpy array
            Restframe smooth continuum array of the spectra.
            Only needed if plot_smooth_cont = True
            Default is None.
            
        plot_smooth_cont : bool
            Whether or not to overplot smooth continuum from fastspecfit

        Returns
        -------
            Figure
        """

        ## Select Hb+[OIII] region
        hb_oiii_lam = (lam_rest >= 4700)&(lam_rest <= 5100)
        lam_hb_oiii = lam_rest[hb_oiii_lam]
        flam_hb_oiii = flam_rest[hb_oiii_lam]

        ## Select [NII]+Ha+[SII] region
        nii_ha_sii_lam = (lam_rest >= 6300)&(lam_rest <= 6800)
        lam_nii_ha_sii = lam_rest[nii_ha_sii_lam]
        flam_nii_ha_sii = flam_rest[nii_ha_sii_lam]

        ## Separate the fits and rchi2 values
        hb_oiii_fit, nii_ha_sii_fit = fits
        hb_oiii_rchi2, nii_ha_sii_rchi2 = rchi2s
        
        ## If plot_smooth_cont = True
        ## Divide into different windows
        if (plot_smooth_cont == True):
            cont_hb_oiii = smooth_cont[hb_oiii_lam]
            cont_nii_ha_sii = smooth_cont[nii_ha_sii_lam]
            
        fig = plt.figure(figsize = (30, 9))
        plt.suptitle(title, fontsize = 16)
        gs = fig.add_gridspec(5, 16)
        gs.update(hspace = 0.0)

        ## Subplots for Hb+[OIII]
        hb = fig.add_subplot(gs[0:4, 0:8])
        hb_res = fig.add_subplot(gs[4:, 0:8], sharex = hb)

        ## Subplots for [NII]+Ha+[SII]
        ha = fig.add_subplot(gs[0:4, 8:])
        ha_res = fig.add_subplot(gs[4:, 8:], sharex = ha)

        ############################################################################################
        ############################ Hb + [OIII] spectra and models ################################

        hb.plot(lam_hb_oiii, flam_hb_oiii, color = 'k')
        hb.plot(lam_hb_oiii, hb_oiii_fit(lam_hb_oiii), color = 'r', lw = 3.0, ls = '--')
        hb.set(ylabel = '$F_{\lambda}$')
        plt.setp(hb.get_xticklabels(), visible = False)

        ## Hb + [OIII] continuum
        hb_cont = hb_oiii_fit['hb_oiii_cont'].amplitude.value

        ## Narrow components
        hb.plot(lam_hb_oiii, hb_oiii_fit['hb_n'](lam_hb_oiii) + hb_cont, color = 'orange')
        hb.plot(lam_hb_oiii, hb_oiii_fit['oiii4959'](lam_hb_oiii) + hb_cont, color = 'orange')
        hb.plot(lam_hb_oiii, hb_oiii_fit['oiii5007'](lam_hb_oiii) + hb_cont, color = 'orange')

        sig_hb_n = mfit.lamspace_to_velspace(hb_oiii_fit['hb_n'].stddev.value, \
                                            hb_oiii_fit['hb_n'].mean.value)
        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                   xy = (4720, 0.9), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        sig_oiii = mfit.lamspace_to_velspace(hb_oiii_fit['oiii5007'].stddev.value, \
                                            hb_oiii_fit['oiii5007'].mean.value)
        hb.annotate('$\sigma \\rm([OIII])$ = '+str(round(sig_oiii, 1))+' km/s', \
                   xy = (4720, 0.8), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        if ('hb_b' in hb_oiii_fit.submodel_names):
            ## Broad component
            hb.plot(lam_hb_oiii, hb_oiii_fit['hb_b'](lam_hb_oiii) + hb_cont, color = 'blue')
            sig_hb_b = mfit.lamspace_to_velspace(hb_oiii_fit['hb_b'].stddev.value, \
                                                hb_oiii_fit['hb_b'].mean.value)
            hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s', \
                       xy = (4720, 0.7), xycoords = hb.get_xaxis_transform(), \
                       fontsize = 16, color = 'k')

        ## Outflow components, if available
        if ('oiii5007_out' in hb_oiii_fit.submodel_names):
            hb.plot(lam_hb_oiii, hb_oiii_fit['oiii4959_out'](lam_hb_oiii) + hb_cont, \
                    color = 'orange')
            hb.plot(lam_hb_oiii, hb_oiii_fit['oiii5007_out'](lam_hb_oiii) + hb_cont, \
                    color = 'orange')

            sig_oiii_out = mfit.lamspace_to_velspace(hb_oiii_fit['oiii5007_out'].stddev.value, \
                                            hb_oiii_fit['oiii5007_out'].mean.value)
            hb.annotate('$\sigma \\rm([OIII];out)$ = '+str(round(sig_oiii_out, 1))+' km/s', \
                       xy = (4720, 0.6), xycoords = hb.get_xaxis_transform(), \
                       fontsize = 16, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            hb.plot(lam_hb_oiii, cont_hb_oiii, color = 'grey')

        hb.annotate('$\chi^{2}_{red}$ = '+str(round(hb_oiii_rchi2, 2)), \
                   xy = (5020, 0.9), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        ## Hb + [OIII] Fit residuals
        res_hb_oiii = (flam_hb_oiii - hb_oiii_fit(lam_hb_oiii))
        hb_res.scatter(lam_hb_oiii, res_hb_oiii, color = 'r', marker = '.')
        hb_res.axhline(0.0, color = 'k', ls = ':')
        hb_res.set(xlabel = '$\lambda_{rest}$')
        hb_res.set_ylabel('Data - Model', fontsize = 14)

        ############################################################################################
        ###################### [NII] + Ha + [SII] spectra and models ###############################

        ha.plot(lam_nii_ha_sii, flam_nii_ha_sii, color = 'k')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit(lam_nii_ha_sii), color = 'r', lw = 3.0, ls = '--')
        ha.set(ylabel = '$F_{\lambda}$')
        plt.setp(ha.get_xticklabels(), visible = False)

        ## [NII]+Ha+[SII] continuum
        ha_cont = nii_ha_sii_fit['nii_ha_sii_cont'].amplitude.value

        ## Narrow components
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['ha_n'](lam_nii_ha_sii) + ha_cont, color = 'orange')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['nii6548'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['nii6583'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['sii6716'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['sii6731'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange')

        sig_ha_n = mfit.lamspace_to_velspace(nii_ha_sii_fit['ha_n'].stddev.value, \
                                            nii_ha_sii_fit['ha_n'].mean.value)
        ha.annotate('$\sigma \\rm(H\\alpha;n)$ = '+str(round(sig_ha_n, 1))+' km/s', \
                   xy = (6320, 0.9), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')
        sig_nii = mfit.lamspace_to_velspace(nii_ha_sii_fit['nii6583'].stddev.value, \
                                            nii_ha_sii_fit['nii6583'].mean.value)
        ha.annotate('$\sigma \\rm([NII])$ = '+str(round(sig_nii, 1))+' km/s', \
                   xy = (6320, 0.8), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        sig_sii = mfit.lamspace_to_velspace(nii_ha_sii_fit['sii6716'].stddev.value, \
                                            nii_ha_sii_fit['sii6716'].mean.value)
        ha.annotate('$\sigma \\rm([SII])$ = '+str(round(sig_nii, 1))+' km/s', \
                   xy = (6320, 0.7), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        if ('ha_b' in nii_ha_sii_fit.submodel_names):
            ## Broad component
            ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['ha_b'](lam_nii_ha_sii) + ha_cont, \
                    color = 'blue')
            sig_ha_b = mfit.lamspace_to_velspace(nii_ha_sii_fit['ha_b'].stddev.value, \
                                                nii_ha_sii_fit['ha_b'].mean.value)
            fwhm_ha_b = mfit.sigma_to_fwhm(sig_ha_b)
            ha.annotate('$\\rm FWHM (H\\alpha;b)$ = '+str(round(fwhm_ha_b, 1))+' km/s', \
                       xy = (6320, 0.6), xycoords = ha.get_xaxis_transform(), \
                       fontsize = 16, color = 'k')
            
        ## Plot smooth continuum if plot_smooth_cont = True
        if (plot_smooth_cont == True):
            ha.plot(lam_nii_ha_sii, cont_nii_ha_sii, color = 'grey')

        ha.annotate('$\chi^{2}_{red}$ = '+str(round(nii_ha_sii_rchi2, 2)), \
                   xy = (6700, 0.9), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 16, color = 'k')

        ## [NII]+Ha+[SII] fit residuals
        res_nii_ha_sii = (flam_nii_ha_sii - nii_ha_sii_fit(lam_nii_ha_sii))
        ha_res.scatter(lam_nii_ha_sii, res_nii_ha_sii, color = 'r', marker = '.')
        ha_res.axhline(0.0, color = 'k', ls = ':')
        ha_res.set(xlabel = '$\lambda_{rest}$')
        ha_res.set_ylabel('Data - Model', fontsize = 14)

        plt.tight_layout()
        plt.close()

        return (fig)

####################################################################################################
####################################################################################################

class plot_fits_from_table:
    """
    Functions to plot spectra, fits, and residuals directly from the table.
        1) normal_fit(table, index, title = None)
        2) extreme_fit(table, index, title = None)
    """
    def normal_fit(table, index, ha_xlim = None, title = None):
        """
        Function to plot the spectra+fits from the table of parameters for default cases.

        Parameters
        ----------
        table : Astropy table
            Table of fit parameters

        index : int
            Index number of the source in the table

        title : str
            Title of the plot. Default is None.

        Returns
        -------
        fig : Figure
            Figure with spectra + best-fit model
        """

        ## Required variables
        specprod = table['SPECPROD'].astype(str).data[index]
        survey = table['SURVEY'].astype(str).data[index]
        program = table['PROGRAM'].astype(str).data[index]
        healpix = table['HEALPIX'].data[index]
        targetid = table['TARGETID'].data[index]
        z = table['Z'].data[index]

        ## Accessing the rest-frame spectra
        _, lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, healpix, \
                                                                         targetid, z, rest_frame = True)
        if (title == None):
            title = f'TARGETID: {targetid}; z : {round(z, 2)}'

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

        ## Construct Fits
        fits = emfit.construct_fits_from_table.normal_fit(table, index)
        ## Separate the fits 
        hb_fit, oiii_fit, nii_ha_fit, sii_fit = fits
        ## Reduced chi2 values from the table
        rchi2_hb = table['HB_RCHI2'].data[index]
        rchi2_oiii = table['OIII_RCHI2'].data[index]
        rchi2_nii_ha = table['NII_HA_RCHI2'].data[index]
        rchi2_sii = table['SII_RCHI2'].data[index]

        fig = plt.figure(figsize = (40, 8))
        plt.suptitle(title, fontsize = 20)
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

        ############################################################################################
        ############################## Hb spectra + models #########################################

        hb.plot(lam_hb, flam_hb, color = 'k')
        hb.plot(lam_hb, hb_fit(lam_hb), color = 'r', lw = 4.0, ls = '--')
        hb.set(ylabel = '$F_{\lambda}$')
        plt.setp(hb.get_xticklabels(), visible = False)

        ## Hb fits
        n_hb = hb_fit.n_submodels

        ## Hb continuum
        hb_cont = hb_fit['hb_cont'].amplitude.value

        ## Narrow component of Hb
        hb.plot(lam_hb, hb_fit['hb_n'](lam_hb) + hb_cont, color = 'orange', lw = 2.0)
        sig_hb_n = table['HB_N_SIGMA'].data[index]

        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                    xy = (4875, 0.9), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 20, color = 'k')

        ## Outflow component of Hb, if available 
        if ('hb_out' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_out'](lam_hb) + hb_cont, color = 'orange', lw = 2.0)
            sig_hb_out = table['HB_OUT_SIGMA'].data[index]
            hb.annotate('$\sigma \\rm(H\\beta;out)$ = '+str(round(sig_hb_out, 1))+' km/s', \
                        xy = (4875, 0.8), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 20, color = 'k')

        ## Broad component of Hb, if available
        if ('hb_b' in hb_fit.submodel_names):
            hb.plot(lam_hb, hb_fit['hb_b'](lam_hb) + hb_cont, color = 'blue', lw = 2.0)
            sig_hb_b = table['HB_B_SIGMA'].data[index]
            hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s',\
                        xy = (4875, 0.7), xycoords = hb.get_xaxis_transform(), \
                        fontsize = 20, color = 'k')

        ## Rchi2 value
        hb.annotate('$H\\beta$', xy = (4800, 0.9), \
                    xycoords = hb.get_xaxis_transform(), \
                    fontsize = 20, color = 'k')
        hb.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_hb, 2)), \
                    xy = (4800, 0.8), xycoords = hb.get_xaxis_transform(), \
                    fontsize = 20, color = 'k')

        ## Hb fit residuals
        res_hb = (flam_hb - hb_fit(lam_hb))/flam_hb
        hb_res.scatter(lam_hb, res_hb, color = 'k', marker = '.', alpha = 0.7)
        hb_res.axhline(0.0, color = 'grey', ls = ':')
        hb_res.set(yticks = [0])
        hb_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10, 10])

        ############################################################################################
        ############################## [OIII] spectra + models #####################################

        oiii.plot(lam_oiii, flam_oiii, color = 'k')
        oiii.plot(lam_oiii, oiii_fit(lam_oiii), color = 'r', lw = 4.0, ls = '--')
        oiii.set(ylabel = '$F_{\lambda}$')
        plt.setp(oiii.get_xticklabels(), visible = False)

        n_oiii = oiii_fit.n_submodels
        names_oiii = oiii_fit.submodel_names

        ### [OIII] continuum
        oiii_cont = oiii_fit['oiii_cont'].amplitude.value

        ## [OIII]4959,5007 components, including outflow components, if available
        for ii in range(1, n_oiii):
            oiii.plot(lam_oiii, oiii_fit[names_oiii[ii]](lam_oiii) + oiii_cont, color = 'orange', lw = 2.0)

        sig_oiii = table['OIII5007_SIGMA'].data[index]
        oiii.annotate('$\sigma$ ([OIII]) = '+str(round(sig_oiii, 1))+' km/s', \
                      xy = (4900, 0.7), xycoords = oiii.get_xaxis_transform(), \
                      fontsize = 20, color = 'k')

        ## Outflow component sigma values if available
        if ('oiii4959_out' in names_oiii):
            sig_oiii_out = table['OIII5007_OUT_SIGMA'].data[index]
            oiii.annotate('$\sigma$ ([OIII];out) = '+str(round(sig_oiii_out, 1))+ ' km/s', \
                          xy = (4900, 0.6), xycoords = oiii.get_xaxis_transform(), \
                          fontsize = 20, color = 'k')

        ## Rchi2 value
        oiii.annotate('[OIII]4959,5007', xy = (4900, 0.9), \
                      xycoords = oiii.get_xaxis_transform(),\
                      fontsize = 20, color = 'k')
        oiii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_oiii, 2)), \
                      xy = (4900, 0.8), xycoords = oiii.get_xaxis_transform(), \
                      fontsize = 20, color = 'k')

        ## [OIII] fit residuals
        res_oiii = (flam_oiii - oiii_fit(lam_oiii))/flam_oiii
        oiii_res.scatter(lam_oiii, res_oiii, color = 'k', marker = '.', alpha = 0.7)
        oiii_res.axhline(0.0, color = 'k', ls = ':')
        oiii_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10, 10], yticks = [0])

        ############################################################################################
        ############################## [NII]+Ha spectra + models ###################################

        ha.plot(lam_nii, flam_nii, color = 'k')
        ha.plot(lam_nii, nii_ha_fit(lam_nii), color = 'r', lw = 4.0, ls = '--')
        ha.set(ylabel = '$F_{\lambda}$')
        plt.setp(ha.get_xticklabels(), visible = False)
        
        if (ha_xlim is not None):
            ha.set(xlim = ha_xlim)
        

        ## [NII] + Ha continuum
        nii_ha_cont = nii_ha_fit['nii_ha_cont'].amplitude.value

        ## Plot narrow components
        ha.plot(lam_nii, nii_ha_fit['nii6548'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)
        ha.plot(lam_nii, nii_ha_fit['nii6583'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)
        ha.plot(lam_nii, nii_ha_fit['ha_n'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)

        sig_nii = table['NII6583_SIGMA'].data[index]
        ha.annotate('$\sigma$ ([NII]) = \n'+str(round(sig_nii, 1))+' km/s', \
                    xy = (0.8, 0.85), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()), \
                    fontsize = 20, color = 'k')

        sig_ha = table['HA_N_SIGMA'].data[index]
        ha.annotate('$\sigma \\rm(H\\alpha)$ = \n'+str(round(sig_ha, 1))+' km/s', \
                    xy = (0.8, 0.7), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()), \
                    fontsize = 20, color = 'k')

        ## Outflow components, if available
        if ('nii6548_out' in nii_ha_fit.submodel_names):
            ha.plot(lam_nii, nii_ha_fit['nii6548_out'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)
            ha.plot(lam_nii, nii_ha_fit['nii6583_out'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)
            ha.plot(lam_nii, nii_ha_fit['ha_out'](lam_nii) + nii_ha_cont, color = 'orange', lw = 2.0)

            sig_nii_out = table['NII6583_OUT_SIGMA'].data[index]
            ha.annotate('$\sigma$ ([NII];out) = \n'+str(round(sig_nii_out, 1))+' km/s', \
                        xy = (0.8, 0.55), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()), \
                        fontsize = 20, color = 'k')

            sig_ha_out = table['HA_OUT_SIGMA'].data[index]
            ha.annotate('$\sigma \\rm(H\\alpha;out)$ = \n'+str(round(sig_ha_out, 1))+' km/s', \
                        xy = (0.8, 0.4), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()), \
                        fontsize = 20, color = 'k')

        ## Broad component, if available
        if ('ha_b' in nii_ha_fit.submodel_names):
            ha.plot(lam_nii, nii_ha_fit['ha_b'](lam_nii) + nii_ha_cont, color = 'blue', lw = 2.0)

            sig_ha_b = table['HA_B_SIGMA'].data[index]
            fwhm_ha_b = 2.355*sig_ha_b
            ha.annotate('FWHM $\\rm(H\\alpha;b)$ = \n'+str(round(fwhm_ha_b, 1))+' km/s', \
                        xy = (0.02, 0.65), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()), \
                        fontsize = 20, color = 'k')


        ## Rchi2 value
        ha.annotate('$H\\alpha$ + [NII]6548,6583', xy = (0.02, 0.9),\
                    xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()),\
                    fontsize = 20, color = 'k')
        ha.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_nii_ha, 2)),\
                    xy = (0.02, 0.8), xycoords = (ha.get_yaxis_transform(), ha.get_xaxis_transform()),\
                    fontsize = 20, color = 'k')

        ## [NII]+Ha fit residuals
        res_nii = (flam_nii - nii_ha_fit(lam_nii))/flam_nii
        ha_res.scatter(lam_nii, res_nii, color = 'k', marker = '.', alpha = 0.7)
        ha_res.axhline(0.0, color = 'k', ls = ':')
        ha_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10, 10], yticks = [0])
        if (ha_xlim is not None):
            ha_res.set(xlim = ha_xlim)

        ############################################################################################
        ############################## [SII] spectra + models ######################################

        sii.plot(lam_sii, flam_sii, color = 'k')
        sii.plot(lam_sii, sii_fit(lam_sii), color = 'r', lw = 4.0, ls = '--')
        sii.set(ylabel = '$F_{\lambda}$')
        plt.setp(sii.get_xticklabels(), visible = False)

        ## [SII] fits
        n_sii = sii_fit.n_submodels
        names_sii = sii_fit.submodel_names

        ## [SII] continuum
        sii_cont = sii_fit['sii_cont'].amplitude.value

        ## All components of [SII], outflow components included, if available
        for ii in range(1, n_sii):
            sii.plot(lam_sii, sii_fit[names_sii[ii]](lam_sii) + sii_cont, color = 'orange')

        sig_sii = table['SII6716_SIGMA'].data[index]
        sii.annotate('$\sigma$ ([SII]) = \n'+str(round(sig_sii, 1))+' km/s', \
                     xy = (6760, 0.85), xycoords = sii.get_xaxis_transform(), \
                     fontsize = 20, color = 'k')

        ## Outflow sigma values, if available
        if ('sii6716_out' in names_sii):
            sig_sii_out = table['SII6716_OUT_SIGMA'].data[index]
            sii.annotate('$\sigma$ ([SII];out) = \n'+str(round(sig_sii_out, 1))+' km/s', \
                         xy = (6760, 0.7), xycoords = sii.get_xaxis_transform(), \
                         fontsize = 20, color = 'k')

        ## Rchi2 value
        sii.annotate('[SII]6716, 6731', xy = (6650, 0.9),\
                     xycoords = sii.get_xaxis_transform(),\
                     fontsize = 20, color = 'k')
        sii.annotate('$\chi^{2}_{red}$ = '+str(round(rchi2_sii, 2)),\
                     xy = (6650, 0.8), xycoords = sii.get_xaxis_transform(),\
                     fontsize = 20, color = 'k')

        ## [SII] fit residuals
        res_sii = (flam_sii - sii_fit(lam_sii))/flam_sii
        sii_res.scatter(lam_sii, res_sii, color = 'k', marker = '.', alpha = 0.7)
        sii_res.axhline(0.0, color = 'k', ls = ':')
        sii_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10, 10], yticks = [0])
        plt.tight_layout()
        plt.close()

        return (fig)
    
####################################################################################################

    def extreme_fit(table, index, title = None):
        """
        Function to plot the spectra+fits from the table of parameters for extreme BL cases.

        Parameters
        ----------
        table : Astropy table
            Table of fit parameters

        index : int
            Index number of the source in the table

        title : str
            Title of the plot. Default is None.

        Returns
        -------
        fig : Figure
            Figure with spectra + best-fit model
        """

        ## Required variables
        specprod = table['SPECPROD'].astype(str).data[index]
        survey = table['SURVEY'].astype(str).data[index]
        program = table['PROGRAM'].astype(str).data[index]
        healpix = table['HEALPIX'].data[index]
        targetid = table['TARGETID'].data[index]
        z = table['Z'].data[index]

        ## Accessing the rest-frame spectra
        _, lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, healpix, \
                                                                         targetid, z, rest_frame = True)
        if (title == None):
            title = f'TARGETID: {targetid}; z : {round(z, 2)}'

        ## Select Hb+[OIII] region
        hb_oiii_lam = (lam_rest >= 4700)&(lam_rest <= 5100)
        lam_hb_oiii = lam_rest[hb_oiii_lam]
        flam_hb_oiii = flam_rest[hb_oiii_lam]

        ## Select [NII]+Ha+[SII] region
        nii_ha_sii_lam = (lam_rest >= 6300)&(lam_rest <= 6800)
        lam_nii_ha_sii = lam_rest[nii_ha_sii_lam]
        flam_nii_ha_sii = flam_rest[nii_ha_sii_lam]

        ## Construct Fits
        fits = emfit.construct_fits_from_table.extreme_fit(table, index)
        ## Separate the fits 
        hb_oiii_fit, nii_ha_sii_fit = fits
        ## Rchi2 values
        hb_oiii_rchi2 = table['HB_OIII_RCHI2'].data[index]
        nii_ha_sii_rchi2 = table['NII_HA_SII_RCHI2'].data[index]

        fig = plt.figure(figsize = (40, 10))
        plt.suptitle(title, fontsize = 20)
        gs = fig.add_gridspec(5, 16)
        gs.update(hspace = 0.0)

        ## Subplots for Hb+[OIII]
        hb = fig.add_subplot(gs[0:4, 0:8])
        hb_res = fig.add_subplot(gs[4:, 0:8], sharex = hb)

        ## Subplots for [NII]+Ha+[SII]
        ha = fig.add_subplot(gs[0:4, 8:])
        ha_res = fig.add_subplot(gs[4:, 8:], sharex = ha)

        ############################################################################################
        ############################ Hb + [OIII] spectra and models ################################

        hb.plot(lam_hb_oiii, flam_hb_oiii, color = 'k')
        hb.plot(lam_hb_oiii, hb_oiii_fit(lam_hb_oiii), color = 'r', lw = 4.0, ls = '--')
        hb.set(ylabel = '$F_{\lambda}$')
        plt.setp(hb.get_xticklabels(), visible = False)

        ## Hb + [OIII] continuum
        hb_cont = hb_oiii_fit['hb_oiii_cont'].amplitude.value

        ## Narrow components
        hb.plot(lam_hb_oiii, hb_oiii_fit['hb_n'](lam_hb_oiii) + hb_cont, color = 'orange', lw = 2.0)
        hb.plot(lam_hb_oiii, hb_oiii_fit['oiii4959'](lam_hb_oiii) + hb_cont, color = 'orange', lw = 2.0)
        hb.plot(lam_hb_oiii, hb_oiii_fit['oiii5007'](lam_hb_oiii) + hb_cont, color = 'orange', lw = 2.0)

        sig_hb_n = table['HB_N_SIGMA'].data[index]
        hb.annotate('$\sigma \\rm(H\\beta;n)$ = '+str(round(sig_hb_n, 1))+' km/s', \
                   xy = (4700, 0.7), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        sig_oiii = table['OIII5007_SIGMA'].data[index]
        hb.annotate('$\sigma \\rm([OIII])$ = '+str(round(sig_oiii, 1))+' km/s', \
                   xy = (4700, 0.6), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        if ('hb_b' in hb_oiii_fit.submodel_names):
            ## Broad component
            hb.plot(lam_hb_oiii, hb_oiii_fit['hb_b'](lam_hb_oiii) + hb_cont, color = 'blue', lw = 2.0)
            sig_hb_b = table['HB_B_SIGMA'].data[index]
            hb.annotate('$\sigma \\rm(H\\beta;b)$ = '+str(round(sig_hb_b, 1))+' km/s', \
                       xy = (5020, 0.9), xycoords = hb.get_xaxis_transform(), \
                       fontsize = 20, color = 'k')

        ## Outflow components, if available
        if ('oiii5007_out' in hb_oiii_fit.submodel_names):
            hb.plot(lam_hb_oiii, hb_oiii_fit['oiii4959_out'](lam_hb_oiii) + hb_cont, \
                    color = 'orange', lw = 2.0)
            hb.plot(lam_hb_oiii, hb_oiii_fit['oiii5007_out'](lam_hb_oiii) + hb_cont, \
                    color = 'orange', lw = 2.0)

            sig_oiii_out = table['OIII5007_OUT_SIGMA'].data[index]
            hb.annotate('$\sigma \\rm([OIII];out)$ = '+str(round(sig_oiii_out, 1))+' km/s', \
                       xy = (4700, 0.5), xycoords = hb.get_xaxis_transform(), \
                       fontsize = 20, color = 'k')

        hb.annotate('$\\rm H\\beta + [OIII]4959,5007$', xy = (4700, 0.9), \
                   xycoords = hb.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        hb.annotate('$\chi^{2}_{red}$ = '+str(round(hb_oiii_rchi2, 2)), \
                   xy = (4700, 0.8), xycoords = hb.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        ## Hb + [OIII] Fit residuals
        res_hb_oiii = (flam_hb_oiii - hb_oiii_fit(lam_hb_oiii))/flam_hb_oiii
        hb_res.scatter(lam_hb_oiii, res_hb_oiii, color = 'k', marker = '.', alpha = 0.7)
        hb_res.axhline(0.0, color = 'k', ls = ':')
        hb_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10,10], yticks = [0])

        ############################################################################################
        ###################### [NII] + Ha + [SII] spectra and models ###############################

        ha.plot(lam_nii_ha_sii, flam_nii_ha_sii, color = 'k')
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit(lam_nii_ha_sii), color = 'r', lw = 4.0, ls = '--')
        ha.set(ylabel = '$F_{\lambda}$')
        plt.setp(ha.get_xticklabels(), visible = False)

        ## [NII]+Ha+[SII] continuum
        ha_cont = nii_ha_sii_fit['nii_ha_sii_cont'].amplitude.value

        ## Narrow components
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['ha_n'](lam_nii_ha_sii) + ha_cont, color = 'orange', lw = 2.0)
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['nii6548'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange', lw = 2.0)
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['nii6583'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange', lw = 2.0)
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['sii6716'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange', lw = 2.0)
        ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['sii6731'](lam_nii_ha_sii) + ha_cont, \
                color = 'orange', lw = 2.0)

        sig_ha_n = table['HA_N_SIGMA'].data[index]
        ha.annotate('$\sigma \\rm(H\\alpha;n)$ = '+str(round(sig_ha_n, 1))+' km/s', \
                   xy = (6300, 0.7), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')
        sig_nii = table['NII6583_SIGMA'].data[index]
        ha.annotate('$\sigma \\rm([NII])$ = '+str(round(sig_nii, 1))+' km/s', \
                   xy = (6300, 0.6), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        sig_sii = table['SII6716_SIGMA'].data[index]
        ha.annotate('$\sigma \\rm([SII])$ = '+str(round(sig_nii, 1))+' km/s', \
                   xy = (6300, 0.5), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        if ('ha_b' in nii_ha_sii_fit.submodel_names):
            ## Broad component
            ha.plot(lam_nii_ha_sii, nii_ha_sii_fit['ha_b'](lam_nii_ha_sii) + ha_cont, \
                    color = 'blue', lw = 2.0)
            sig_ha_b = table['HA_B_SIGMA'].data[index]
            fwhm_ha_b = mfit.sigma_to_fwhm(sig_ha_b)
            ha.annotate('$\\rm FWHM (H\\alpha;b)$ = '+str(round(fwhm_ha_b, 1))+' km/s', \
                       xy = (6680, 0.9), xycoords = ha.get_xaxis_transform(), \
                       fontsize = 20, color = 'k')

        ha.annotate('$\\rm [NII]6548,6582 + H\\alpha + [SII]6716,6731$', xy = (6300, 0.9), \
                   xycoords = ha.get_xaxis_transform(), fontsize = 20, color = 'k')

        ha.annotate('$\chi^{2}_{red}$ = '+str(round(nii_ha_sii_rchi2, 2)), \
                   xy = (6300, 0.8), xycoords = ha.get_xaxis_transform(), \
                   fontsize = 20, color = 'k')

        ## [NII]+Ha+[SII] fit residuals
        res_nii_ha_sii = (flam_nii_ha_sii - nii_ha_sii_fit(lam_nii_ha_sii))/flam_nii_ha_sii
        ha_res.scatter(lam_nii_ha_sii, res_nii_ha_sii, color = 'k', marker = '.', alpha = 0.7)
        ha_res.axhline(0.0, color = 'k', ls = ':')
        ha_res.set(xlabel = '$\lambda_{rest}$', ylim = [-10, 10], yticks = [0])

        plt.tight_layout()
        plt.close()

        return (fig)

####################################################################################################
####################################################################################################

def plot_from_params(table, index, ha_xlim = None, title = None):
    """
    Function to plot the spectra+fits from the table of parameters.
    
    Parameters
    ----------
    table : Astropy table
        Table of fit parameters
        
    index : int
        Index number of the source in the table
        
    title : str
        Title of the plot. Default is None.
        
    Returns
    -------
    fig : Figure
        Figure with spectra + best-fit model
    """
    
    if (table['HB_NDOF'].data[index] != 0):
        ## Default mode fit
        fig = plot_fits_from_table.normal_fit(table, index, ha_xlim, title = title)
        
    else:
        ## EBL mode fit
        fig = plot_fits_from_table.extreme_fit(table, index, title = title)
    
    return (fig)

####################################################################################################
####################################################################################################

def plot_fits(table, index, title = None, plot_smooth_cont = False):
    """
    Function to plot the spectra+fits from table.
    Old Version of the code
    
    Parameters
    ----------
    table : Astropy table
        Table of fit parameters
        
    index : int
        Index number of the source in the table
        
    title : str
        Title of the plot. Default is None.
        
    plot_smooth_cont : bool
        Whether or not to overplot the smooth continuum model.
        Default is False.
        
    Returns
    -------
    fig : Figure
        Figure with spectra + best-fit model
    """

    ## Required variables
    specprod = table['SPECPROD'].astype(str).data[index]
    survey = table['SURVEY'].astype(str).data[index]
    program = table['PROGRAM'].astype(str).data[index]
    healpix = table['HEALPIX'].data[index]
    targetid = table['TARGETID'].data[index]
    z = table['Z'].data[index]
    
    ## Accessing the rest-frame spectra
    _, lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                                     healpix, targetid, z, \
                                                                     rest_frame = True)
    
    if (plot_smooth_cont == True):
    ## Get the smooth continuum
        _, _, smooth_cont,_ = spec_utils.find_fastspec_models(specprod, survey, \
                                                              program, healpix, targetid)
        smooth_cont_rest = smooth_cont*(1+z)
    
    if (title == None):
        title = f'TARGETID: {targetid}; z : {round(z, 2)}'
    
    if (table['HB_RCHI2'].data[index] != 0):
        ## Constructing the fits from the table
        fits = emfit.construct_fits_from_table.normal_fit(table, index)

        ## Reduced chi2 values from the table
        hb_rchi2 = table['HB_RCHI2'].data[index]
        oiii_rchi2 = table['OIII_RCHI2'].data[index]
        nii_ha_rchi2 = table['NII_HA_RCHI2'].data[index]
        sii_rchi2 = table['SII_RCHI2'].data[index]

        rchi2s = [hb_rchi2, oiii_rchi2, nii_ha_rchi2, sii_rchi2]
        
        ## Plot the figure
        fig = plot_spectra_fits.normal_fit(lam_rest, flam_rest, fits, rchi2s, title = title, \
                                           smooth_cont = smooth_cont_rest, \
                                           plot_smooth_cont = plot_smooth_cont)
        
    else:
        ## Constructing the fits from the table
        fits = emfit.construct_fits_from_table.extreme_fit(table, index)
        
        ## Reduced chi2 values from the table
        hb_oiii_rchi2 = table['HB_OIII_RCHI2'].data[index]
        nii_ha_sii_rchi2 = table['NII_HA_SII_RCHI2'].data[index]
        
        rchi2s = [hb_oiii_rchi2, nii_ha_sii_rchi2]
        
        ## Plot the figure
        fig = plot_spectra_fits.extreme_fit(lam_rest, flam_rest, fits, rchi2s, title = title, \
                                            smooth_cont = smooth_cont_rest, \
                                            plot_smooth_cont = plot_smooth_cont)    
    
    return (fig)

####################################################################################################
####################################################################################################
