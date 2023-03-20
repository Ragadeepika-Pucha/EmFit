"""
The functions in this script are useful for fitting different emission-lines.

Ragadeepika Pucha
Version : 2023, March 14
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D

import fit_utils

###################################################################################################

def fit_sii_lines(lam_sii, flam_sii, ivar_sii):
    """
    Function to fit [SII]-doublet 6716, 6731 emission lines.
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
    
    ## Initial estimate of amplitudes
    amp_sii = max(flam_sii)
    
    #####################################################################################
    ########################### One-component fit #######################################
    
    ## Initial gaussian fits  
    g_sii6716 = Gaussian1D(amplitude = amp_sii, mean = 6718.294, \
                           stddev = 1.0, name = 'sii6716')
    g_sii6731 = Gaussian1D(amplitude = amp_sii, mean = 6732.673, \
                           stddev = 1.0, name = 'sii6731')
    
    ## Set amplitudes > 0
    g_sii6716.amplitude.bounds = (0.0, None)
    g_sii6731.amplitude.bounds = (0.0, None)
    
    ## Tie means of the two gaussians
    def tie_mean_sii(model):
        return (model['sii6716'].mean + 14.329)
    
    g_sii6731.mean.tied = tie_mean_sii
    
    ## Tie standard deviations of the two gaussians
    def tie_std_sii(model):
        return (model['sii6716'].stddev)*(model['sii6731'].mean/model['sii6716'].mean)
    
    g_sii6731.stddev.tied = tie_std_sii
    
    ## Initial Gaussian fit
    g_init = g_sii6716 + g_sii6731
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    ## Fit
    gfit_1comp = fitter(g_init, lam_sii, flam_sii, weights = np.sqrt(ivar_sii), maxiter = 300)
    rchi2_1comp = fit_utils.calculate_red_chi2(flam_sii, gfit_1comp(lam_sii), ivar_sii, n_free_params = 4)
    
    #####################################################################################
    ########################### Two-component fit #######################################
    
    ## Initial gaussian fits
    g_sii6716 = Gaussian1D(amplitude = amp_sii/2, mean = 6718.294, \
                           stddev = 1.0, name = 'sii6716')
    g_sii6731 = Gaussian1D(amplitude = amp_sii/2, mean = 6732.673, \
                           stddev = 1.0, name = 'sii6731')
    
    g_sii6716_out = Gaussian1D(amplitude = amp_sii/4, mean = 6718.294, \
                           stddev = 3.0, name = 'sii6716_out')
    g_sii6731_out = Gaussian1D(amplitude = amp_sii/4, mean = 6732.673, \
                           stddev = 3.0, name = 'sii6731_out')
    
    ## Set amplitudes > 0
    g_sii6716.amplitude.bounds = (0.0, None)
    g_sii6716_out.amplitude.bounds = (0.0, None)
    g_sii6731.amplitude.bounds = (0.0, None)
    g_sii6731_out.amplitude.bounds = (0.0, None)
    
    ## Tie means of the main gaussian components
    def tie_mean_sii(model):
        return (model['sii6716'].mean + 14.379)
    
    g_sii6731.mean.tied = tie_mean_sii
    
    ## Tie standard deviations of the main gaussian components
    def tie_std_sii(model):
        return (model['sii6716'].stddev)*(model['sii6731'].mean/model['sii6716'].mean)
    
    g_sii6731.stddev.tied = tie_std_sii
    
    ## Tie means of the outflow components
    def tie_mean_sii_out(model):
        return (model['sii6716_out'].mean + 14.379)
    
    g_sii6731_out.mean.tied = tie_mean_sii_out
    
    ## Tie standard deviations of the outflow components
    def tie_std_sii_out(model):
        return (model['sii6716_out'].stddev)*(model['sii6731_out'].mean/model['sii6716_out'].mean)
    
    g_sii6731_out.stddev.tied = tie_std_sii_out
    
    ## Tie amplitudes of all the four components
    def tie_amp_sii(model):
        return ((model['sii6731'].amplitude/model['sii6716'].amplitude)*model['sii6716_out'].amplitude)
    
    g_sii6731_out.amplitude.tied = tie_amp_sii
    
    ## Initial gaussian
    g_init = g_sii6716 + g_sii6731 + g_sii6716_out + g_sii6731_out
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_2comp = fitter(g_init, lam_sii, flam_sii, weights = np.sqrt(ivar_sii), maxiter = 300)
    rchi2_2comp = fit_utils.calculate_red_chi2(flam_sii, gfit_2comp(lam_sii), ivar_sii, n_free_params = 7)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100
    
    if (del_rchi2 >= 20):
        return (gfit_2comp, rchi2_2comp)
    else:
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################

def fit_oiii_lines(lam_oiii, flam_oiii, ivar_oiii):
    """
    Function to fit [OIII]-doublet 4959, 5007 emission lines.
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
    
    # Find initial estimates of amplitudes
    amp_oiii4959 = np.max(flam_oiii[(lam_oiii >= 4959)&(lam_oiii <= 4961)])
    amp_oiii5007 = np.max(flam_oiii[(lam_oiii >= 5007)&(lam_oiii <= 5009)])
    
    #####################################################################################
    ########################### One-component fit #######################################
    
    ## Initial gaussian fits
    g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959, mean = 4960.295, \
                            stddev = 1.0, name = 'oiii4959')
    g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007, mean = 5008.239, \
                          stddev = 1.0, name = 'oiii5007')
    
    ## Set amplitudes > 0
    g_oiii4959.amplitude.bounds = (0.0, None)
    g_oiii5007.amplitude.bounds = (0.0, None)
    
    ## Tie Means of the two gaussians
    def tie_mean_oiii(model):
        return (model['oiii4959'].mean + 47.934)

    g_oiii5007.mean.tied = tie_mean_oiii

    ## Tie Amplitudes of the two gaussians
    def tie_amp_oiii(model):
        return (model['oiii4959'].amplitude*2.98)

    g_oiii5007.amplitude.tied = tie_amp_oiii

    ## Tie standard deviations in velocity space
    def tie_std_oiii(model):
        return (model['oiii4959'].stddev)*(model['oiii5007'].mean/model['oiii4959'].mean)

    g_oiii5007.stddev.tied = tie_std_oiii
    
    ## Initial Gaussian fit
    g_init = g_oiii4959 + g_oiii5007
    
    ## Fitter
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_1comp = fitter(g_init, lam_oiii, flam_oiii, weights = np.sqrt(ivar_oiii), maxiter = 300)
    rchi2_1comp = fit_utils.calculate_red_chi2(flam_oiii, gfit_1comp(lam_oiii), ivar_oiii, n_free_params = 3) 
    
    #####################################################################################
    ########################### Two-component fit #######################################
    
    ## Initial gaussians
    g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959/2, mean = 4960.295, \
                            stddev = 1.0, name = 'oiii4959')
    g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007/2, mean = 5008.239, \
                          stddev = 1.0, name = 'oiii5007')
    
    g_oiii4959_out = Gaussian1D(amplitude = amp_oiii4959/4, mean = 4960.295, \
                                stddev = 4.0, name = 'oiii4959_out')
    g_oiii5007_out = Gaussian1D(amplitude = amp_oiii5007/4, mean = 5008.239, \
                                stddev = 4.0, name = 'oiii5007_out')
    
    ## Set amplitudes > 0
    g_oiii4959.amplitude.bounds = (0.0, None)
    g_oiii5007.amplitude.bounds = (0.0, None)
    g_oiii4959_out.amplitude.bounds = (0.0, None)
    g_oiii5007_out.amplitude.bounds = (0.0, None)
    
    ## Tie Means of the two gaussians
    def tie_mean_oiii(model):
        return (model['oiii4959'].mean + 47.934)

    g_oiii5007.mean.tied = tie_mean_oiii

    ## Tie Amplitudes of the two gaussians
    def tie_amp_oiii(model):
        return (model['oiii4959'].amplitude*2.98)

    g_oiii5007.amplitude.tied = tie_amp_oiii

    ## Tie standard deviations in velocity space
    def tie_std_oiii(model):
        return (model['oiii4959'].stddev)*(model['oiii5007'].mean/model['oiii4959'].mean)

    g_oiii5007.stddev.tied = tie_std_oiii
    
    ## Tie Means of the two gaussian outflow components
    def tie_mean_oiii_out(model):
        return (model['oiii4959_out'].mean + 47.934)

    g_oiii5007_out.mean.tied = tie_mean_oiii_out

    ## Tie Amplitudes of the two gaussian outflow components
    def tie_amp_oiii_out(model):
        return (model['oiii4959_out'].amplitude*2.98)

    g_oiii5007_out.amplitude.tied = tie_amp_oiii_out

    ## Tie standard deviations of the outflow components in the velocity space
    def tie_std_oiii_out(model):
        return (model['oiii4959_out'].stddev)*(model['oiii5007_out'].mean/model['oiii4959_out'].mean)

    g_oiii5007_out.stddev.tied = tie_std_oiii_out
    
    ## Initial Gaussian fit
    g_init = g_oiii4959 + g_oiii5007 + g_oiii4959_out + g_oiii5007_out
    
    ## Fitter
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_2comp = fitter(g_init, lam_oiii, flam_oiii, weights = np.sqrt(ivar_oiii), maxiter = 300)
    rchi2_2comp = fit_utils.calculate_red_chi2(flam_oiii, gfit_2comp(lam_oiii), ivar_oiii, n_free_params = 6)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100
    
    if (del_rchi2 >= 20):
        return (gfit_2comp, rchi2_2comp)
    else:
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################

def fit_hb_line(lam_hb, flam_hb, ivar_hb):
    """
    Function to fit Hb emission line
    The code fits both with and without broad-component fits and picks the best version.
    The "with-broad" component fit needs to be >20% better to be picked.
    
    Parameters
    ----------
    lam_hb : numpy array
        Wavelength array of the Hb region where the fits need to be performed.
        
    flam_hb : numpy array
        Flux array of the spectra in the Hb region.
        
    ivar_hb : numpy array
        Inverse variance array of the spectra in the Hb region.
        
    Returns
    -------
    gfit : Astropy model
        Best-fit "without-broad" or "with-broad" component
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    
    ## Initial estimate of amplitude
    amp_hb = np.max(flam_hb)
    
    #####################################################################################
    ########################### Fit without broad component #############################
    
    ## Single component fit
    ## Set default value = 130 km/s
    g_hb = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                      stddev = 2.1, name = 'hb_n')
    
    ## Set amplitudes > 0
    g_hb.amplitude.bounds = (0.0, None)
    
    ## Set narrow Hb sigma between 55-500 km/s  
    g_hb.stddev.bounds = (0.9, 8.1)
        
    ## Initial fit
    g_init = g_hb
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)

    gfit_no_broad = fitter(g_init, lam_hb, flam_hb, \
                           weights = np.sqrt(ivar_hb), maxiter = 300)
    rchi2_no_broad = fit_utils.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                  ivar_hb, n_free_params = 3)
    
    #####################################################################################
    ########################### Fit with broad component ################################
    
    ## Two component fit
    ## Default narrow sigma = 130 km/s
    ## Default broad sigma -- double narrow sigma ~ 260 km/s
    g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                      stddev = 2.1, name = 'hb_n')
    g_hb_b = Gaussian1D(amplitude = amp_hb/3, mean = 4862.683, \
                      stddev = 4.2, name = 'hb_b')
    
    ## Set amplitudes > 0
    g_hb_n.amplitude.bounds = (0.0, None)
    g_hb_b.amplitude.bounds = (0.0, None)
    
    ## Set narrow Hb sigma between 55-500 km/s  
    g_hb_n.stddev.bounds = (0.9, 8.1)
    ## Set min broad Hb sigma ~ 200 km/s
    g_hb_b.stddev.bounds = (3.0, None)
    
    ## Initial fit
    g_init = g_hb_n + g_hb_b
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_broad = fitter(g_init, lam_hb, flam_hb, \
                        weights = np.sqrt(ivar_hb), maxiter = 300)
    rchi2_broad = fit_utils.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                               ivar_hb, n_free_params = 6)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    
    if (del_rchi2 >= 20):
        return (gfit_broad, rchi2_broad)
    else:
        return (gfit_no_broad, rchi2_no_broad)
    
####################################################################################################

def fit_nii_ha_lines(lam_nii, flam_nii, ivar_nii, hb_bestfit, sii_bestfit):
    """
    Function to fit [NII]-doublet 6548, 6583 + Ha emission lines.
    The code uses [SII] best fit as a template for [NII]-doublet 
    and uses Hb best fit as a template for narrow Ha component.
    The two-component fit needs to be >20% better to be picked.
    
    Parameters
    ----------
    lam_nii : numpy array
        Wavelength array of the [NII]+Ha region where the fits need to be performed.
        
    flam_nii : numpy array
        Flux array of the spectra in the [NII]+Ha region.
        
    ivar_nii : numpy array
        Inverse variance array of the spectra in the [NII]+Ha region.
        
    hb_bestfit : Astropy model
        Best fit model for the Hb emission-line.
        
    sii_bestfit : Astropy model
        Best fit model for the [SII] emission-lines.
        
    Returns
    -------
    gfit : Astropy model
        Best-fit 1 component or 2 component model
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    n_hb = hb_bestfit.n_submodels
    
    if (n_hb == 1):
        g_hb = hb_bestfit
    elif (n_hb == 2):
        g_hb = hb_bestfit['hb_n']
        
    ## Ha parameters
    ## Initial guess of amplitude
    amp_ha = amp_ha_n = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])
    ## Model narrow Ha as narrow Hb
    stddev_ha = (6564.312/g_hb.mean)*g_hb.stddev
    
    g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                      stddev = stddev_ha, name = 'ha_n')
    
    ## Set amplitude > 0
    g_ha_n.amplitude.bounds = (0.0, None)
    
    ## Tie standard deviation of Ha
    def tie_std_ha(model):
        return ((model['ha_n'].mean/g_hb.mean)*g_hb.stddev)
    
    g_ha_n.stddev.tied = tie_std_ha
    g_ha_n.stddev.fixed = True
    
    ## Broad Ha parameters
    g_ha_b = Gaussian1D(amplitude = amp_ha/3, mean = 6564.312, \
                        stddev = 2*stddev_ha, name = 'ha_b')
    
    ## Set amplitude > 0
    g_ha_b.amplitude.bounds
    
    ## [NII] parameters
    ## Model [NII] as [SII] including outflows
    n_sii = sii_bestfit.n_submodels
    
    if (n_sii == 2):
        ## If n = 2, no outflow components
        names = sii_bestfit.submodel_names
        
        ## Initial estimates of standard deviation
        stddev_nii6548 = (6549.852/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev
        stddev_nii6583 = (6585.277/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev
        
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])
        
        ## Single component fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548, mean = 6549.852, \
                               stddev = stddev_nii6548, name = 'nii6548')
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583')
        
        ## Set all amplitudes > 0
        g_nii6548.amplitude.bounds = (0.0, None)
        g_nii6583.amplitude.bounds = (0.0, None)
       
        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return (model['nii6548'].mean + 35.425)
        
        g_nii6583.mean.tied = tie_mean_nii
        
        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*3.05)
        
        g_nii6583.amplitude.tied = tie_amp_nii
        
        ## Tie standard deviations of all the narrow components
        def tie_std_nii6548(model):
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev)
        
        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True
        
        g_nii = g_nii6548 + g_nii6583
        
    else:
        ## If n = 4, two outflow components for [NII]
        names = sii_bestfit.submodel_names
        
        ## Initial estimates of standard deviation
        stddev_nii6548 = (6549.852/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev
        stddev_nii6583 = (6585.277/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev
        
        stddev_nii6548_out = (6549.852/sii_bestfit[names[2]].mean)*sii_bestfit[names[2]].stddev
        stddev_nii6583_out = (6585.277/sii_bestfit[names[2]].mean)*sii_bestfit[names[2]].stddev
        
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])
        
        ## Two component fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548/2, mean = 6549.852, \
                               stddev = stddev_nii6548, name = 'nii6548')
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583')
        
        g_nii6548_out = Gaussian1D(amplitude = amp_nii6548/4, mean = 6549.852, \
                               stddev = stddev_nii6548_out, name = 'nii6548_out')
        g_nii6583_out = Gaussian1D(amplitude = amp_nii6583/4, mean = 6585.277, \
                               stddev = stddev_nii6583_out, name = 'nii6583_out')

        ## Set all amplitudes > 0
        g_nii6548.amplitude.bounds = (0.0, None)
        g_nii6583.amplitude.bounds = (0.0, None)
        
        g_nii6548_out.amplitude.bounds = (0.0, None)
        g_nii6583_out.amplitude.bounds = (0.0, None)
        
        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return (model['nii6548'].mean + 35.425)
        
        g_nii6583.mean.tied = tie_mean_nii
        
        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*3.05)
        
        g_nii6583.amplitude.tied = tie_amp_nii
        
        ## Tie standard deviations of all the narrow components
        def tie_std_nii6548(model):
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*sii_bestfit[names[0]].stddev)
        
        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True
        
        ## Tie means of [NII] outflow components
        def tie_mean_nii_out(model):
            return (model['nii6548_out'].mean + 35.425)
        
        g_nii6583_out.mean.tied = tie_mean_nii_out
        
        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii_out(model):
            return (model['nii6548_out'].amplitude*3.05)
        
        g_nii6583_out.amplitude.tied = tie_amp_nii_out
        
        ## Tie standard deviations of all the outflow components
        def tie_std_nii6548_out(model):
            return ((model['nii6548_out'].mean/sii_bestfit[names[2]].mean)*sii_bestfit[names[2]].stddev)
        
        g_nii6548_out.stddev.tied = tie_std_nii6548_out
        g_nii6548_out.stddev.fixed = True
        
        def tie_std_nii6583_out(model):
            return ((model['nii6583_out'].mean/sii_bestfit[names[2]].mean)*sii_bestfit[names[2]].stddev)
        
        g_nii6583_out.stddev.tied = tie_std_nii6583_out
        g_nii6583_out.stddev.fixed = True
        
        g_nii = g_nii6548 + g_nii6583 + g_nii6548_out + g_nii6583_out
        
    #####################################################################################
    ########################## Fit without broad component ##############################
    
    ## Initial gaussian fit
    g_init = g_nii + g_ha_n
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_no_broad = fitter(g_init, lam_nii, flam_nii, weights = np.sqrt(ivar_nii), maxiter = 300)
    
    if (n_sii == 2):
        rchi2_no_broad = fit_utils.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii), ivar_nii, n_free_params = 4)
    else:
        rchi2_no_broad = fit_utils.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii), ivar_nii, n_free_params = 6)
    
    #####################################################################################
    ########################## Fit with broad component #################################
    
    ## Initial gaussian fit
    g_init = g_nii + g_ha_n + g_ha_b
    fitter = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_broad = fitter(g_init, lam_nii, flam_nii, weights = np.sqrt(ivar_nii), maxiter = 300)
    
    if (n_sii == 2):
        rchi2_broad = fit_utils.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), ivar_nii, n_free_params = 7)
    else:
        rchi2_broad = fit_utils.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), ivar_nii, n_free_params = 9)
        
    #####################################################################################
    #####################################################################################
        
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    
    if (del_rchi2 >= 20):
        return (gfit_broad, rchi2_broad)
    else:
        return (gfit_no_broad, rchi2_no_broad)
    
####################################################################################################