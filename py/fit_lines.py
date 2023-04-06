"""
The functions in this script are useful for fitting different emission-lines.

Ragadeepika Pucha
Version : 2023, April 5
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import fit_utils
import measure_fits as mfit

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
    fitter : Astropy fitter
        Fitter for computing uncertainties
    
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
    ## Set default sigma values to 130 km/s ~ 2.9 in wavelength space
    ## Set amplitudes > 0
    g_sii6716 = Gaussian1D(amplitude = amp_sii, mean = 6718.294, \
                           stddev = 2.9, name = 'sii6716', \
                           bounds = {'amplitude' : (0.0, None)})
    g_sii6731 = Gaussian1D(amplitude = amp_sii, mean = 6732.673, \
                           stddev = 2.9, name = 'sii6731', \
                           bounds = {'amplitude' : (0.0, None)})
    
    ## Tie means of the two gaussians
    def tie_mean_sii(model):
        return (model['sii6716'].mean + 14.329)
    
    g_sii6731.mean.tied = tie_mean_sii
    
    ## Tie standard deviations of the two gaussians
    def tie_std_sii(model):
        return ((model['sii6716'].stddev)*(model['sii6731'].mean/model['sii6716'].mean))
    
    g_sii6731.stddev.tied = tie_std_sii
    
    ## Initial Gaussian fit
    g_init = g_sii6716 + g_sii6731
    fitter_1comp = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    ## Fit
    gfit_1comp = fitter_1comp(g_init, lam_sii, flam_sii, \
                        weights = np.sqrt(ivar_sii), maxiter = 1000)
    rchi2_1comp = mfit.calculate_red_chi2(flam_sii, gfit_1comp(lam_sii),\
                                               ivar_sii, n_free_params = 4)
    
    #####################################################################################
    ########################### Two-component fit #######################################
    
    ## Initial gaussian fits
    ## Default values of sigma ~ 130 km/s ~ 2.9
    ## Set amplitudes > 0
    g_sii6716 = Gaussian1D(amplitude = amp_sii/2, mean = 6718.294, \
                           stddev = 2.9, name = 'sii6716', \
                          bounds = {'amplitude' : (0.0, None)})
    g_sii6731 = Gaussian1D(amplitude = amp_sii/2, mean = 6732.673, \
                           stddev = 2.9, name = 'sii6731', \
                          bounds = {'amplitude' : (0.0, None)})
    
    g_sii6716_out = Gaussian1D(amplitude = amp_sii/4, mean = 6718.294, \
                               stddev = 4.0, name = 'sii6716_out', \
                               bounds = {'amplitude' : (0.0, None)})
    g_sii6731_out = Gaussian1D(amplitude = amp_sii/4, mean = 6732.673, \
                               stddev = 4.0, name = 'sii6731_out', \
                               bounds = {'amplitude' : (0.0, None)})
    
    ## Tie means of the main gaussian components
    def tie_mean_sii(model):
        return (model['sii6716'].mean + 14.379)
    
    g_sii6731.mean.tied = tie_mean_sii
    
    ## Tie standard deviations of the main gaussian components
    def tie_std_sii(model):
        return ((model['sii6716'].stddev)*\
                (model['sii6731'].mean/model['sii6716'].mean))
    
    g_sii6731.stddev.tied = tie_std_sii
    
    ## Tie means of the outflow components
    def tie_mean_sii_out(model):
        return (model['sii6716_out'].mean + 14.379)
    
    g_sii6731_out.mean.tied = tie_mean_sii_out
    
    ## Tie standard deviations of the outflow components
    def tie_std_sii_out(model):
        return ((model['sii6716_out'].stddev)*\
                (model['sii6731_out'].mean/model['sii6716_out'].mean))
    
    g_sii6731_out.stddev.tied = tie_std_sii_out
    
    ## Tie amplitudes of all the four components
    def tie_amp_sii(model):
        return ((model['sii6731'].amplitude/model['sii6716'].amplitude)*\
                model['sii6716_out'].amplitude)
    
    g_sii6731_out.amplitude.tied = tie_amp_sii
    
    ## Initial gaussian
    g_init = g_sii6716 + g_sii6731 + g_sii6716_out + g_sii6731_out
    fitter_2comp = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_2comp = fitter_2comp(g_init, lam_sii, flam_sii, \
                        weights = np.sqrt(ivar_sii), maxiter = 1000)
    rchi2_2comp = mfit.calculate_red_chi2(flam_sii, gfit_2comp(lam_sii), \
                                               ivar_sii, n_free_params = 7)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100
    
    if (del_rchi2 >= 20):
        return (fitter_2comp, gfit_2comp, rchi2_2comp)
    else:
        return (fitter_1comp, gfit_1comp, rchi2_1comp)
    
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
    fitter : Astropy fitter
        Fitter for computing uncertainties
        
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
    ## Set default values of sigma ~ 130 km/s ~ 2.1
    ## Set amplitudes > 0
    
    g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959, mean = 4960.295, \
                            stddev = 2.1, name = 'oiii4959', \
                            bounds = {'amplitude' : (0.0, None)})
    g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007, mean = 5008.239, \
                            stddev = 2.1, name = 'oiii5007', \
                            bounds = {'amplitude' : (0.0, None)})
    
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
        return ((model['oiii4959'].stddev)*\
                (model['oiii5007'].mean/model['oiii4959'].mean))

    g_oiii5007.stddev.tied = tie_std_oiii
    
    ## Initial Gaussian fit
    g_init = g_oiii4959 + g_oiii5007
    
    ## Fitter
    fitter_1comp = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_1comp = fitter_1comp(g_init, lam_oiii, flam_oiii, \
                        weights = np.sqrt(ivar_oiii), maxiter = 1000)
    rchi2_1comp = mfit.calculate_red_chi2(flam_oiii, gfit_1comp(lam_oiii), \
                                               ivar_oiii, n_free_params = 3) 
    
    #####################################################################################
    ########################### Two-component fit #######################################
    
    ## Initial gaussians
    ## Set default values of sigma ~ 130 km/s ~ 2.1
    ## Set amplitudes > 0
    g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959/2, mean = 4960.295, \
                            stddev = 2.1, name = 'oiii4959', \
                            bounds = {'amplitude' : (0.0, None)})
    g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007/2, mean = 5008.239, \
                            stddev = 2.1, name = 'oiii5007', \
                            bounds = {'amplitude' : (0.0, None)})
    
    g_oiii4959_out = Gaussian1D(amplitude = amp_oiii4959/4, mean = 4960.295, \
                                stddev = 2.1, name = 'oiii4959_out', \
                                bounds = {'amplitude' : (0.0, None)})
    g_oiii5007_out = Gaussian1D(amplitude = amp_oiii5007/4, mean = 5008.239, \
                                stddev = 2.1, name = 'oiii5007_out', \
                                bounds = {'amplitude' : (0.0, None)})
    
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
        return ((model['oiii4959'].stddev)*\
                (model['oiii5007'].mean/model['oiii4959'].mean))

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
        return ((model['oiii4959_out'].stddev)*\
    (model['oiii5007_out'].mean/model['oiii4959_out'].mean))

    g_oiii5007_out.stddev.tied = tie_std_oiii_out
    
    ## Initial Gaussian fit
    g_init = g_oiii4959 + g_oiii5007 + g_oiii4959_out + g_oiii5007_out
    
    ## Fitter
    fitter_2comp = fitting.LevMarLSQFitter()
    
    gfit_2comp = fitter_2comp(g_init, lam_oiii, flam_oiii, \
                        weights = np.sqrt(ivar_oiii), maxiter = 1000)
    rchi2_2comp = mfit.calculate_red_chi2(flam_oiii, gfit_2comp(lam_oiii), \
                                               ivar_oiii, n_free_params = 6)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_1comp - rchi2_2comp)/rchi2_1comp)*100
    
    if (del_rchi2 >= 20):
        return (fitter_2comp, gfit_2comp, rchi2_2comp)
    else:
        return (fitter_1comp, gfit_1comp, rchi2_1comp)

####################################################################################################

def fit_hb_line(lam_hb, flam_hb, ivar_hb, outflow = False):
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
    fitter : Astropy fitter
        Fitter for computing uncertainties
    
    gfit : Astropy model
        Best-fit "without-broad" or "with-broad" component
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    ## Initial estimate of amplitude
    amp_hb = np.max(flam_hb)
    
    if (outflow == False):
        ## Single component fit
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                          stddev = 1.0, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None), 'stddev' : (None, 8.1)})
        
        g_hb = g_hb_n
        
    else:
        ## Two component fit
        g_hb_n = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                          stddev = 1.0, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None), 'stddev' : (None, 8.1)})
        
        g_hb_out = Gaussian1D(amplitude = amp_hb/4, mean = 4862.683, \
                              stddev = 2.0, name = 'hb_out', \
                              bounds = {'amplitude' : (0.0, None)})
        
        g_hb = g_hb_n + g_hb_out
    
    #####################################################################################
    ########################### Fit without broad component #############################
            
    ## Initial fit
    g_init = g_hb 
    fitter_no_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)

    gfit_no_broad = fitter_no_broad(g_init, lam_hb, flam_hb, \
                           weights = np.sqrt(ivar_hb), maxiter = 1000)
    
    if (outflow == False):
        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                      ivar_hb, n_free_params = 3)
    else:
        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                      ivar_hb, n_free_params = 6)
    
    #####################################################################################
    ########################### Fit with broad component ################################
    
    g_hb_b = Gaussian1D(amplitude = amp_hb/3, mean = 4862.683, \
                        stddev = 2.0, name = 'hb_b', \
                        bounds = {'amplitude' : (0.0, None)})
    
    ## Initial fit
    g_init = g_hb + g_hb_b 
    fitter_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_broad = fitter_broad(g_init, lam_hb, flam_hb, \
                        weights = np.sqrt(ivar_hb), maxiter = 1000)
    
    if (outflow == False):
        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                                   ivar_hb, n_free_params = 6)
    else:
        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                                   ivar_hb, n_free_params = 9)
    
    #####################################################################################
    #####################################################################################
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    
    if (del_rchi2 >= 20):
        return (fitter_broad, gfit_broad, rchi2_broad)
    else:
        return (fitter_no_broad, gfit_no_broad, rchi2_no_broad)
    
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
    fitter : Astropy fitter
        Fitter for computing uncertainties
    
    gfit : Astropy model
        Best-fit 1 component or 2 component model
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    ## Matching Ha as Hb
    ## Number of submodels for Hb best fit
    ## if n_hb = 1, no broad component
    n_hb = hb_bestfit.n_submodels
    
    ## Considering only the narrow component, if there is a broad component available.
    if (n_hb == 1):
        g_hb = hb_bestfit
    elif (n_hb == 2):
        g_hb = hb_bestfit['hb_n']
        
    ## Ha parameters
    ## Initial guess of amplitude
    amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])
    ## Model narrow Ha as narrow Hb
    stddev_ha = (6564.312/g_hb.mean)*g_hb.stddev
    
    g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                        stddev = stddev_ha, name = 'ha_n', \
                        bounds = {'amplitude' : (0.0, None)})
    
    ## Tie standard deviation of Ha
    def tie_std_ha(model):
        return ((model['ha_n'].mean/g_hb.mean)*g_hb.stddev)
    
    g_ha_n.stddev.tied = tie_std_ha
    g_ha_n.stddev.fixed = True
    
    ## Broad Ha parameters
    g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                        stddev = 3.0, name = 'ha_b', \
                        bounds = {'amplitude' : (0.0, None)})
    
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
                               stddev = stddev_nii6548, name = 'nii6548', \
                               bounds = {'amplitude' : (0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583', \
                               bounds = {'amplitude' : (0.0, None)})
       
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
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
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
                               stddev = stddev_nii6548, name = 'nii6548', \
                               bounds = {'amplitude' : (0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583', \
                               bounds = {'amplitude' : (0.0, None)})
        
        g_nii6548_out = Gaussian1D(amplitude = amp_nii6548/4, mean = 6549.852, \
                                   stddev = stddev_nii6548_out, name = 'nii6548_out', \
                                   bounds = {'amplitude' : (0.0, None)})
        g_nii6583_out = Gaussian1D(amplitude = amp_nii6583/4, mean = 6585.277, \
                                   stddev = stddev_nii6583_out, name = 'nii6583_out', \
                                   bounds = {'amplitude' : (0.0, None)})
        
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
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
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
            return ((model['nii6548_out'].mean/sii_bestfit[names[2]].mean)*\
                    sii_bestfit[names[2]].stddev)
        
        g_nii6548_out.stddev.tied = tie_std_nii6548_out
        g_nii6548_out.stddev.fixed = True
        
        def tie_std_nii6583_out(model):
            return ((model['nii6583_out'].mean/sii_bestfit[names[2]].mean)*\
                    sii_bestfit[names[2]].stddev)
        
        g_nii6583_out.stddev.tied = tie_std_nii6583_out
        g_nii6583_out.stddev.fixed = True
        
        g_nii = g_nii6548 + g_nii6583 + g_nii6548_out + g_nii6583_out
        
    #####################################################################################
    ########################## Fit without broad component ##############################
    
    ## Initial gaussian fit
    g_init = g_nii + g_ha_n
    fitter_no_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii, \
                           weights = np.sqrt(ivar_nii), maxiter = 1000)
    
    if (n_sii == 2):
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii), \
                                                      ivar_nii, n_free_params = 4)
    else:
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                      ivar_nii, n_free_params = 6)
    
    #####################################################################################
    ########################## Fit with broad component #################################
    
    ## Initial gaussian fit
    g_init = g_nii + g_ha_n + g_ha_b
    fitter_broad = fitting.LevMarLSQFitter()
    
    gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                        weights = np.sqrt(ivar_nii), maxiter = 1000)
    
    if (n_sii == 2):
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii),\
                                                   ivar_nii, n_free_params = 7)
    else:
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                   ivar_nii, n_free_params = 9)
        
    #####################################################################################
    #####################################################################################
        
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    
    if (del_rchi2 >= 20):
        return (fitter_broad, gfit_broad, rchi2_broad)
    else:
        return (fitter_no_broad, gfit_no_broad, rchi2_no_broad)
    
    
####################################################################################################

def fit_hb_line_template(lam_hb, flam_hb, ivar_hb, sii_bestfit, frac_temp = 40):
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
        
    temp_fit : astropy model fit
        Template fit for narrow Hbeta
        Sigma of narrow Hb bounds are set to be within 20% of the template fit
        
    frac_temp : float
        The %age of [SII] width within which narrow Hbeta width can vary
        
    Returns
    -------
    fitter : Astropy fitter
        Fitter for computing uncertainties
        
    gfit : Astropy model
        Best-fit "without-broad" or "with-broad" component
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    n_sii = sii_bestfit.n_submodels
    ## If n_sii = 2, no outflow components
    ## If n_sii = 4, outflow components
    
    ## Template fit
    temp_std = sii_bestfit['sii6716'].stddev.value
    temp_std_kms = mfit.lamspace_to_velspace(temp_std, 6718.294)
    
    min_std_kms = temp_std_kms - ((frac_temp/100)*temp_std_kms)
    max_std_kms = temp_std_kms + ((frac_temp/100)*temp_std_kms)
    
    min_std = mfit.velspace_to_lamspace(min_std_kms, 4862.683)
    max_std = mfit.velspace_to_lamspace(max_std_kms, 4862.683)
    
    ## Initial estimate of amplitude
    amp_hb = np.max(flam_hb)
    
    if (n_sii == 2):
        ## No outflow components
    
        ## Single component fit
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                          stddev = 1.0, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None)})
        
        g_hb_n.stddev.bounds = (min_std, max_std)
        
        g_hb = g_hb_n
    
    else:
        ## Outflow components
        temp_out_std = sii_bestfit['sii6716_out'].stddev.value
        temp_out_std_kms = mfit.lamspace_to_velspace(temp_out_std, 6718.294)
        
        min_out_kms = temp_out_std_kms - ((frac_temp/100)*temp_out_std_kms)
        max_out_kms = temp_out_std_kms + ((frac_temp/100)*temp_out_std_kms)
        
        min_out = mfit.velspace_to_lamspace(min_out_kms, 4862.683)
        max_out = mfit.velspace_to_lamspace(max_out_kms, 4862.683)
        
        ## Two component fit for the narrow Hb
        g_hb_n = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                            stddev = 1.0, name = 'hb_n', \
                            bounds = {'amplitude' : (0.0, None)})
        g_hb_out = Gaussian1D(amplitude = amp_hb/4, mean = 4862.683, \
                              stddev = 2.0, name = 'hb_out', \
                              bounds = {'amplitude' : (0.0, None)})
        
        g_hb_n.stddev.bounds = (min_std, max_std)
        g_hb_out.stddev.bounds = (min_out, max_out)
        
        g_hb = g_hb_n + g_hb_out
    
    #####################################################################################
    ########################### Fit without broad component #############################
        
    ## Initial fit
    g_init = g_hb 
    fitter_no_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)

    gfit_no_broad = fitter_no_broad(g_init, lam_hb, flam_hb, \
                                    weights = np.sqrt(ivar_hb), maxiter = 1000)
    
    if (n_sii == 2):
        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                 ivar_hb, n_free_params = 3)
    else:
        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                 ivar_hb, n_free_params = 6)
    
    #####################################################################################
    ########################### Fit with broad component ################################
    
    ## Two component fit
    ## Default narrow sigma = 130 km/s
    ## Default broad sigma -- double narrow sigma ~ 260 km/s
    g_hb_b = Gaussian1D(amplitude = amp_hb/3, mean = 4862.683, \
                        stddev = 2.0, name = 'hb_b', \
                        bounds = {'amplitude' : (0.0, None)})
    
    ## Initial fit
    g_init = g_hb + g_hb_b 
    fitter_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)
    
    gfit_broad = fitter_broad(g_init, lam_hb, flam_hb, \
                              weights = np.sqrt(ivar_hb), maxiter = 1000)
    
    if (n_sii == 2):
        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                              ivar_hb, n_free_params = 6)
    else:
        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                              ivar_hb, n_free_params = 9)
    
    #####################################################################################
    #####################################################################################
    
    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    
    if (del_rchi2 >= 20):
        return (fitter_broad, gfit_broad, rchi2_broad)
    else:
        return (fitter_no_broad, gfit_no_broad, rchi2_no_broad)
    
####################################################################################################

def fit_nii_ha_lines_template(lam_nii, flam_nii, ivar_nii, sii_bestfit, frac_temp = 40.):
    """
    Function to fit [NII]-doublet 6548, 6583 + Ha emission lines.
    The code uses a template fit for narrow and outflow components for [NII]
    The sigma values of the narrow Ha is bound to be 
    within some perfect of [SII] width of the template fits
    The two-component fit needs to be >20% better to be picked.
    
    Parameters
    ----------
    lam_nii : numpy array
        Wavelength array of the [NII]+Ha region where the fits need to be performed.
        
    flam_nii : numpy array
        Flux array of the spectra in the [NII]+Ha region.
        
    ivar_nii : numpy array
        Inverse variance array of the spectra in the [NII]+Ha region.
        
    sii_bestfit : Astropy Model
        Best fit for the [SII] lines
        
    frac_temp : float
        The %age of [SII] width within which narrow Halpha width can vary
        
    Returns
    -------
    gfit : Astropy model
        Best-fit 1 component or 2 component model
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    ## Template fit
    ## If AoN ([SII]) > 3, use sigma values of narrow lines within 20% of the template [SII]
    ## If AoN ([OIII]) > 3, use sigma values of narrow lines within 20% of the template [OIII]
    temp_std = sii_bestfit['sii6716'].stddev.value
    temp_std_kms = mfit.lamspace_to_velspace(temp_std, 6718.294)
    
    min_std_kms = temp_std_kms - ((frac_temp/100)*temp_std_kms)
    max_std_kms = temp_std_kms + ((frac_temp/100)*temp_std_kms)

    min_std_ha = mfit.velspace_to_lamspace(min_std_kms, 6549.852)
    max_std_ha = mfit.velspace_to_lamspace(max_std_kms, 6549.852)
    
    ## Ha parameters
    ## Initial guess of amplitude
    amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])

    g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                        stddev = temp_std, name = 'ha_n', \
                        bounds = {'amplitude' : (0.0, None)})

    ## Set narrow Ha within 20% of the template fit
    g_ha_n.stddev.bounds = (min_std_ha, max_std_ha)

    ## Broad Ha parameters
    g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                        stddev = 3.0, name = 'ha_b', \
                        bounds = {'amplitude' : (0.0, None), 'stddev' : (3.1, None)})

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
                               stddev = stddev_nii6548, name = 'nii6548', \
                               bounds = {'amplitude' : (0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583', \
                               bounds = {'amplitude' : (0.0, None)})
       
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
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
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
                               stddev = stddev_nii6548, name = 'nii6548', \
                               bounds = {'amplitude' : (0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
                               stddev = stddev_nii6583, name = 'nii6583', \
                               bounds = {'amplitude' : (0.0, None)})
        
        g_nii6548_out = Gaussian1D(amplitude = amp_nii6548/4, mean = 6549.852, \
                                   stddev = stddev_nii6548_out, name = 'nii6548_out', \
                                   bounds = {'amplitude' : (0.0, None)})
        g_nii6583_out = Gaussian1D(amplitude = amp_nii6583/4, mean = 6585.277, \
                                   stddev = stddev_nii6583_out, name = 'nii6583_out', \
                                   bounds = {'amplitude' : (0.0, None)})
        
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
            return ((model['nii6548'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True
        
        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit[names[0]].mean)*\
                    sii_bestfit[names[0]].stddev)
        
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
            return ((model['nii6548_out'].mean/sii_bestfit[names[2]].mean)*\
                    sii_bestfit[names[2]].stddev)
        
        g_nii6548_out.stddev.tied = tie_std_nii6548_out
        g_nii6548_out.stddev.fixed = True
        
        def tie_std_nii6583_out(model):
            return ((model['nii6583_out'].mean/sii_bestfit[names[2]].mean)*\
                    sii_bestfit[names[2]].stddev)
        
        g_nii6583_out.stddev.tied = tie_std_nii6583_out
        g_nii6583_out.stddev.fixed = True
        
        g_nii = g_nii6548 + g_nii6583 + g_nii6548_out + g_nii6583_out
        
    #####################################################################################
    ########################## Fit without broad component ##############################

    ## Initial gaussian fit
    g_init = g_nii + g_ha_n
    
    fitter_no_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)

    gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii,\
                                 weights = np.sqrt(ivar_nii), maxiter = 1000)

    if (n_sii == 2):
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                      ivar_nii, n_free_params = 5)
    else:
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii), \
                                                      ivar_nii, n_free_params = 7)

    #####################################################################################
    ########################## Fit with broad component #################################

    ## Initial gaussian fit
    g_init = g_nii + g_ha_n + g_ha_b
    fitter_broad = fitting.LevMarLSQFitter()

    gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                              weights = np.sqrt(ivar_nii), maxiter = 1000)

    if (n_sii == 2):
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                   ivar_nii, n_free_params = 9)
    else:
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                   ivar_nii, n_free_params = 11)

    #####################################################################################
    #####################################################################################

    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    #print (rchi2_no_broad, rchi2_broad)

    if (del_rchi2 >= 20):
        if (gfit_broad['ha_b'].stddev.value < gfit_broad['ha_n'].stddev.value):
            g_ha_n = Gaussian1D(amplitude = gfit_broad['ha_b'].amplitude, \
                                mean = gfit_broad['ha_b'].mean, \
                                stddev = gfit_broad['ha_b'].stddev, \
                                name = 'ha_n')
            g_ha_b = Gaussian1D(amplitude = gfit_broad['ha_n'].amplitude, \
                                mean = gfit_broad['ha_n'].mean, \
                                stddev = gfit_broad['ha_n'].stddev, \
                                name = 'ha_b')
            gfit_broad = g_nii + g_ha_n + g_ha_b
        return (fitter_broad, gfit_broad, rchi2_broad)
    else:
        return (fitter_no_broad, gfit_no_broad, rchi2_no_broad)

####################################################################################################

def fit_nii_ha_lines_template1(lam_nii, flam_nii, ivar_nii, temp_fit, \
                              frac_temp = 40, temp_out_fit = None):
    """
    Function to fit [NII]-doublet 6548, 6583 + Ha emission lines.
    The code uses a template fit for narrow and outflow components.
    The sigma values of the narrow components is bound to be 
    within 20% of the template fits
    The two-component fit needs to be >20% better to be picked.
    
    Parameters
    ----------
    lam_nii : numpy array
        Wavelength array of the [NII]+Ha region where the fits need to be performed.
        
    flam_nii : numpy array
        Flux array of the spectra in the [NII]+Ha region.
        
    ivar_nii : numpy array
        Inverse variance array of the spectra in the [NII]+Ha region.
        
    temp_fit : Astropy model
        Template fit for the narrow lines
        
    temp_out_fit : Astropy model
        Template fit for the outflow components
        
    frac_temp : float
        The %age of [SII] width within which narrow Hbeta width can vary
        
    Returns
    -------
    gfit : Astropy model
        Best-fit 1 component or 2 component model
        
    rchi2: float
        Reduced chi2 of the best-fit
    """
    
    ## Template fit
    ## If AoN ([SII]) > 3, use sigma values of narrow lines within 20% of the template [SII]
    ## If AoN ([OIII]) > 3, use sigma values of narrow lines within 20% of the template [OIII]
    temp_std = temp_fit.stddev.value
    temp_std_kms = mfit.lamspace_to_velspace(temp_std, 6718.294)
    
    min_std_kms = temp_std_kms - ((frac_temp/100)*temp_std_kms)
    max_std_kms = temp_std_kms + ((frac_temp/100)*temp_std_kms)
    
    min_std = mfit.velspace_to_lamspace(min_std_kms, 6549.852)
    max_std = mfit.velspace_to_lamspace(max_std_kms, 6549.852)
    
    min_std_ha = mfit.velspace_to_lamspace(min_std_kms, 6549.852)
    max_std_ha = mfit.velspace_to_lamspace(max_std_kms, 6549.852)

    ## Ha parameters
    ## Initial guess of amplitude
    amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])

    g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                        stddev = temp_std, name = 'ha_n')

    ## Set amplitude > 0
    g_ha_n.amplitude.bounds = (0.0, None)

    ## Set narrow Ha within 20% of the template fit
    g_ha_n.stddev.bounds = (min_std_ha, max_std_ha)

    ## Broad Ha parameters
    g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                        stddev = 3.0, name = 'ha_b')

    ## Set amplitude > 0
    g_ha_b.amplitude.bounds = (0.0, None)
    g_ha_b.stddev.bounds = (3.2, None)

    ## [NII] parameters
    ## Model [NII] as [SII]/[OIII] within 20% including outflows

    if (temp_out_fit is None):
        ## No outflow components
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])

        ## Single component fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548, mean = 6549.852, \
                           stddev = temp_std, name = 'nii6548')
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                               stddev = temp_std, name = 'nii6583')

        ## Set all amplitudes > 0
        g_nii6548.amplitude.bounds = (0.0, None)
        g_nii6583.amplitude.bounds = (0.0, None)

        ## Set narrow [NII] within 20% of the template fit
        g_nii6548.stddev.bounds = (min_std, max_std)
        g_nii6583.stddev.bounds = (min_std, max_std)

        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return (model['nii6548'].mean + 35.425)

        g_nii6583.mean.tied = tie_mean_nii

        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*3.05)

        g_nii6583.amplitude.tied = tie_amp_nii

        ## Tie standard deviations together
        def tie_std_nii(model):
            return ((model['nii6548'].stddev)*\
                    (model['nii6583'].mean/model['nii6548'].mean))

        g_nii6583.stddev.tied = tie_std_nii

        g_nii = g_nii6548 + g_nii6583

    else:
        temp_out_std = temp_out_fit.stddev.value
        temp_out_std_kms = mfit.lamspace_to_velspace(temp_out_std, 6718.294)
        
        min_out_kms = temp_out_std_kms - ((frac_temp/100)*temp_out_std_kms)
        max_out_kms = temp_out_std_kms + ((frac_temp/100)*temp_out_std_kms)
        
        min_out = mfit.velspace_to_lamspace(min_out_kms, 6549.852)
        max_out = mfit.velspace_to_lamspace(max_out_kms, 6549.852)

        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])

        ## Two component fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548/2, mean = 6549.852, \
                               stddev = temp_std, name = 'nii6548')
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
                               stddev = temp_std, name = 'nii6583')

        g_nii6548_out = Gaussian1D(amplitude = amp_nii6548/4, mean = 6549.852, \
                               stddev = temp_out_std, name = 'nii6548_out')
        g_nii6583_out = Gaussian1D(amplitude = amp_nii6583/4, mean = 6585.277, \
                               stddev = temp_out_std, name = 'nii6583_out')

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

        ## Set sigma of [NII] within 20% of [SII] or [OIII]
        g_nii6548.stddev.bounds = (min_std, max_std)
        g_nii6583.stddev.bounds = (min_std, max_std)

        ## Tie standard deviations together
        def tie_std_nii(model):
            return ((model['nii6548'].stddev)*\
                    (model['nii6583'].mean/model['nii6548'].mean))

        g_nii6583.stddev.tied = tie_std_nii

        ## Tie means of [NII] outflow components
        def tie_mean_nii_out(model):
            return (model['nii6548_out'].mean + 35.425)

        g_nii6583_out.mean.tied = tie_mean_nii_out

        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii_out(model):
            return (model['nii6548_out'].amplitude*3.05)

        g_nii6583_out.amplitude.tied = tie_amp_nii_out

        ## Set sigma of [NII] outflows within 20% of [SII] or [OIII] outflows
        g_nii6548_out.stddev.bounds = (min_out, max_out)
        g_nii6583_out.stddev.bounds = (min_out, max_out)

        ## Tie standard deviations together
        def tie_std_nii_out(model):
            return ((model['nii6548_out'].stddev)*\
                    (model['nii6583_out'].mean/model['nii6548_out'].mean))

        g_nii6583_out.stddev.tied = tie_std_nii_out

        g_nii = g_nii6548 + g_nii6583 + g_nii6548_out + g_nii6583_out

    #####################################################################################
    ########################## Fit without broad component ##############################

    ## Initial gaussian fit
    g_init = g_nii + g_ha_n
    
    fitter_no_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)

    gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii,\
                                 weights = np.sqrt(ivar_nii), maxiter = 1000)

    if (temp_out_fit is None):
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                      ivar_nii, n_free_params = 6)
    else:
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii), \
                                                      ivar_nii, n_free_params = 9)

    #####################################################################################
    ########################## Fit with broad component #################################

    ## Initial gaussian fit
    g_init = g_nii + g_ha_n + g_ha_b
    fitter_broad = fitting.LevMarLSQFitter()

    gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                              weights = np.sqrt(ivar_nii), maxiter = 1000)

    if (temp_out_fit is not None):
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                   ivar_nii, n_free_params = 9)
    else:
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                   ivar_nii, n_free_params = 12)

    #####################################################################################
    #####################################################################################

    ## Select the best-fit based on rchi2
    ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
    ## Otherwise, 1-component fit is the best fit.
    del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
    #print (rchi2_no_broad, rchi2_broad)

    if (del_rchi2 >= 20):
        if (gfit_broad['ha_b'].stddev.value < gfit_broad['ha_n'].stddev.value):
            g_ha_n = Gaussian1D(amplitude = gfit_broad['ha_b'].amplitude, \
                                mean = gfit_broad['ha_b'].mean, \
                                stddev = gfit_broad['ha_b'].stddev, \
                                name = 'ha_n')
            g_ha_b = Gaussian1D(amplitude = gfit_broad['ha_n'].amplitude, \
                                mean = gfit_broad['ha_n'].mean, \
                                stddev = gfit_broad['ha_n'].stddev, \
                                name = 'ha_b')
            
            gfit_broad = g_nii + g_ha_n + g_ha_b
        return (fitter_broad, gfit_broad, rchi2_broad)
    else:
        return (fitter_no_broad, gfit_no_broad, rchi2_no_broad)

####################################################################################################