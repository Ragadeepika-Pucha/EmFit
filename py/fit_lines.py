"""
This script consists of funcitons for fitting emission-lines.
The different functions are divided into different classes for different emission lines.

Author : Ragadeepika Pucha
Version : 2023, April 24
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import fit_utils
import measure_fits as mfit

###################################################################################################

class fit_sii_lines:
    """
    Different functions associated with [SII]6716, 6731 doublet fitting:
        1) fit_one_component(lam_sii, flam_sii, ivar_sii)
        2) fit_two_components(lam_sii, flam_sii, ivar_sii)
    """
    
    def fit_one_component(lam_sii, flam_sii, ivar_sii):
        """
        Function to fit a single component to [SII]6716, 6731 doublet.
        
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
            Best-fit 1 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        ## Initial estimate of amplitudes
        amp_sii = max(flam_sii)

        ## Initial gaussian fits  
        ## Set default sigma values to 130 km/s ~ 2.9 in wavelength space
        ## Set amplitudes > 0, sigma > 40 km/s
        
        g_sii6716 = Gaussian1D(amplitude = amp_sii, mean = 6718.294, \
                               stddev = 2.9, name = 'sii6716', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})
        g_sii6731 = Gaussian1D(amplitude = amp_sii, mean = 6732.673, \
                               stddev = 2.9, name = 'sii6731', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})

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
        fitter_1comp = fitting.LevMarLSQFitter()

        ## Fit
        gfit_1comp = fitter_1comp(g_init, lam_sii, flam_sii, \
                            weights = np.sqrt(ivar_sii), maxiter = 1000)
        rchi2_1comp = mfit.calculate_red_chi2(flam_sii, gfit_1comp(lam_sii),\
                                                   ivar_sii, n_free_params = 4)
        
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################
    
    def fit_two_components(lam_sii, flam_sii, ivar_sii):
        """
        Function to fit two components to [SII]6716, 6731 doublet.
        
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
            Best-fit 2 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        ## Initial estimate of amplitudes
        amp_sii = max(flam_sii)
        
        ## Initial gaussian fits
        ## Default values of sigma ~ 130 km/s ~ 2.9
        ## Set amplitudes > 0, sigma > 40 km/s
        g_sii6716 = Gaussian1D(amplitude = amp_sii/2, mean = 6718.294, \
                               stddev = 2.9, name = 'sii6716', \
                              bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})
        g_sii6731 = Gaussian1D(amplitude = amp_sii/2, mean = 6732.673, \
                               stddev = 2.9, name = 'sii6731', \
                              bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})

        g_sii6716_out = Gaussian1D(amplitude = amp_sii/4, mean = 6718.294, \
                                   stddev = 4.0, name = 'sii6716_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (1.6, None)})
        g_sii6731_out = Gaussian1D(amplitude = amp_sii/4, mean = 6732.673, \
                                   stddev = 4.0, name = 'sii6731_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (1.6, None)})

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
        fitter_2comp = fitting.LevMarLSQFitter()

        gfit_2comp = fitter_2comp(g_init, lam_sii, flam_sii, \
                            weights = np.sqrt(ivar_sii), maxiter = 1000)
        rchi2_2comp = mfit.calculate_red_chi2(flam_sii, gfit_2comp(lam_sii), \
                                                   ivar_sii, n_free_params = 7)
        
        return (gfit_2comp, rchi2_2comp)    
    
####################################################################################################
####################################################################################################

class fit_oiii_lines:
    """
    Different functions associated with [OIII]4959, 5007 doublet fitting:
        1) fit_one_component(lam_oiii, flam_oiii, ivar_oiii)
        2) fit_two_components(lam_oiii, flam_oiii, ivar_oiii)
    """

    def fit_one_component(lam_oiii, flam_oiii, ivar_oiii):
        """
        Function to fit a single component to [OIII]4959,5007 doublet.
        
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
            Best-fit 1 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        # Find initial estimates of amplitudes
        amp_oiii4959 = np.max(flam_oiii[(lam_oiii >= 4959)&(lam_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_oiii[(lam_oiii >= 5007)&(lam_oiii <= 5009)])

        ## Initial gaussian fits
        ## Set default values of sigma ~ 130 km/s ~ 2.1
        ## Set amplitudes > 0

        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959, mean = 4960.295, \
                                stddev = 1.0, name = 'oiii4959', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007, mean = 5008.239, \
                                stddev = 1.0, name = 'oiii5007', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})

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
        fitter_1comp = fitting.LevMarLSQFitter()

        gfit_1comp = fitter_1comp(g_init, lam_oiii, flam_oiii, \
                            weights = np.sqrt(ivar_oiii), maxiter = 1000)
        rchi2_1comp = mfit.calculate_red_chi2(flam_oiii, gfit_1comp(lam_oiii), \
                                                   ivar_oiii, n_free_params = 3) 
        
        
        return (gfit_1comp, rchi2_1comp)
    
####################################################################################################

    def fit_two_components(lam_oiii, flam_oiii, ivar_oiii):
        """
        Function to fit a two components to [OIII]4959,5007 doublet.
        
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
            Best-fit 2 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        # Find initial estimates of amplitudes
        amp_oiii4959 = np.max(flam_oiii[(lam_oiii >= 4959)&(lam_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_oiii[(lam_oiii >= 5007)&(lam_oiii <= 5009)])
        
        ## Initial gaussians
        ## Set default values of sigma ~ 130 km/s ~ 2.1
        ## Set amplitudes > 0
        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959/2, mean = 4960.295, \
                                stddev = 1.0, name = 'oiii4959', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007/2, mean = 5008.239, \
                                stddev = 1.0, name = 'oiii5007', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})

        g_oiii4959_out = Gaussian1D(amplitude = amp_oiii4959/4, mean = 4960.295, \
                                    stddev = 6.0, name = 'oiii4959_out', \
                                    bounds = {'amplitude' : (0.0, None), 'stddev' : (1.2, None)})
        g_oiii5007_out = Gaussian1D(amplitude = amp_oiii5007/4, mean = 5008.239, \
                                    stddev = 6.0, name = 'oiii5007_out', \
                                    bounds = {'amplitude' : (0.0, None), 'stddev' : (1.2, None)})

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
        
        return (gfit_2comp, rchi2_2comp)

####################################################################################################
####################################################################################################

class fit_hb_line:
    """
    Different functions associated with fitting the Hbeta emission-line, 
    including a broad-component:
        1) fit_free_one_component(lam_hb, flam_hb, ivar_hb, sii_bestfit, frac_temp)
        2) fit_fixed_one_component(lam_hb, flam_hb, ivar_hb, sii_bestfit)
        3) fit_fixed_two_components(lam_hb, flam_hb, ivar_hb, sii_bestfit)
    """
    
    def fit_free_one_component(lam_hb, flam_hb, ivar_hb, sii_bestfit, frac_temp = 30):
        """
        Function to fit Hb emission lines - with a single narrow compoent.
        The width is set to be within some percent (default = 30%) of [SII] width.
        This is only when [SII] does not have extra components.
        
        The code fits both with and without broad-component fits and picks the best version.
        The broad-component is allowed if the chi2 improves by 20%
        
        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region where the fits need to be performed.

        flam_hb : numpy array
            Flux array of the spectra in the Hb region.

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        sii_bestfit : astropy model fit
            Best fit for [SII] emission lines.
            Sigma of narrow Hb bounds are set to be within some percent of [SII] width.

        frac_temp : float
            The %age of [SII] width within which narrow Hbeta width can vary

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component

        rchi2: float
            Reduced chi2 of the best-fit
        """

        ## Template fit
        temp_std = sii_bestfit['sii6716'].stddev.value
        temp_std_kms = mfit.lamspace_to_velspace(temp_std, sii_bestfit['sii6716'].mean.value)

        min_std_kms = temp_std_kms - ((frac_temp/100)*temp_std_kms)
        max_std_kms = temp_std_kms + ((frac_temp/100)*temp_std_kms)

        min_std = mfit.velspace_to_lamspace(min_std_kms, 4862.683)
        max_std = mfit.velspace_to_lamspace(max_std_kms, 4862.683)

        ## Initial estimate of amplitude
        amp_hb = np.max(flam_hb)

        ## No outflow components
        ## Single component fit
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                          stddev = temp_std, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None)})

        g_hb_n.stddev.bounds = (min_std, max_std)

        g_hb = g_hb_n
        
        #####################################################################################
        ########################### Fit without broad component #############################

        ## Initial fit
        g_init = g_hb 
        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_hb, flam_hb, \
                                        weights = np.sqrt(ivar_hb), maxiter = 1000)

        
        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                     ivar_hb, n_free_params = 3)

        #####################################################################################
        ########################### Fit with broad component ################################

        ## Two component fit
        g_hb_b = Gaussian1D(amplitude = amp_hb/3, mean = 4862.683, \
                            stddev = 3.0, name = 'hb_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial fit
        g_init = g_hb + g_hb_b 
        fitter_broad = fitting.LevMarLSQFitter()

        gfit_broad = fitter_broad(g_init, lam_hb, flam_hb, \
                                  weights = np.sqrt(ivar_hb), maxiter = 1000)

        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                              ivar_hb, n_free_params = 6)
    
        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Further conditions -- sigma_broad > sigma_narrow
        sig_hb_n = gfit_broad['hb_n'].stddev.value
        sig_hb_b = gfit_broad['hb_b'].stddev.value
        
        if ((del_rchi2 >= 20)&(sig_hb_b > sig_hb_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
####################################################################################################

    def fit_fixed_one_component(lam_hb, flam_hb, ivar_hb, sii_bestfit):
        """
        Function to fit Hbeta line -- fixing the width to the [SII] best-fit.
        Only for a single component - no outflow compoenent.
        The broad-component needs to be >20% better to be picked.
        
        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region where the fits need to be performed.

        flam_hb : numpy array
            Flux array of the spectra in the Hb region.

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        sii_bestfit : astropy model fit
            Best fit for [SII] emission lines
            Sigma of narrow Hbeta is fixed to [SII].

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        ## Initial estimate of amplitude
        amp_hb = np.max(flam_hb)
        
        ## No outflow components
        ## Initial estimate of standard deviation
        std_hb = (4862.683/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value

        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                          stddev = std_hb, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None)})

        ## Fix sigma of Hb narrow to [SII]
        def tie_std_hb(model):
            if (model.n_submodels == 1):
                return ((model.mean/sii_bestfit['sii6716'].mean)*\
                        sii_bestfit['sii6716'].stddev)
            else:
                return ((model['hb_n'].mean/sii_bestfit['sii6716'].mean)*\
                        sii_bestfit['sii6716'].stddev)

        g_hb_n.stddev.tied = tie_std_hb
        g_hb_n.stddev.fixed = True

        g_hb = g_hb_n
        
        #####################################################################################
        ########################### Fit without broad component #############################

        ## Initial fit
        g_init = g_hb 
        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_hb, flam_hb, \
                                        weights = np.sqrt(ivar_hb), maxiter = 1000)


        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                 ivar_hb, n_free_params = 2)
        
        #####################################################################################
        ########################### Fit with broad component ################################

        ## Two component fit
        g_hb_b = Gaussian1D(amplitude = amp_hb/4, mean = 4862.683, \
                            stddev = 4.0, name = 'hb_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial fit
        g_init = g_hb + g_hb_b 
        fitter_broad = fitting.LevMarLSQFitter()

        gfit_broad = fitter_broad(g_init, lam_hb, flam_hb, \
                                  weights = np.sqrt(ivar_hb), maxiter = 1000)

        
        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                              ivar_hb, n_free_params = 5)
        
        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Further conditions -- sigma_broad > sigma_narrow
        sig_hb_n = gfit_broad['hb_n'].stddev.value
        sig_hb_b = gfit_broad['hb_b'].stddev.value

        if ((del_rchi2 >= 20)&(sig_hb_b > sig_hb_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
####################################################################################################

    def fit_fixed_two_components(lam_hb, flam_hb, ivar_hb, sii_bestfit):
        """
        Function to fit Hbeta line -- fixing the width to the [SII] best-fit.
        Includes extra component for both Hbeta and [SII].
        The broad-component needs to be >20% better to be picked.
        
        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region where the fits need to be performed.

        flam_hb : numpy array
            Flux array of the spectra in the Hb region.

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        sii_bestfit : astropy model fit
            Best fit for [SII] emission lines, including outflow component.
            Sigma of narrow (outflow) Hb bounds are set to be within some percent of [SII] width.

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component

        rchi2: float
            Reduced chi2 of the best-fit
        """

        ## Initial estimate of amplitude
        amp_hb = np.max(flam_hb)
        
        ## Initial estimate of standard deviation
        std_hb = (4862.683/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value

        std_hb_out = (4862.683/sii_bestfit['sii6716_out'].mean.value)*\
        sii_bestfit['sii6716_out'].stddev.value

        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                          stddev = std_hb, name = 'hb_n', \
                          bounds = {'amplitude' : (0.0, None)})

        g_hb_out = Gaussian1D(amplitude = amp_hb/3, mean = 4862.683, \
                             stddev = std_hb_out, name = 'hb_out', \
                             bounds = {'amplitude' : (0.0, None)})

        ## Fix sigma of narrow Hb to narrow [SII]
        def tie_std_hb(model):
            return ((model['hb_n'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_hb_n.stddev.tied = tie_std_hb
        g_hb_n.stddev.fixed = True

        ## Fix sigma of outflow Hb to outflow [SII]
        def tie_std_hb_out(model):
            return ((model['hb_out'].mean/sii_bestfit['sii6716_out'].mean)*\
                    sii_bestfit['sii6716_out'].stddev)

        g_hb_out.stddev.tied = tie_std_hb_out
        g_hb_out.stddev.fixed = True

        g_hb = g_hb_n + g_hb_out
        
        #####################################################################################
        ########################### Fit without broad component #############################

        ## Initial fit
        g_init = g_hb 
        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_hb, flam_hb, \
                                        weights = np.sqrt(ivar_hb), maxiter = 1000)

        rchi2_no_broad = mfit.calculate_red_chi2(flam_hb, gfit_no_broad(lam_hb), \
                                                     ivar_hb, n_free_params = 4)

        #####################################################################################
        ########################### Fit with broad component ################################

        ## Two component fit
        g_hb_b = Gaussian1D(amplitude = amp_hb/4, mean = 4862.683, \
                            stddev = 4.0, name = 'hb_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial fit
        g_init = g_hb + g_hb_b 
        fitter_broad = fitting.LevMarLSQFitter()

        gfit_broad = fitter_broad(g_init, lam_hb, flam_hb, \
                                  weights = np.sqrt(ivar_hb), maxiter = 1000)


        rchi2_broad = mfit.calculate_red_chi2(flam_hb, gfit_broad(lam_hb), \
                                              ivar_hb, n_free_params = 7)

        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Further conditions -- sigma_broad > sigma_narrow
        sig_hb_n = gfit_broad['hb_n'].stddev.value
        sig_hb_b = gfit_broad['hb_b'].stddev.value

        if ((del_rchi2 >= 20)&(sig_hb_b > sig_hb_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
####################################################################################################
####################################################################################################

class fit_nii_ha_lines:
    """
    Different functions associated with fitting [NII]+Ha emission-lines:
        1) fit_fixed_nii_free_ha(lam_nii, flam_nii, ivar_nii, sii_bestfit, frac_temp)
        2) fit_fixed_one_component(lam_nii, flam_nii, ivar_nii, sii_bestfit)
        3) fit_fixed_two_components(lam_nii, flam_nii, ivar_nii, sii_bestfit)
    """
    
    def fit_fixed_nii_free_ha(lam_nii, flam_nii, ivar_nii, sii_bestfit, frac_temp = 30.):
        """
        Function to fit [NII]6548, 6583 emission lines.
        Sigma of [NII] is kept fixed to [SII] and
        Ha is allowed to vary within some percent (default = 30%) of [SII].
        The broad component fit needs to be >20% better to be picked.
        Code works only for twp-component [SII] fits, including outflow components.
        
        Parameters
        ----------
        lam_nii : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        frac_temp : float
            The %age of [SII] width within which narrow Halpha width can vary
            
         Returns
        -------
        gfit : Astropy model
            Best-fit 1 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
                
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])
        
        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])
        
        ## Initial estimates of standard deviation
        stddev_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        stddev_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        
        ## Two component fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548/2, mean = 6549.852, \
                               stddev = stddev_nii6548, name = 'nii6548', \
                               bounds = {'amplitude' : (0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
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
            return ((model['nii6548'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6583 
        
        ## Template fit
        temp_std = sii_bestfit['sii6716'].stddev.value
        temp_std_kms = mfit.lamspace_to_velspace(temp_std, \
                                                 sii_bestfit['sii6716'].mean.value)
        min_std_kms = temp_std_kms - ((frac_temp/100)*temp_std_kms)
        max_std_kms = temp_std_kms + ((frac_temp/100)*temp_std_kms)

        min_std_ha = mfit.velspace_to_lamspace(min_std_kms, 6564.312)
        max_std_ha = mfit.velspace_to_lamspace(max_std_kms, 6564.312)
        
        g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                            stddev = temp_std, name = 'ha_n', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Set narrow Ha within 20% of the template fit
        g_ha_n.stddev.bounds = (min_std_ha, max_std_ha)
        
        # Total Halpha components
        g_ha = g_ha_n
        
        #####################################################################################
        ########################## Fit without broad component ##############################

        ## Initial gaussian fit
        g_init = g_nii + g_ha

        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii,\
                                     weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                          ivar_nii, n_free_params = 5)
        

        #####################################################################################
        ########################## Fit with broad component #################################

        ## Broad Ha parameters
        g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                            stddev = 4.0, name = 'ha_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial gaussian fit
        g_init = g_nii + g_ha + g_ha_b
        fitter_broad = fitting.LevMarLSQFitter()

        gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                                  weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                       ivar_nii, n_free_params = 8)

        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Also sigma (broad Ha) > sigma (narrow Ha)
        std_ha_b = gfit_broad['ha_b'].stddev.value
        std_ha_n = gfit_broad['ha_n'].stddev.value

        if ((del_rchi2 >= 20)&(std_ha_b > std_ha_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
#################################################################################################### 
    
    def fit_fixed_one_component(lam_nii, flam_nii, ivar_nii, sii_bestfit):
        """
        Function to fit [NII]6548, 6583 emission lines.
        The code uses [SII] best fit as a template for both [NII] and Ha.
        The broad component fit needs to be >20% better to be picked.
        Code works only for single-component [SII] fits. No outflow components.
        
        Parameters
        ----------
        lam_nii : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
         Returns
        -------
        gfit : Astropy model
            Best-fit 1 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])
        
        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])
        
        ## Initial estimates of standard deviation
        stddev_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        stddev_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value

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
            return ((model['nii6548'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6583

        ## Ha parameters
        ## Model narrow Ha as narrow [SII]
        stddev_ha = (6564.312/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value

        g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                            stddev = stddev_ha, name = 'ha_n', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Tie standard deviation of Ha
        def tie_std_ha(model):
            return ((model['ha_n'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_ha_n.stddev.tied = tie_std_ha
        g_ha_n.stddev.fixed = True

        g_ha = g_ha_n
        
        #####################################################################################
        ########################## Fit without broad component ##############################

        ## Initial gaussian fit
        g_init = g_nii + g_ha

        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii,\
                                     weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                          ivar_nii, n_free_params = 4)
        

        #####################################################################################
        ########################## Fit with broad component #################################

        ## Broad Ha parameters
        g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                            stddev = 4.0, name = 'ha_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial gaussian fit
        g_init = g_nii + g_ha + g_ha_b
        fitter_broad = fitting.LevMarLSQFitter(calc_uncertainties = True)

        gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                                  weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                       ivar_nii, n_free_params = 7)

        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Also sigma (broad Ha) > sigma (narrow Ha)
        std_ha_b = gfit_broad['ha_b'].stddev.value
        std_ha_n = gfit_broad['ha_n'].stddev.value

        if ((del_rchi2 >= 20)&(std_ha_b > std_ha_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
####################################################################################################

    def fit_fixed_two_components(lam_nii, flam_nii, ivar_nii, sii_bestfit):
        """
        Function to fit [NII]6548, 6583 emission lines.
        The code uses [SII] best fit as a template for both [NII] and Ha.
        The broad component fit needs to be >20% better to be picked.
        Code works only for two-component [SII] fits, including outflow components.
        
        Parameters
        ----------
        lam_nii : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
         Returns
        -------
        gfit : Astropy model
            Best-fit 1 component model

        rchi2: float
            Reduced chi2 of the best-fit
        """
        
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii[(lam_nii > 6548)&(lam_nii < 6550)])
        amp_nii6583 = np.max(flam_nii[(lam_nii > 6583)&(lam_nii < 6586)])
        
        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii[(lam_nii > 6563)&(lam_nii < 6565)])
        
        ## Initial estimates of standard deviation
        stddev_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        stddev_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        
        stddev_nii6548_out = (6549.852/sii_bestfit['sii6716_out'].mean.value)*\
        sii_bestfit['sii6716_out'].stddev.value
        stddev_nii6583_out = (6585.277/sii_bestfit['sii6716_out'].mean.value)*\
        sii_bestfit['sii6716_out'].stddev.value

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
            return ((model['nii6548'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            return ((model['nii6583'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

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
            return ((model['nii6548_out'].mean/sii_bestfit['sii6716_out'].mean)*\
                    sii_bestfit['sii6716_out'].stddev)

        g_nii6548_out.stddev.tied = tie_std_nii6548_out
        g_nii6548_out.stddev.fixed = True

        def tie_std_nii6583_out(model):
            return ((model['nii6583_out'].mean/sii_bestfit['sii6716_out'].mean)*\
                    sii_bestfit['sii6716_out'].stddev)

        g_nii6583_out.stddev.tied = tie_std_nii6583_out
        g_nii6583_out.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6583 + g_nii6548_out + g_nii6583_out

        ### Halpha Parameters ##

        ## Ha parameters
        ## Model narrow Ha as narrow [SII]
        stddev_ha = (6564.312/sii_bestfit['sii6716'].mean.value)*\
        sii_bestfit['sii6716'].stddev.value
        ## Model outflow Ha as outflow [SII]
        stddev_ha_out = (6564.312/sii_bestfit['sii6716_out'].mean.value)*\
        sii_bestfit['sii6716_out'].stddev.value

        g_ha_n = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                            stddev = stddev_ha, name = 'ha_n', \
                            bounds = {'amplitude' : (0.0, None)})

        g_ha_out = Gaussian1D(amplitude = amp_ha/3, mean = 6564.312, \
                              stddev = stddev_ha_out, name = 'ha_out', \
                              bounds = {'amplitude' : (0.0, None)})

        ## Tie standard deviation of Ha
        def tie_std_ha(model):
            return ((model['ha_n'].mean/sii_bestfit['sii6716'].mean)*\
                    sii_bestfit['sii6716'].stddev)

        g_ha_n.stddev.tied = tie_std_ha
        g_ha_n.stddev.fixed = True

        ## Tie standard deviation of Ha outflow
        def tie_std_ha_out(model):
            return ((model['ha_out'].mean/sii_bestfit['sii6716_out'].mean)*\
                    sii_bestfit['sii6716_out'].stddev)

        g_ha_out.stddev.tied = tie_std_ha_out
        g_ha_out.stddev.fixed = True

        g_ha = g_ha_n + g_ha_out
        
        #####################################################################################
        ########################## Fit without broad component ##############################

        ## Initial gaussian fit
        g_init = g_nii + g_ha

        fitter_no_broad = fitting.LevMarLSQFitter()

        gfit_no_broad = fitter_no_broad(g_init, lam_nii, flam_nii,\
                                     weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_no_broad = mfit.calculate_red_chi2(flam_nii, gfit_no_broad(lam_nii),\
                                                          ivar_nii, n_free_params = 8)

        #####################################################################################
        ########################## Fit with broad component #################################

        ## Broad Ha parameters
        g_ha_b = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                            stddev = 4.0, name = 'ha_b', \
                            bounds = {'amplitude' : (0.0, None)})

        ## Initial gaussian fit
        g_init = g_nii + g_ha + g_ha_b
        fitter_broad = fitting.LevMarLSQFitter()

        gfit_broad = fitter_broad(g_init, lam_nii, flam_nii,\
                                  weights = np.sqrt(ivar_nii), maxiter = 1000)

        
        rchi2_broad = mfit.calculate_red_chi2(flam_nii, gfit_broad(lam_nii), \
                                                       ivar_nii, n_free_params = 11)

        #####################################################################################
        #####################################################################################

        ## Select the best-fit based on rchi2
        ## If the rchi2 of 2-component is better by 20%, then the 2-component fit is picked.
        ## Otherwise, 1-component fit is the best fit.
        del_rchi2 = ((rchi2_no_broad - rchi2_broad)/rchi2_no_broad)*100
        
        ## Also sigma (broad Ha) > sigma (narrow Ha)
        std_ha_b = gfit_broad['ha_b'].stddev.value
        std_ha_n = gfit_broad['ha_n'].stddev.value

        if ((del_rchi2 >= 20)&(std_ha_b > std_ha_n)):
            return (gfit_broad, rchi2_broad)
        else:
            return (gfit_no_broad, rchi2_no_broad)
        
####################################################################################################
####################################################################################################