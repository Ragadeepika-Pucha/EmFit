"""
This script consists of funcitons for fitting emission-lines.
The different functions are divided into different classes for different emission lines.

Author : Ragadeepika Pucha
Version : 2024, April 8
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D, Const1D

import measure_fits as mfit

from scipy.stats import chi2

###################################################################################################

class fit_sii_lines:
    """
    Different functions associated with [SII]6716, 6731 doublet fitting:
        1) fit_one_component(lam_sii, flam_sii, ivar_sii, rsig_sii)
        2) fit_two_components(lam_sii, flam_sii, ivar_sii, rsig_sii)
    """
    
    def fit_one_component(lam_sii, flam_sii, ivar_sii, rsig_sii):
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
            
        rsig_sii : float
            Median Resolution element in the [SII] region.

        Returns
        -------
        gfit : Astropy model
            Best-fit 1 component model
        """
        
        ## Initial estimate of amplitudes
        amp_sii = max(flam_sii)

        ## Initial gaussian fits  
        ## Set default sigma values to 130 km/s ~ 2.9 in wavelength space
        ## Set amplitudes > 0, sigma > 35 km/s
        g_sii6716 = Gaussian1D(amplitude = amp_sii, mean = 6718.294, \
                               stddev = 2.9, name = 'sii6716', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_sii6731 = Gaussian1D(amplitude = amp_sii, mean = 6732.673, \
                               stddev = 2.9, name = 'sii6731', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        
        ## Tie means of the two gaussians
        def tie_mean_sii(model):
            return ((6732.673/6718.294)*model['sii6716'].mean)

        g_sii6731.mean.tied = tie_mean_sii

        ## Tie standard deviations of the two gaussians
        ## Intrinsic sigma of the two components should be equal
        def tie_std_sii(model):
            term1 = (model['sii6731'].mean/model['sii6716'].mean)**2
            term2 = ((model['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_sii**2)
            
            return (np.sqrt(term3))

        g_sii6731.stddev.tied = tie_std_sii
        
        ## Continuum as a constant
        cont = Const1D(amplitude = 0.0, name = 'sii_cont')

        ## Initial Gaussian fit
        g_init = cont + g_sii6716 + g_sii6731
        fitter_1comp = fitting.LevMarLSQFitter()

        ## Fit
        gfit_1comp = fitter_1comp(g_init, lam_sii, flam_sii, \
                            weights = np.sqrt(ivar_sii), maxiter = 1000)       
                
        return (gfit_1comp)
    
####################################################################################################
    
    def fit_two_components(lam_sii, flam_sii, ivar_sii, rsig_sii):
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
            
        rsig_sii : float
            Median Resolution element in the [SII] region.

        Returns
        -------
        gfit : Astropy model
            Best-fit 2 component model
        """
        
        ## Initial estimate of amplitudes
        amp_sii = max(flam_sii)
        
        ## Initial gaussian fits
        ## Default values of sigma ~ 130 km/s ~ 2.9
        ## Set amplitudes > 0, sigma > 40 km/s
        ## Sigma of outflows >~ 80 km/s
        g_sii6716 = Gaussian1D(amplitude = amp_sii/3, mean = 6718.294, \
                               stddev = 2.9, name = 'sii6716', \
                              bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_sii6731 = Gaussian1D(amplitude = amp_sii/3, mean = 6732.673, \
                               stddev = 2.9, name = 'sii6731', \
                              bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        g_sii6716_out = Gaussian1D(amplitude = amp_sii/5, mean = 6718.294, \
                                   stddev = 4.5, name = 'sii6716_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})
        g_sii6731_out = Gaussian1D(amplitude = amp_sii/5, mean = 6732.673, \
                                   stddev = 4.5, name = 'sii6731_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (0.8, None)})

        ## Tie means of the main gaussian components
        def tie_mean_sii(model):
            return ((6732.673/6718.294)*model['sii6716'].mean)

        g_sii6731.mean.tied = tie_mean_sii

        ## Tie standard deviations of the main gaussian components
        ## The intrinsic sigma values of the two components should be equal
        def tie_std_sii(model):
            term1 = (model['sii6731'].mean/model['sii6716'].mean)**2
            term2 = ((model['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_sii**2)
            
            return (np.sqrt(term3))

        g_sii6731.stddev.tied = tie_std_sii
        
        ## Tie means of the outflow components
        def tie_mean_sii_out(model):
            return ((6732.673/6718.294)*model['sii6716_out'].mean)

        g_sii6731_out.mean.tied = tie_mean_sii_out

        ## Tie standard deviations of the outflow components
        ## The intrinsic sigma values of the two components should be equal
        def tie_std_sii_out(model):
            term1 = (model['sii6731_out'].mean/model['sii6716_out'].mean)**2
            term2 = ((model['sii6716_out'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_sii**2)
            
            return (np.sqrt(term3))

        g_sii6731_out.stddev.tied = tie_std_sii_out

        ## Tie amplitudes of all the four components
        def tie_amp_sii(model):
            return ((model['sii6731'].amplitude/model['sii6716'].amplitude)*\
                    model['sii6716_out'].amplitude)

        g_sii6731_out.amplitude.tied = tie_amp_sii
        
        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'sii_cont')

        ## Initial gaussian
        g_init = cont + g_sii6716 + g_sii6731 + g_sii6716_out + g_sii6731_out
        fitter_2comp = fitting.LevMarLSQFitter()
        
        gfit_2comp = fitter_2comp(g_init, lam_sii, flam_sii, \
                            weights = np.sqrt(ivar_sii), maxiter = 1000)
                
        ## Set the broader component as the outflow component
        sii_out_sig = mfit.lamspace_to_velspace(gfit_2comp['sii6716_out'].stddev.value, \
                                               gfit_2comp['sii6716_out'].mean.value)
        sii_sig = mfit.lamspace_to_velspace(gfit_2comp['sii6716'].stddev.value, \
                                            gfit_2comp['sii6716'].mean.value)
        if (sii_out_sig < sii_sig):
            ## Set the broader component as "outflow" component
            gfit_sii6716 = Gaussian1D(amplitude = gfit_2comp['sii6716_out'].amplitude, \
                                     mean = gfit_2comp['sii6716_out'].mean, \
                                     stddev = gfit_2comp['sii6716_out'].stddev, \
                                     name = 'sii6716')
            gfit_sii6731 = Gaussian1D(amplitude = gfit_2comp['sii6731_out'].amplitude, \
                                     mean = gfit_2comp['sii6731_out'].mean, \
                                     stddev = gfit_2comp['sii6731_out'].stddev, \
                                     name = 'sii6731')
            gfit_sii6716_out = Gaussian1D(amplitude = gfit_2comp['sii6716'].amplitude, \
                                         mean = gfit_2comp['sii6716'].mean, \
                                         stddev = gfit_2comp['sii6716'].stddev, \
                                         name = 'sii6716_out')
            gfit_sii6731_out = Gaussian1D(amplitude = gfit_2comp['sii6731'].amplitude, \
                                         mean = gfit_2comp['sii6731_out'].mean, \
                                         stddev = gfit_2comp['sii6731_out'].stddev, \
                                         name = 'sii6731_out')
            cont = gfit_2comp['sii_cont']
            
            gfit_2comp = cont + gfit_sii6716 + gfit_sii6731 + gfit_sii6716_out + gfit_sii6731_out
        
        return (gfit_2comp)    
    
####################################################################################################
####################################################################################################

class fit_oiii_lines:
    """
    Different functions associated with [OIII]4959, 5007 doublet fitting:
        1) fit_one_component(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii)
        2) fit_two_components(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii)
    """

    def fit_one_component(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii):
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
            
        rsig_oiii : float
            Median Resolution element in the [SII] region.

        Returns
        -------
        gfit : Astropy model
            Best-fit 1 component model
        """
        
        # Find initial estimates of amplitudes
        amp_oiii4959 = np.max(flam_oiii[(lam_oiii >= 4959)&(lam_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_oiii[(lam_oiii >= 5007)&(lam_oiii <= 5009)])

        ## Initial gaussian fits
        ## Set default values of sigma ~ 130 km/s ~ 2.1
        ## Set amplitudes > 0
        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959, mean = 4960.295, \
                                stddev = 2.1, name = 'oiii4959', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007, mean = 5008.239, \
                                stddev = 2.1, name = 'oiii5007', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        ## Tie Means of the two gaussians
        def tie_mean_oiii(model):
            return ((5008.239/4960.295)*model['oiii4959'].mean)

        g_oiii5007.mean.tied = tie_mean_oiii

        ## Tie Amplitudes of the two gaussians
        def tie_amp_oiii(model):
            return (model['oiii4959'].amplitude*2.98)

        g_oiii5007.amplitude.tied = tie_amp_oiii

        ## Tie standard deviations in velocity space
        ## Intrinsic sigma of the two components should be equal
        def tie_std_oiii(model):
            term1 = (model['oiii5007'].mean/model['oiii4959'].mean)**2
            term2 = ((model['oiii4959'].stddev)**2) - (rsig_oiii**2)
            term3 = (term1*term2)+(rsig_oiii**2)
            
            return (np.sqrt(term3))

        g_oiii5007.stddev.tied = tie_std_oiii
        
    
        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'oiii_cont')

        ## Initial Gaussian fit
        g_init = cont + g_oiii4959 + g_oiii5007

        ## Fitter
        fitter_1comp = fitting.LevMarLSQFitter()

        gfit_1comp = fitter_1comp(g_init, lam_oiii, flam_oiii, \
                            weights = np.sqrt(ivar_oiii), maxiter = 1000)
            
        return (gfit_1comp)
    
####################################################################################################

    def fit_two_components(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii):
        """
        Function to fit two components to [OIII]4959,5007 doublet.
        
        Parameters
        ----------
        lam_oiii : numpy array
            Wavelength array of the [OIII] region where the fits need to be performed.

        flam_oiii : numpy array
            Flux array of the spectra in the [OIII] region.

        ivar_oiii : numpy array
            Inverse variance array of the spectra in the [OIII] region.
            
        rsig_oiii : float
            Median Resolution element in the [OIII] region.

        Returns
        -------
        gfit : Astropy model
            Best-fit 2 component model
        """
        
        # Find initial estimates of amplitudes
        amp_oiii4959 = np.max(flam_oiii[(lam_oiii >= 4959)&(lam_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_oiii[(lam_oiii >= 5007)&(lam_oiii <= 5009)])
        
        ## Initial gaussians
        ## Set default values of sigma ~ 130 km/s ~ 2.1
        ## Set amplitudes > 0
        
        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959/2, mean = 4960.295, \
                                stddev = 1.0, name = 'oiii4959', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007/2, mean = 5008.239, \
                                stddev = 1.0, name = 'oiii5007', \
                                bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        g_oiii4959_out = Gaussian1D(amplitude = amp_oiii4959/4, mean = 4960.295, \
                                    stddev = 4.0, name = 'oiii4959_out', \
                                    bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})
        g_oiii5007_out = Gaussian1D(amplitude = amp_oiii5007/4, mean = 5008.239, \
                                    stddev = 4.0, name = 'oiii5007_out', \
                                    bounds = {'amplitude' : (0.0, None), 'stddev' : (0.6, None)})

        ## Tie Means of the two gaussians
        def tie_mean_oiii(model):
            return ((5008.239/4960.295)*model['oiii4959'].mean)

        g_oiii5007.mean.tied = tie_mean_oiii

        ## Tie Amplitudes of the two gaussians
        def tie_amp_oiii(model):
            return (model['oiii4959'].amplitude*2.98)

        g_oiii5007.amplitude.tied = tie_amp_oiii

        ## Tie standard deviations in velocity space
        ## Intrinsic sigma of the two components should be equal
        def tie_std_oiii(model):
            term1 = (model['oiii5007'].mean/model['oiii4959'].mean)**2
            term2 = ((model['oiii4959'].stddev)**2) - (rsig_oiii**2)
            term3 = (term1*term2)+(rsig_oiii**2)
            
            return (np.sqrt(term3))

        g_oiii5007.stddev.tied = tie_std_oiii

        ## Tie Means of the two gaussian outflow components
        def tie_mean_oiii_out(model):
            return ((5008.239/4960.295)*model['oiii4959_out'].mean)

        g_oiii5007_out.mean.tied = tie_mean_oiii_out

        ## Tie Amplitudes of the two gaussian outflow components
        def tie_amp_oiii_out(model):
            return (model['oiii4959_out'].amplitude*2.98)

        g_oiii5007_out.amplitude.tied = tie_amp_oiii_out

        ## Tie standard deviations of the outflow components in the velocity space
        ## Intrinsic sigma of the two components should be equal
        def tie_std_oiii_out(model):
            term1 = (model['oiii5007_out'].mean/model['oiii4959_out'].mean)**2
            term2 = ((model['oiii4959_out'].stddev)**2) - (rsig_oiii**2)
            term3 = (term1*term2)+(rsig_oiii**2)
            
            return (np.sqrt(term3))

        g_oiii5007_out.stddev.tied = tie_std_oiii_out
        
        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'oiii_cont')

        ## Initial Gaussian fit
        g_init = cont + g_oiii4959 + g_oiii5007 + g_oiii4959_out + g_oiii5007_out

        ## Fitter
        fitter_2comp = fitting.LevMarLSQFitter()

        gfit_2comp = fitter_2comp(g_init, lam_oiii, flam_oiii, \
                            weights = np.sqrt(ivar_oiii), maxiter = 1000)
        
        ## Set the broad component as the "outflow" component
        oiii_out_sig = mfit.lamspace_to_velspace(gfit_2comp['oiii5007_out'].stddev.value, \
                                                 gfit_2comp['oiii5007_out'].mean.value)
        oiii_sig = mfit.lamspace_to_velspace(gfit_2comp['oiii5007'].stddev.value, \
                                            gfit_2comp['oiii5007'].mean.value)
        
        if (oiii_out_sig < oiii_sig):
            gfit_oiii4959 = Gaussian1D(amplitude = gfit_2comp['oiii4959_out'].amplitude, \
                                      mean = gfit_2comp['oiii4959_out'].mean, \
                                      stddev = gfit_2comp['oiii4959_out'].stddev, \
                                      name = 'oiii4959')
            gfit_oiii5007 = Gaussian1D(amplitude = gfit_2comp['oiii5007_out'].amplitude, \
                                      mean = gfit_2comp['oiii5007_out'].mean, \
                                      stddev = gfit_2comp['oiii5007_out'].stddev, \
                                      name = 'oiii5007')
            gfit_oiii4959_out = Gaussian1D(amplitude = gfit_2comp['oiii4959'].amplitude, \
                                          mean = gfit_2comp['oiii4959'].mean, \
                                          stddev = gfit_2comp['oiii4959'].stddev, \
                                          name = 'oiii4959_out')
            gfit_oiii5007_out = Gaussian1D(amplitude = gfit_2comp['oiii5007'].amplitude, \
                                          mean = gfit_2comp['oiii5007'].mean, \
                                          stddev = gfit_2comp['oiii5007'].stddev, \
                                          name = 'oiii5007_out')
            cont = gfit_2comp['oiii_cont'] 
            gfit_2comp = cont + gfit_oiii4959 + gfit_oiii5007 + \
            gfit_oiii4959_out + gfit_oiii5007_out
            
        return (gfit_2comp)

####################################################################################################
####################################################################################################

class fit_nii_ha_lines:
    """
    Different functions associated with fitting [NII]+Ha emission-lines:
        1) fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                         sii_bestfit, rsig_sii,
                                         priors = [4,5], broad_comp = True)
        2) fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                    sii_bestfit, rsig_sii, 
                                    priors = [4,5], broad_comp = True)
        3) fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                    sii_bestfit, rsig_sii,
                                    priors = [4,5], broad_comp = True)                                 
    """
    
    def fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                      sii_bestfit, rsig_sii, priors = [4, 5], broad_comp = True):
        """
        Function to fit [NII]6548,6583 + Ha emission lines.
        The width of [NII] is kept fixed to [SII] and Ha is allowed to vary 
        upto twice of [SII]. This is when [SII] has only a single component.
        
        The code can fit with and without a broad component, depending on whether the 
        broad_comp keyword is set to True/False
        
        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
        priors : list
            Initial priors for the amplitude and stddev of the broad component
            
        broad_comp : bool
            Whether or not to add a broad component for the fit
            Default is True
            
        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            Depends on what the broad_comp is set to
        """
    
        ############################## [NII]6548,6583 doublet ###########################
        ## Initial estimate of amplitude for [NII]6583, 6583
        amp_nii6548 = np.max(flam_nii_ha[(lam_nii_ha > 6548)&(lam_nii_ha < 6550)])
        amp_nii6583 = np.max(flam_nii_ha[(lam_nii_ha > 6583)&(lam_nii_ha < 6586)])

        ## Initial estimates of standard deviation for [NII]
        sii_std = sii_bestfit['sii6716'].stddev.value

        std_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*sii_std
        std_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*sii_std

        ## [NII] Gaussians
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548, mean = 6549.852, \
                              stddev = std_nii6548, name = 'nii6548', \
                              bounds = {'amplitude' : (0.0, None)})

        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                              stddev = std_nii6583, name = 'nii6583', \
                              bounds = {'amplitude' : (0.0, None)})

        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return ((6585.277/6549.852)*model['nii6548'].mean)

        g_nii6583.mean.tied = tie_mean_nii

        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*2.96)

        g_nii6583.amplitude.tied = tie_amp_nii

        ## Tie standard deviations of all the narrow components
        ## Intrinsic sigma values match with [SII]
        def tie_std_nii6548(model):
            term1 = (model['nii6548'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))
            
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            term1 = (model['nii6583'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))

        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6583

        ######################## HALPHA #################################################

        ## Template fit
        ## [SII] width in AA
        temp_std = sii_bestfit['sii6716'].stddev.value
        ## [SII] width in km/s
        temp_std_kms = mfit.lamspace_to_velspace(temp_std, sii_bestfit['sii6716'].mean.value)

        ## Set up max_std to be 100% of [SII] width
        max_std_kms = 2*temp_std_kms

        ## In AA
        max_std = mfit.velspace_to_lamspace(max_std_kms, 6564.312)

        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii_ha[(lam_nii_ha > 6550)&(lam_nii_ha < 6575)])

        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'nii_ha_cont')

        ## No outflow components
        ## Single component fit

        if (broad_comp == True):
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                               stddev = temp_std, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, max_std)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Broad component
            g_ha_b = Gaussian1D(amplitude = amp_ha/priors[0], mean = 6564.312, \
                               stddev = priors[1], name = 'ha_b', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (1.0, None)})

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n + g_ha_b
            fitter_b = fitting.LevMarLSQFitter()

            gfit_b = fitter_b(g_init, lam_nii_ha, flam_nii_ha, \
                             weights = np.sqrt(ivar_nii_ha), maxiter = 1000)
            
            ## Exchange broad and narrow Ha components 
            ## if narrow Ha component has lower amplitude and broader sigma
            ha_b_amp = gfit_b['ha_b'].amplitude.value
            ha_n_amp = gfit_b['ha_n'].amplitude.value
            ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                                gfit_b['ha_b'].mean.value)
            ha_n_sig = mfit.lamspace_to_velspace(gfit_b['ha_n'].stddev.value, \
                                                gfit_b['ha_n'].mean.value)

            if ((ha_b_amp > ha_n_amp)&(ha_b_sig < ha_n_sig)):
                g_ha_n = Gaussian1D(amplitude = gfit_b['ha_b'].amplitude, \
                                   mean = gfit_b['ha_b'].mean, \
                                   stddev = gfit_b['ha_b'].stddev, \
                                   name = 'ha_n')
                g_ha_b = Gaussian1D(amplitude = gfit_b['ha_n'].amplitude, \
                                   mean = gfit_b['ha_n'].mean, \
                                   stddev = gfit_b['ha_n'].stddev, \
                                   name = 'ha_b')
                gfit_b = gfit_b['nii_ha_cont'] + gfit_b['nii6548'] + gfit_b['nii6583']+\
                g_ha_n + g_ha_b
            ## Returns fit with broad component if broad_comp = True
            return (gfit_b)

        else:
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                               stddev = temp_std, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, max_std)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n
            fitter_no_b = fitting.LevMarLSQFitter()

            gfit_no_b = fitter_no_b(g_init, lam_nii_ha, flam_nii_ha, \
                             weights = np.sqrt(ivar_nii_ha), maxiter = 1000)

            ## Returns fit without broad component if broad_comp = False
            return (gfit_no_b)
        
####################################################################################################

    def fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                sii_bestfit, rsig_sii, priors = [4, 5], broad_comp = True):
        """
        Function to fit [NII]6548,6583 + Ha emission lines.
        The width of narrow [NII] and Ha is kept fixed to narrow [SII] 
        This is when [SII] has one component.

        The code can fit with and without a broad component, depending on whether the 
        broad_comp keyword is set to True/False

        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region

        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
        priors : list
            Initial priors for the amplitude and stddev of the broad component

        broad_comp : bool
            Whether or not to add a broad component for the fit
            Default is True

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            Depends on what the broad_comp is set to
        """

        ############################## [NII]6548,6583 doublet ###########################
        ## Initial estimate of amplitude for [NII]6583, 6583
        amp_nii6548 = np.max(flam_nii_ha[(lam_nii_ha > 6548)&(lam_nii_ha < 6550)])
        amp_nii6583 = np.max(flam_nii_ha[(lam_nii_ha > 6583)&(lam_nii_ha < 6586)])

        ## Initial estimates of standard deviation for [NII]
        sii_std = sii_bestfit['sii6716'].stddev.value

        std_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*sii_std
        std_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*sii_std

        ## [NII] Gaussians
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548, mean = 6549.852, \
                              stddev = std_nii6548, name = 'nii6548', \
                              bounds = {'amplitude' : (0.0, None)})

        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                              stddev = std_nii6583, name = 'nii6583', \
                              bounds = {'amplitude' : (0.0, None)})

        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return ((6585.277/6549.852)*model['nii6548'].mean)

        g_nii6583.mean.tied = tie_mean_nii

        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*2.96)

        g_nii6583.amplitude.tied = tie_amp_nii

        ## Tie standard deviations of all the narrow components
        ## Intrinsic sigma values match with [SII]
        def tie_std_nii6548(model):
            term1 = (model['nii6548'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))
            
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            term1 = (model['nii6583'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))

        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6583

        ######################## HALPHA #################################################

        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii_ha[(lam_nii_ha > 6550)&(lam_nii_ha < 6575)])

        ## Initial estimate of standard deviation
        std_ha = (6564.312/sii_bestfit['sii6716'].mean.value)*sii_std

        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'nii_ha_cont')

        ## Two components
        if (broad_comp == True):
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                               stddev = std_ha, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Fix intrinsic sigma of narrow Ha to [SII]
            def tie_std_ha(model):
                term1 = (model['ha_n'].mean/sii_bestfit['sii6716'].mean)**2
                term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2)+(rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_n.stddev.tied = tie_std_ha
            g_ha_n.stddev.fixed = True

            ## Broad component
            g_ha_b = Gaussian1D(amplitude = amp_ha/priors[0], mean = 6564.312, \
                               stddev = priors[1], name = 'ha_b', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (1.0, None)})

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n + g_ha_b
            fitter_b = fitting.LevMarLSQFitter()
            gfit_b = fitter_b(g_init, lam_nii_ha, flam_nii_ha, \
                             weights = np.sqrt(ivar_nii_ha), maxiter = 1000)
            
            ## Exchange broad and narrow Ha components 
            ## if narrow Ha component has lower amplitude and broader sigma
            ha_b_amp = gfit_b['ha_b'].amplitude.value
            ha_n_amp = gfit_b['ha_n'].amplitude.value
            ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                                gfit_b['ha_b'].mean.value)
            ha_n_sig = mfit.lamspace_to_velspace(gfit_b['ha_n'].stddev.value, \
                                                gfit_b['ha_n'].mean.value)

            if ((ha_b_amp > ha_n_amp)&(ha_b_sig < ha_n_sig)):
                g_ha_n = Gaussian1D(amplitude = gfit_b['ha_b'].amplitude, \
                                   mean = gfit_b['ha_b'].mean, \
                                   stddev = gfit_b['ha_b'].stddev, \
                                   name = 'ha_n')
                g_ha_b = Gaussian1D(amplitude = gfit_b['ha_n'].amplitude, \
                                   mean = gfit_b['ha_n'].mean, \
                                   stddev = gfit_b['ha_n'].stddev, \
                                   name = 'ha_b')
                gfit_b = gfit_b['nii_ha_cont'] + gfit_b['nii6548'] + gfit_b['nii6583']+\
                g_ha_n + g_ha_b

            ## Returns fit with broad component if broad_comp = True
            return (gfit_b)

        else:
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                               stddev = std_ha, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Fix intrinsic sigma of narrow Ha to [SII]
            def tie_std_ha(model):
                term1 = (model['ha_n'].mean/sii_bestfit['sii6716'].mean)**2
                term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2)+(rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_n.stddev.tied = tie_std_ha
            g_ha_n.stddev.fixed = True

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n 
            fitter_no_b = fitting.LevMarLSQFitter()
            gfit_no_b = fitter_no_b(g_init, lam_nii_ha, flam_nii_ha, \
                                    weights = np.sqrt(ivar_nii_ha), maxiter = 1000)


            ## Returns fit without broad component if broad_comp = False
            return (gfit_no_b)
        
####################################################################################################
     
    def fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                  sii_bestfit, rsig_sii, priors = [4, 5], broad_comp = True):
        """
        Function to fit [NII]6548,6583 + Ha emission lines.
        The width of narrow (outflow) [NII] and Ha is kept fixed to narrow (outflow) [SII]. 
        This is when [SII] has two components.

        The code can fit with and without a broad component, depending on whether the 
        broad_comp keyword is set to True/False

        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region.

        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
        priors : list
            Initial priors for the amplitude and stddev of the broad component

        broad_comp : bool
            Whether or not to add a broad component for the fit
            Default is True

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            Depends on what the broad_comp is set to
        """

        ############################## [NII]6548,6583 doublet ###########################
        ## Initial estimate of amplitude for [NII]6583, 6583
        amp_nii6548 = np.max(flam_nii_ha[(lam_nii_ha > 6548)&(lam_nii_ha < 6550)])
        amp_nii6583 = np.max(flam_nii_ha[(lam_nii_ha > 6583)&(lam_nii_ha < 6586)])

        ## Information from [SII] Bestfit
        sii_std = sii_bestfit['sii6716'].stddev.value
        sii_out_std = sii_bestfit['sii6716_out'].stddev.value
        del_lam_sii = (sii_bestfit['sii6716_out'].mean.value - sii_bestfit['sii6716'].mean.value)

        ## Initial estimates of standard deviation for [NII]
        std_nii6548 = (6549.852/sii_bestfit['sii6716'].mean.value)*sii_std
        std_nii6583 = (6585.277/sii_bestfit['sii6716'].mean.value)*sii_std

        std_nii6548_out = (6549.852/sii_bestfit['sii6716_out'].mean.value)*sii_out_std
        std_nii6583_out = (6585.277/sii_bestfit['sii6716_out'].mean.value)*sii_out_std

        ## [NII] Gaussians
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548/2, mean = 6549.852, \
                              stddev = std_nii6548, name = 'nii6548', \
                              bounds = {'amplitude' : (0.0, None)})

        g_nii6583 = Gaussian1D(amplitude = amp_nii6583/2, mean = 6585.277, \
                              stddev = std_nii6583, name = 'nii6583', \
                              bounds = {'amplitude' : (0.0, None)})

        ## Tie means of [NII] doublet gaussians
        def tie_mean_nii(model):
            return ((6585.277/6549.852)*model['nii6548'].mean)

        g_nii6583.mean.tied = tie_mean_nii

        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*2.96)

        g_nii6583.amplitude.tied = tie_amp_nii

        ## Tie standard deviations of all the narrow components
        ## Intrinsic sigma values match with [SII]
        def tie_std_nii6548(model):
            term1 = (model['nii6548'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))
            
        g_nii6548.stddev.tied = tie_std_nii6548
        g_nii6548.stddev.fixed = True

        def tie_std_nii6583(model):
            term1 = (model['nii6583'].mean/sii_bestfit['sii6716'].mean)**2
            term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))

        g_nii6583.stddev.tied = tie_std_nii6583
        g_nii6583.stddev.fixed = True

        ## [NII] outflow Gaussians
        g_nii6548_out = Gaussian1D(amplitude = amp_nii6548/3, mean = 6549.852, \
                                  stddev = std_nii6548_out, name = 'nii6548_out', \
                                  bounds = {'amplitude' : (0.0, None)})

        g_nii6583_out = Gaussian1D(amplitude = amp_nii6583/3, mean = 6585.277, \
                                  stddev = std_nii6583_out, name = 'nii6583_out', \
                                  bounds = {'amplitude' : (0.0, None)})
        
        ## Tie relative positions of narrow and outflow components
        def tie_relmean_nii6548_out(model):
            return (((6549.852/6718.294)*del_lam_sii) + model['nii6548'].mean)
        
        g_nii6548_out.mean.tied = tie_relmean_nii6548_out
        
        def tie_relmean_nii6583_out(model):
            return (((6585.277/6718.294)*del_lam_sii) + model['nii6583'].mean)
        
        g_nii6583_out.mean.tied = tie_relmean_nii6583_out

        ## Tie means of [NII] doublet outflow gaussians
        def tie_mean_nii_out(model):
            return ((6585.277/6549.852)*model['nii6548_out'].mean)

        g_nii6583_out.mean.tied = tie_mean_nii_out
        
        ## Tie amplitudes of two [NII] gaussians
        def tie_amp_nii_out(model):
            return (model['nii6548_out'].amplitude*2.96)

        g_nii6583_out.amplitude.tied = tie_amp_nii_out
        
        ## Tie standard deviations of the outflow components
        ## Intrinsic sigma values match with [SII]out
        def tie_std_nii6548_out(model):
            term1 = (model['nii6548_out'].mean/sii_bestfit['sii6716_out'].mean)**2
            term2 = ((sii_bestfit['sii6716_out'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))

        g_nii6548_out.stddev.tied = tie_std_nii6548_out
        g_nii6548_out.stddev.fixed = True

        def tie_std_nii6583_out(model):
            term1 = (model['nii6583_out'].mean/sii_bestfit['sii6716_out'].mean)**2
            term2 = ((sii_bestfit['sii6716_out'].stddev)**2) - (rsig_sii**2)
            term3 = (term1*term2)+(rsig_nii_ha**2)
            
            return (np.sqrt(term3))

        g_nii6583_out.stddev.tied = tie_std_nii6583_out
        g_nii6583_out.stddev.fixed = True

        g_nii = g_nii6548 + g_nii6548_out + g_nii6583 + g_nii6583_out

        ######################## HALPHA #################################################

        ## Initial guess of amplitude for Ha
        amp_ha = np.max(flam_nii_ha[(lam_nii_ha > 6550)&(lam_nii_ha < 6575)])

        ## Initial estimate of standard deviation
        std_ha = (6564.312/sii_bestfit['sii6716'].mean.value)*sii_std
        std_ha_out = (6564.312/sii_bestfit['sii6716_out'].mean.value)*sii_out_std

        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'nii_ha_cont')
        ## Two compoenent model for Ha

        if (broad_comp == True):
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                               stddev = std_ha, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Fix intrinsic sigma of narrow Ha to narrow [SII]
            def tie_std_ha(model):
                term1 = (model['ha_n'].mean/sii_bestfit['sii6716'].mean)**2
                term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2) + (rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_n.stddev.tied = tie_std_ha
            g_ha_n.stddev.fixed = True

            ## Outflow component
            g_ha_out = Gaussian1D(amplitude = amp_ha/3, mean = 6564.312, \
                                 stddev = std_ha_out, name = 'ha_out', \
                                 bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie relative positions of narrow and outflow components
            def tie_relmean_ha_out(model):
                return (((6564.312/6718.294)*del_lam_sii) + model['ha_n'].mean)

            g_ha_out.mean.tied = tie_relmean_ha_out
            
            ## Tie mean of outflow Ha to outflow [NII]
            def tie_mean_ha_out(model):
                return ((6564.312/6549.852)*model['nii6548_out'].mean)
            
            g_ha_out.mean.tied = tie_mean_ha_out

            ## Fix intrinsic sigma of outflow Ha to outflow [SII]
            def tie_std_ha_out(model):
                term1 = (model['ha_out'].mean/sii_bestfit['sii6716_out'].mean)**2
                term2 = ((sii_bestfit['sii6716_out'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2) + (rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_out.stddev.tied = tie_std_ha_out
            g_ha_out.stddev.fixed = True

            ## Broad component
            g_ha_b = Gaussian1D(amplitude = amp_ha/priors[0], mean = 6564.312, \
                               stddev = priors[1], name = 'ha_b', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (1.0, None)})

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n + g_ha_out + g_ha_b
            fitter_b = fitting.LevMarLSQFitter()

            gfit_b = fitter_b(g_init, lam_nii_ha, flam_nii_ha, \
                             weights = np.sqrt(ivar_nii_ha), maxiter = 1000)
            
            ## Exchange broad and outflow Ha components
            ## If outflow Ha component has lower amplitude and broad sigma
            ha_b_amp = gfit_b['ha_b'].amplitude.value
            ha_out_amp = gfit_b['ha_out'].amplitude.value
            ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                                gfit_b['ha_b'].mean.value)
            ha_out_sig = mfit.lamspace_to_velspace(gfit_b['ha_out'].stddev.value, \
                                                  gfit_b['ha_out'].mean.value)
            
            if ((ha_b_amp > ha_out_amp)&(ha_b_sig < ha_out_sig)):
                g_ha_out = Gaussian1D(amplitude = gfit_b['ha_b'].amplitude, \
                                     mean = gfit_b['ha_b'].mean, \
                                     stddev = gfit_b['ha_b'].stddev, \
                                     name = 'ha_out')
                g_ha_b = Gaussian1D(amplitude = gfit_b['ha_out'].amplitude, \
                                   mean = gfit_b['ha_out'].mean, \
                                   stddev = gfit_b['ha_out'].stddev, \
                                   name = 'ha_b')
                gfit_b = gfit_b['nii_ha_cont'] + gfit_b['nii6548'] + gfit_b['nii6548_out'] +\
                gfit_b['nii6583'] + gfit_b['nii6583_out'] + gfit_b['ha_n'] + g_ha_out + g_ha_b
            
            ## Returns fit with broad component if broad_comp = True
            return (gfit_b)

        else:
            ## Narrow component
            g_ha_n = Gaussian1D(amplitude = amp_ha/2, mean = 6564.312, \
                               stddev = std_ha, name = 'ha_n', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie mean of Ha to [NII]
            def tie_mean_ha(model):
                return ((6564.312/6549.852)*model['nii6548'].mean)
            
            g_ha_n.mean.tied = tie_mean_ha

            ## Fix intrinsic sigma of narrow Ha to narrow [SII]
            def tie_std_ha(model):
                term1 = (model['ha_n'].mean/sii_bestfit['sii6716'].mean)**2
                term2 = ((sii_bestfit['sii6716'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2) + (rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_n.stddev.tied = tie_std_ha
            g_ha_n.stddev.fixed = True

            ## Outflow component
            g_ha_out = Gaussian1D(amplitude = amp_ha/3, mean = 6564.312, \
                                 stddev = std_ha_out, name = 'ha_out', \
                                 bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
            
            ## Tie relative positions of narrow and outflow components
            def tie_relmean_ha_out(model):
                return (((6564.312/6718.294)*del_lam_sii) + model['ha_n'].mean)

            g_ha_out.mean.tied = tie_relmean_ha_out
            
            ## Tie mean of outflow Ha to outflow [NII]
            def tie_mean_ha_out(model):
                return ((6564.312/6549.852)*model['nii6548_out'].mean)
            
            g_ha_out.mean.tied = tie_mean_ha_out

            ## Fix intrinsic sigma of outflow Ha to outflow [SII]
            def tie_std_ha_out(model):
                term1 = (model['ha_out'].mean/sii_bestfit['sii6716_out'].mean)**2
                term2 = ((sii_bestfit['sii6716_out'].stddev)**2) - (rsig_sii**2)
                term3 = (term1*term2) + (rsig_nii_ha**2)
                
                return (np.sqrt(term3))

            g_ha_out.stddev.tied = tie_std_ha_out
            g_ha_out.stddev.fixed = True

            ## Initial Fit
            g_init = cont + g_nii + g_ha_n + g_ha_out
            fitter_no_b = fitting.LevMarLSQFitter()

            gfit_no_b = fitter_no_b(g_init, lam_nii_ha, flam_nii_ha, \
                             weights = np.sqrt(ivar_nii_ha), maxiter = 1000)

            ## Returns fit without broad component if broad_comp = False
            return (gfit_no_b)

####################################################################################################
####################################################################################################

class fit_hb_line:
    """
    Different functions associated with fitting Hb emission line:
        1) fit_hb_one_component(lam_hb, flam_hb, ivar_hb, nii_ha_bestfit)
        2) fit_hb_two_components(lam_hb, flam_hb, ivar_hb, nii_ha_bestfit)
    """
    
    def fit_hb_one_component(lam_hb, flam_hb, ivar_hb, nii_ha_bestfit):
        """
        Function to fit Hb emission-line, when [SII] has one component.
        The width of narrow Hb line is fixed to narrow Ha component.
        The width of broad Hb is fixed to broad Hb, if available.
        
        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region where the fits need to be performed.

        flam_hb : numpy array
            Flux array of the spectra in the Hb region.

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        nii_ha_bestfit : astropy model fit
            Best fit for [NII]+Ha emission lines.
            Sigma of narrow Hb is fixed to narrow Ha
            Sigma of broad Hb is fixed to broad Ha (if available)

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component.
            
        """
        
        ## Initial estimate of amplitude of Hb
        amp_hb = np.max(flam_hb[(lam_hb >= 4861)&(lam_hb <=4863)])

        ## Std_Ha
        std_ha = nii_ha_bestfit['ha_n'].stddev.value

        ## Starting value for Hb
        std_hb = (4862.683/nii_ha_bestfit['ha_n'].mean.value)*std_ha

        ## Narrow Hb Gaussian
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                           stddev = std_hb, name = 'hb_n', \
                           bounds = {'amplitude' : (0.0, None)})
        
        ## Tie mean of Hb to Ha
        def tie_mean_hb(model):
            return ((4862.683/6564.312)*nii_ha_bestfit['ha_n'].mean)
        
        g_hb_n.mean.tied = tie_mean_hb
        g_hb_n.mean.fixed = True

        ## Tie standard deviation of Hb to Ha in velocity space
        def tie_std_hb(model):
            return ((model['hb_n'].mean/nii_ha_bestfit['ha_n'].mean)*\
                   nii_ha_bestfit['ha_n'].stddev)

        g_hb_n.stddev.tied = tie_std_hb
        g_hb_n.stddev.fixed = True

        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'hb_cont')

        g_hb = cont + g_hb_n

        if ('ha_b' in nii_ha_bestfit.submodel_names):
            ## Std_Ha_b
            std_ha_b = nii_ha_bestfit['ha_b'].stddev.value

            ## Starting value for Hb broad
            std_hb_b = (4862.683/nii_ha_bestfit['ha_b'].mean.value)*std_ha_b

            ## Broad Hb Gaussian
            g_hb_b = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                               stddev = std_hb_b, name = 'hb_b', \
                               bounds = {'amplitude' : (0.0, None)})
            
            ## Tie mean of broad Hb to broad Ha
            def tie_mean_hb_b(model):
                return ((4862.683/6564.312)*nii_ha_bestfit['ha_b'].mean)

            g_hb_b.mean.tied = tie_mean_hb_b
            g_hb_b.mean.fixed = True

            ## Tie standard deviation of Hb to Ha in velocity space
            def tie_std_hb_b(model):
                return ((model['hb_b'].mean/nii_ha_bestfit['ha_b'].mean)*\
                       nii_ha_bestfit['ha_b'].stddev)

            g_hb_b.stddev.tied = tie_std_hb_b
            g_hb_b.stddev.fixed = True

            g_hb = g_hb + g_hb_b

        ## Initial Fit
        g_init = g_hb
        fitter = fitting.LevMarLSQFitter()
        gfit = fitter(g_init, lam_hb, flam_hb, \
                     weights = np.sqrt(ivar_hb), maxiter = 1000)

        ## Return with/without broad component depending on the presence of broad line in Ha
        return (gfit)

####################################################################################################
    
    def fit_hb_two_components(lam_hb, flam_hb, ivar_hb, nii_ha_bestfit):
        """
        Function to fit Hb emission-line, when [SII] has two components.
        The width of narrow (outflow) Hb line is fixed to narrow (outflow) Ha component.
        The width of broad Hb is fixed to broad Hb, if available.
        
        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region where the fits need to be performed.

        flam_hb : numpy array
            Flux array of the spectra in the Hb region.

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        nii_ha_bestfit : astropy model fit
            Best fit for [NII]+Ha emission lines.
            Sigma of narrow Hb is fixed to narrow Ha
            Sigma of outflow Hb is fixed to outflow Hb
            Sigma of broad Hb is fixed to broad Ha (if available)

        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component.
            
        """
        
        ## Initial estimate of amplitude of Hb
        amp_hb = np.max(flam_hb[(lam_hb >= 4861)&(lam_hb <=4863)])

        ## Std_Ha
        std_ha = nii_ha_bestfit['ha_n'].stddev.value

        ## Starting value for Hb
        std_hb = (4862.683/nii_ha_bestfit['ha_n'].mean.value)*std_ha

        ## Narrow Hb Gaussian
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                           stddev = std_hb, name = 'hb_n', \
                           bounds = {'amplitude' : (0.0, None)})
        
        ## Tie mean of Hb to Ha
        def tie_mean_hb(model):
            return ((4862.683/6564.312)*nii_ha_bestfit['ha_n'].mean)
        
        g_hb_n.mean.tied = tie_mean_hb
        g_hb_n.mean.fixed = True

        ## Tie standard deviation of Hb to Ha in velocity space
        def tie_std_hb(model):
            return ((model['hb_n'].mean/nii_ha_bestfit['ha_n'].mean)*\
                   nii_ha_bestfit['ha_n'].stddev)

        g_hb_n.stddev.tied = tie_std_hb
        g_hb_n.stddev.fixed = True

        ## Std_Ha_out
        std_ha_out = nii_ha_bestfit['ha_out'].stddev.value

        ## Starting value for Hb out
        std_hb_out = (4862.683/nii_ha_bestfit['ha_out'].mean.value)*std_ha_out

        ## Outflow Hb Gaussian
        g_hb_out = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                             stddev = std_hb_out, name = 'hb_out', \
                             bounds = {'amplitude' : (0.0, None)})
        
        ## Tie mean of outflow Hb to outflow Ha
        def tie_mean_hb_out(model):
            return ((4862.683/6564.312)*nii_ha_bestfit['ha_out'].mean)
        
        g_hb_out.mean.tied = tie_mean_hb_out
        g_hb_out.mean.fixed = True

        ## Tie standard deviation of outflow Hb to outflow Ha in velocity space
        def tie_std_hb_out(model):
            return ((model['hb_out'].mean/nii_ha_bestfit['ha_out'].mean)*\
                   nii_ha_bestfit['ha_out'].stddev)

        g_hb_out.stddev.tied = tie_std_hb_out
        g_hb_out.stddev.fixed = True

        ## Continuum
        cont = Const1D(amplitude = 0.0, name = 'hb_cont')

        g_hb = cont + g_hb_n + g_hb_out

        if ('ha_b' in nii_ha_bestfit.submodel_names):
            ## Std_Ha_b
            std_ha_b = nii_ha_bestfit['ha_b'].stddev.value

            ## Starting value for Hb broad
            std_hb_b = (4862.683/nii_ha_bestfit['ha_b'].mean.value)*std_ha_b

            ## Broad Hb Gaussian
            g_hb_b = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                               stddev = std_hb_b, name = 'hb_b', \
                               bounds = {'amplitude' : (0.0, None)})
            
            ## Tie mean of broad Hb to broad Ha
            def tie_mean_hb_b(model):
                return ((4862.683/6564.312)*nii_ha_bestfit['ha_b'].mean)

            g_hb_b.mean.tied = tie_mean_hb_b
            g_hb_b.mean.fixed = True
            
            ## Tie standard deviation of broad Hb to broad Ha in velocity space
            def tie_std_hb_b(model):
                return ((model['hb_b'].mean/nii_ha_bestfit['ha_b'].mean)*\
                       nii_ha_bestfit['ha_b'].stddev)

            g_hb_b.stddev.tied = tie_std_hb_b
            g_hb_b.stddev.fixed = True

            g_hb = g_hb + g_hb_b

        ## Initial Fit
        g_init = g_hb
        fitter = fitting.LevMarLSQFitter()
        gfit = fitter(g_init, lam_hb, flam_hb, \
                     weights = np.sqrt(ivar_hb), maxiter = 1000)

        ## Return with/without broad component depending on the presence of broad line in Ha
        return (gfit)

####################################################################################################
####################################################################################################

class fit_extreme_broadline_sources:
    """
    Different functions associated with fitting extreme broadline sources:
        1) fit_nii_ha_sii(lam_nii_ha_sii, flam_nii_ha_sii, ivar_nii_ha_sii)
        2) fit_hb_oiii_1comp(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, nii_ha_sii_bestfit)
        3) fit_hb_oiii_2comp(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, nii_ha_sii_bestfit)
        
    """
    def fit_nii_ha_sii(lam_nii_ha_sii, flam_nii_ha_sii, ivar_nii_ha_sii, priors = [5, 8]):
        """
        Function to fit [NII]+Ha+[SII] together for extreme broadline (quasar-like) sources
        The widths of all the narrow-line components are tied together.
        The function also allows for a broad component in Ha.

        Parameters
        ----------
        lam_nii_ha_sii : numpy array
            Wavelength array of the [NII]+Ha+[SII] region.

        flam_nii_ha_sii : numpy array
            Flux array of the spectra in the [NII]+Ha+[SII] region.

        ivar_nii_ha_sii : numpy array
            Inverse Variance array of the spectra in the [NII]+Ha+[SII] region.
            
        priors : list
            Initial priors for the amplitude and stddev of the broad component

        Returns
        -------
        gfit : Astropy model
            Best-fit model for the [NII]+Ha+[SII] region "without-broad" or "with-broad" component.
            Depends on what the broad_comp is set to
        """
        
        ############################ [SII]6716,6731 doublet ########################
        ## Initial estimate of amplitudes
        amp_sii6716 = np.max(flam_nii_ha_sii[(lam_nii_ha_sii >= 6716)&(lam_nii_ha_sii <= 6719)])
        amp_sii6731 = np.max(flam_nii_ha_sii[(lam_nii_ha_sii >= 6731)&(lam_nii_ha_sii <= 6734)])
        
        ## Initial gaussian fits
        g_sii6716 = Gaussian1D(amplitude = amp_sii6716, mean = 6718.294, \
                               stddev = 2.0, name = 'sii6716', \
                               bounds = {'amplitude': (0.0, None), 'stddev':(0.0, None)})
        g_sii6731 = Gaussian1D(amplitude = amp_sii6731, mean = 6732.673, \
                              stddev = 2.0, name = 'sii6731', \
                              bounds = {'amplitude': (0.0, None), 'stddev':(0.0, None)})

        ## Tie means of the two gaussians
        def tie_mean_sii(model):
            return ((6732.673/6718.294)*model['sii6716'].mean)

        g_sii6731.mean.tied = tie_mean_sii

        ## Tie sigma of the two gaussians in velocity space
        def tie_std_sii(model):
            return ((model['sii6716'].stddev)*(model['sii6731'].mean/model['sii6716'].mean))

        g_sii6731.stddev.tied = tie_std_sii

        g_sii = g_sii6716 + g_sii6731

        ############################ [NII]6548,6583 doublet ########################
        ## Initial estimate of amplitudes
        amp_nii6548 = np.max(flam_nii_ha_sii[(lam_nii_ha_sii > 6542)&(lam_nii_ha_sii < 6552)])
        amp_nii6583 = np.max(flam_nii_ha_sii[(lam_nii_ha_sii > 6580)&(lam_nii_ha_sii < 6590)])

        ## Initial gaussian fits
        g_nii6548 = Gaussian1D(amplitude = amp_nii6548, mean = 6549.852, \
                              stddev = 2.0, name = 'nii6548', \
                              bounds = {'amplitude':(0.0, None), 'stddev':(0.0, None)})
        g_nii6583 = Gaussian1D(amplitude = amp_nii6583, mean = 6585.277, \
                              stddev = 2.0, name = 'nii6583', \
                              bounds = {'amplitude':(0.0, None), 'stddev':(0.0, None)})
        
        ## Tie means of [NII] to [SII]
        def tie_mean_nii_sii(model):
            return ((6549.852/6718.294)*model['sii6716'].mean)
        
        g_nii6548.mean.tied = tie_mean_nii_sii 

        ## Tie means of the two gaussians
        def tie_mean_nii(model):
            return ((6585.277/6549.852)*model['nii6548'].mean)

        g_nii6583.mean.tied = tie_mean_nii

        ## Tie amplitudes of the two gaussians
        def tie_amp_nii(model):
            return (model['nii6548'].amplitude*2.96)

        g_nii6583.amplitude.tied = tie_amp_nii

        ## Tie sigma of both Gaussians to [SII] in velocity space
        def tie_std_nii6548(model):
            return ((model['sii6716'].stddev)*(model['nii6548'].mean/model['sii6716'].mean))

        g_nii6548.stddev.tied = tie_std_nii6548

        def tie_std_nii6583(model):
            return ((model['sii6716'].stddev)*(model['nii6583'].mean/model['sii6716'].mean))

        g_nii6583.stddev.tied = tie_std_nii6583

        g_nii = g_nii6548 + g_nii6583
        
        ############################ Continuum #####################################

        cont = Const1D(amplitude = 0.0, name = 'nii_ha_sii_cont')

        ############################ HALPHA ########################################
        ## Initial estimate of amplitude
        amp_ha = np.max(flam_nii_ha_sii[(lam_nii_ha_sii > 6560)&(lam_nii_ha_sii < 6568)])

        ## Initial gaussian fits
        g_ha_n = Gaussian1D(amplitude = amp_ha, mean = 6564.312, \
                           stddev = 2.0, name = 'ha_n', \
                           bounds = {'amplitude':(0.0, None), 'stddev':(0.0, None)})
        
        ## Tie mean of narrow Ha to narrow [NII]
        def tie_mean_ha(model):
            return ((6564.312/6549.852)*model['nii6548'].mean)
        
        g_ha_n.mean.tied = tie_mean_ha        
        
        ## Tie sigma of narrow Ha to [SII] in velocity space
        def tie_std_ha(model):
            return ((model['sii6716'].stddev)*(model['ha_n'].mean/model['sii6716'].mean))

        g_ha_n.stddev.tied = tie_std_ha
        
        ## Broad Ha component
        g_ha_b = Gaussian1D(amplitude = amp_ha/priors[0], mean = 6564.312, \
                           stddev = priors[1], name = 'ha_b', \
                           bounds = {'amplitude':(0.0, None), 'stddev':(0.0, None)})

        ## Initial Fit
        g_init = cont + g_nii + g_ha_n + g_ha_b + g_sii
        fitter = fitting.LevMarLSQFitter()
        gfit = fitter(g_init, lam_nii_ha_sii, flam_nii_ha_sii, \
                         weights = np.sqrt(ivar_nii_ha_sii), maxiter = 1000)

        return (gfit)

####################################################################################################

    def fit_hb_oiii_1comp(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, nii_ha_sii_bestfit):
        """
        Function to fit Hb+[OIII] together for extreme broadline (quasar-like) sources
        The widths of [OIII] are tied together and the widths of narrow and broad Hb components 
        are tied to the narrow and broad Ha. 
        This function fits one component each for the [OIII] doublet. 
        It also allows for a broad component for Hb.

        Parameters
        ----------
        lam_hb_oiii : numpy array
            Wavelength array of the Hb+[OIII] region.

        flam_hb_oiii : numpy array
            Flux array of the spectra in the Hb+[OIII] region.

        ivar_hb_oiii : numpy array
            Inverse variance array of the spectra in the Hb+[OIII] region.
            
        nii_ha_sii_bestfit : Astropy model
            Best fit model for the [NII]+Ha+[SII] emission-lines.

        Returns
        -------
        gfit : Astropy model
            Best-fit model for the Hb+[OIII] region with a broad component   
        """
        ############################ [OIII]4959,5007 doublet #######################
        ## Initial estimates of amplitude
        amp_oiii4959 = np.max(flam_hb_oiii[(lam_hb_oiii >= 4959)&(lam_hb_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_hb_oiii[(lam_hb_oiii >= 5007)&(lam_hb_oiii <= 5009)])

        ## Initial gaussian fits
        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959, mean = 4960.295, \
                               stddev = 1.0, name = 'oiii4959', \
                               bounds = {'amplitude' : (0.0, None), 'stddev':(0.0, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007, mean = 5008.239, \
                               stddev = 1.0, name = 'oiii5007', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        ## Tie means of the two gaussians
        def tie_mean_oiii(model):
            return ((5008.239/4960.295)*model['oiii4959'].mean)

        g_oiii5007.mean.tied = tie_mean_oiii

        ## Tie amplitudes of the two gaussians
        def tie_amp_oiii(model):
            return (model['oiii4959'].amplitude*2.98)

        g_oiii5007.amplitude.tied = tie_amp_oiii

        ## Tie standard deviations in velocity space
        def tie_std_oiii(model):
            return ((model['oiii4959'].stddev)*\
                   (model['oiii5007'].mean/model['oiii4959'].mean))

        g_oiii5007.stddev.tied = tie_std_oiii

        g_oiii = g_oiii4959 + g_oiii5007
        
        ############################ Continuum #####################################

        cont = Const1D(amplitude = 0.0, name = 'hb_oiii_cont')

        ############################ HBETA #########################################
        ## Initial estimate of amplitude
        amp_hb = np.max(flam_hb_oiii[(lam_hb_oiii >= 4860)&(lam_hb_oiii <= 4864)])
        
        ha_n_std = nii_ha_sii_bestfit['ha_n'].stddev.value
        ## Initial estimates of standard deviation for Hb
        std_hb_n = (4862.683/nii_ha_sii_bestfit['ha_n'].mean.value)*ha_n_std
        
        ## Initial gaussian fits
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                           stddev = std_hb_n, name = 'hb_n', \
                           bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        
        ## Tie mean of Hb to Ha
        def tie_mean_hb(model):
            return ((4862.683/6564.312)*nii_ha_sii_bestfit['ha_n'].mean)
        
        g_hb_n.mean.tied = tie_mean_hb
        g_hb_n.mean.fixed = True
        
        ## Fix sigma of narrow Hb to narrow Ha
        def tie_std_hb_n(model):
            return ((model['hb_n'].mean/nii_ha_sii_bestfit['ha_n'].mean)*\
                   nii_ha_sii_bestfit['ha_n'].stddev)

        g_hb_n.stddev.tied = tie_std_hb_n
        g_hb_n.stddev.fixed = True
                
        ## Broad component
        ## Initial values 
        ha_b_std = nii_ha_sii_bestfit['ha_b'].stddev.value
        std_hb_b = (4862.683/nii_ha_sii_bestfit['ha_b'].mean.value)*ha_b_std

        ## Broad Hb Gaussian
        g_hb_b = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                           stddev = std_hb_b, name = 'hb_b', \
                           bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        ## Tie mean of broad Hb to broad Ha
        def tie_mean_hb_b(model):
            return ((4862.683/6564.312)*nii_ha_sii_bestfit['ha_b'].mean)

        g_hb_b.mean.tied = tie_mean_hb_b
        g_hb_b.mean.fixed = True

        ## Fix sigma of broad Hb to broad Ha
        def tie_std_hb_b(model):
            return ((model['hb_b'].mean/nii_ha_sii_bestfit['ha_b'].mean)*\
                   nii_ha_sii_bestfit['ha_b'].stddev)

        g_hb_b.stddev.tied = tie_std_hb_b
        g_hb_b.stddev.fixed = True

        g_hb = g_hb_n + g_hb_b

        ## Initial Fit
        g_init = cont + g_hb + g_oiii
        fitter = fitting.LevMarLSQFitter()

        gfit = fitter(g_init, lam_hb_oiii, flam_hb_oiii, \
                     weights = np.sqrt(ivar_hb_oiii), maxiter = 1000)

        return (gfit)

####################################################################################################

    def fit_hb_oiii_2comp(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, nii_ha_sii_bestfit):
        """
        Function to fit Hb+[OIII] together for extreme broadline (quasar-like) sources
        The widths of [OIII] are tied together and the widths of narrow and broad Hb components 
        are tied to the narrow and broad Ha. 
        This function fits two components each for the [OIII] doublet. 
        It also allows for a broad component for Hb.

        Parameters
        ----------
        lam_hb_oiii : numpy array
            Wavelength array of the Hb+[OIII] region.

        flam_hb_oiii : numpy array
            Flux array of the spectra in the Hb+[OIII] region.

        ivar_hb_oiii : numpy array
            Inverse variance array of the spectra in the Hb+[OIII] region.
            
        nii_ha_sii_bestfit : Astropy model
            Best fit model for the [NII]+Ha+[SII] emission-lines.

        Returns
        -------
        gfit : Astropy model
            Best-fit model for the Hb+[OIII] region with a broad component   
        """

        ############################ [OIII]4959,5007 doublet #######################
        ## Initial estimates of amplitude
        amp_oiii4959 = np.max(flam_hb_oiii[(lam_hb_oiii >= 4959)&(lam_hb_oiii <= 4961)])
        amp_oiii5007 = np.max(flam_hb_oiii[(lam_hb_oiii >= 5007)&(lam_hb_oiii <= 5009)])

        ## Initial gaussian fits
        g_oiii4959 = Gaussian1D(amplitude = amp_oiii4959/2, mean = 4960.295, \
                               stddev = 1.0, name = 'oiii4959', \
                               bounds = {'amplitude' : (0.0, None), 'stddev':(0.0, None)})
        g_oiii5007 = Gaussian1D(amplitude = amp_oiii5007/2, mean = 5008.239, \
                               stddev = 1.0, name = 'oiii5007', \
                               bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_oiii4959_out = Gaussian1D(amplitude = amp_oiii4959/4, mean = 4960.295, \
                                   stddev = 4.0, name = 'oiii4959_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        g_oiii5007_out = Gaussian1D(amplitude = amp_oiii5007/4, mean = 5008.239, \
                                   stddev = 4.0, name = 'oiii5007_out', \
                                   bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        ## Tie means of the narrow components
        def tie_mean_oiii(model):
            return ((5008.239/4960.295)*model['oiii4959'].mean)

        g_oiii5007.mean.tied = tie_mean_oiii

        ## Tie amplitudes of the narrow components
        def tie_amp_oiii(model):
            return (model['oiii4959'].amplitude*2.98)

        g_oiii5007.amplitude.tied = tie_amp_oiii

        ## Tie standard deviations of narrow components in velocity space
        def tie_std_oiii(model):
            return ((model['oiii4959'].stddev)*\
                   (model['oiii5007'].mean/model['oiii4959'].mean))

        g_oiii5007.stddev.tied = tie_std_oiii

        ## Tie means of outflow components
        def tie_mean_oiii_out(model):
            return ((5008.239/4960.295)*model['oiii4959_out'].mean)

        g_oiii5007_out.mean.tied = tie_mean_oiii_out

        ## Tie amplitudes of the outflow components
        def tie_amp_oiii_out(model):
            return (model['oiii4959_out'].amplitude*2.98)

        g_oiii5007_out.amplitude.tied = tie_amp_oiii_out

        ## Tie standard deviations of outflow components in velocity space
        def tie_std_oiii_out(model):
            return ((model['oiii4959_out'].stddev)*\
                   (model['oiii5007_out'].mean/model['oiii4959_out'].mean))

        g_oiii5007_out.stddev.tied = tie_std_oiii_out

        g_oiii = g_oiii4959 + g_oiii5007 + g_oiii4959_out + g_oiii5007_out
        
        ############################ Continuum #####################################

        cont = Const1D(amplitude = 0.0, name = 'hb_oiii_cont')
        
        ############################ HBETA #########################################

        ## Initial estimate of amplitude
        amp_hb = np.max(flam_hb_oiii[(lam_hb_oiii >= 4860)&(lam_hb_oiii <= 4864)])
        
        ha_n_std = nii_ha_sii_bestfit['ha_n'].stddev.value
        ## Initial estimates of standard deviation for Hb
        std_hb_n = (4862.683/nii_ha_sii_bestfit['ha_n'].mean.value)*ha_n_std
        
        ## Initial gaussian fits
        g_hb_n = Gaussian1D(amplitude = amp_hb, mean = 4862.683, \
                           stddev = std_hb_n, name = 'hb_n', \
                           bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})
        
        ## Tie mean of Hb to Ha
        def tie_mean_hb(model):
            return ((4862.683/6564.312)*nii_ha_sii_bestfit['ha_n'].mean)
        
        g_hb_n.mean.tied = tie_mean_hb
        g_hb_n.mean.fixed = True
        
        ## Fix sigma of narrow Hb to narrow Ha
        def tie_std_hb_n(model):
            return ((model['hb_n'].mean/nii_ha_sii_bestfit['ha_n'].mean)*\
                   nii_ha_sii_bestfit['ha_n'].stddev)

        g_hb_n.stddev.tied = tie_std_hb_n
        g_hb_n.stddev.fixed = True
        
        ## Broad component
        ## Initial values 
        ha_b_std = nii_ha_sii_bestfit['ha_b'].stddev.value
        std_hb_b = (4862.683/nii_ha_sii_bestfit['ha_b'].mean.value)*ha_b_std

        ## Broad Hb Gaussian
        g_hb_b = Gaussian1D(amplitude = amp_hb/2, mean = 4862.683, \
                           stddev = std_hb_b, name = 'hb_b', \
                           bounds = {'amplitude' : (0.0, None), 'stddev' : (0.0, None)})

        ## Tie mean of broad Hb to broad Ha
        def tie_mean_hb_b(model):
            return ((4862.683/6564.312)*nii_ha_sii_bestfit['ha_b'].mean)

        g_hb_b.mean.tied = tie_mean_hb_b
        g_hb_b.mean.fixed = True

        ## Fix sigma of broad Hb to broad Ha
        def tie_std_hb_b(model):
            return ((model['hb_b'].mean/nii_ha_sii_bestfit['ha_b'].mean)*\
                   nii_ha_sii_bestfit['ha_b'].stddev)

        g_hb_b.stddev.tied = tie_std_hb_b
        g_hb_b.stddev.fixed = True

        g_hb = g_hb_n + g_hb_b

        ## Initial Fit
        g_init = cont + g_hb + g_oiii
        fitter = fitting.LevMarLSQFitter()

        gfit = fitter(g_init, lam_hb_oiii, flam_hb_oiii, \
                     weights = np.sqrt(ivar_hb_oiii), maxiter = 1000)
        
        ## Set the broad component as the "outflow" component
        oiii_out_sig = mfit.lamspace_to_velspace(gfit['oiii5007_out'].stddev.value, \
                                                 gfit['oiii5007_out'].mean.value)
        oiii_sig = mfit.lamspace_to_velspace(gfit['oiii5007'].stddev.value, \
                                            gfit['oiii5007'].mean.value)
        
        if (oiii_out_sig < oiii_sig):
            gfit_oiii4959 = Gaussian1D(amplitude = gfit['oiii4959_out'].amplitude, \
                                      mean = gfit['oiii4959_out'].mean, \
                                      stddev = gfit['oiii4959_out'].stddev, \
                                      name = 'oiii4959')
            gfit_oiii5007 = Gaussian1D(amplitude = gfit['oiii5007_out'].amplitude, \
                                      mean = gfit['oiii5007_out'].mean, \
                                      stddev = gfit['oiii5007_out'].stddev, \
                                      name = 'oiii5007')
            gfit_oiii4959_out = Gaussian1D(amplitude = gfit['oiii4959'].amplitude, \
                                          mean = gfit['oiii4959'].mean, \
                                          stddev = gfit['oiii4959'].stddev, \
                                          name = 'oiii4959_out')
            gfit_oiii5007_out = Gaussian1D(amplitude = gfit['oiii5007'].amplitude, \
                                          mean = gfit['oiii5007'].mean, \
                                          stddev = gfit['oiii5007'].stddev, \
                                          name = 'oiii5007_out')
            cont = gfit['hb_oiii_cont'] 
            
            gfit_hb = gfit['hb_n'] + gfit['hb_b']
                
            gfit = cont + gfit_hb + gfit_oiii4959 + gfit_oiii5007 + \
            gfit_oiii4959_out + gfit_oiii5007_out

        return (gfit)

####################################################################################################
####################################################################################################