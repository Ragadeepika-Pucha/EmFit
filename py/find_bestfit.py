"""
This script consists of functions for fitting emission-lines. 
The different functions are divided into different classes for different emission lines.

Author : Ragadeepika Pucha
Version : 2024, March 14
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import measure_fits as mfit
import fit_lines as fl

from scipy.stats import chi2

###################################################################################################

def find_sii_best_fit(lam_sii, flam_sii, ivar_sii):
    """
    Find the best fit for [SII]6716,6731 doublet.
    The code fits both one-component and two-component fits and picks the best version.
    The two-component fit is picked if the p-value for chi2 distribution is < 3e-7 --> 
    5-sigma confidence for an extra component statistically.
    
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
    sii_bestfit : Astropy model
        Best-fit 1 component or 2 component model
    
    n_dof : int
        Number of degrees of freedom
        
    sii_flag : int
        Flags based on some decisions in selecting one- or two-component fits.
    """
    
    ## Single component fit
    gfit_1comp = fl.fit_sii_lines.fit_one_component(lam_sii, flam_sii, ivar_sii)
    
    ## Two-component fit
    gfit_2comp = fl.fit_sii_lines.fit_two_components(lam_sii, flam_sii, ivar_sii)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_sii, gfit_1comp(lam_sii), ivar_sii)
    chi2_2comp = mfit.calculate_chi2(flam_sii, gfit_2comp(lam_sii), ivar_sii)
    
    ## Statistical check for the second component
    df = 8-5
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Critetion to have min([SII]) in two-component model to be 35 km/s
    sig_sii = mfit.lamspace_to_velspace(gfit_2comp['sii6716'].stddev.value, \
                                       gfit_2comp['sii6716'].mean.value)
    
    ## Criterion for defaulting back to one-component model
    ## rel-redshift > 450 km/s or < -450 km/s
    ## [SII]outflow sigma > 800 km/s or < 1000 km/s
    mean_sii = gfit_2comp['sii6716'].mean.value
    mean_sii_out = gfit_2comp['sii6716_out'].mean.value
    sig_sii_out = mfit.lamspace_to_velspace(gfit_2comp['sii6716_out'].stddev.value, \
                                           gfit_2comp['sii6716_out'].mean.value)
    
    delz_sii = (mean_sii_out - mean_sii)*3e+5/6718.294
    
    default_cond = (delz_sii < -450)|(delz_sii > 450)|((sig_sii_out > 600)&(sig_sii_out < 1000))
    
    ## If the sigma ([SII]) > 450 km/s in a single-component model
    ## Default back to two-component model
    sig_sii_1comp = mfit.lamspace_to_velspace(gfit_1comp['sii6716'].stddev.value, \
                                             gfit_1comp['sii6716'].mean.value)
    
    ### Set [SII] Flags
    sii_flags = []
    if (sig_sii < 35):
        sii_flags.append(0)
    if ((delz_sii < -450)|(delz_sii > 450)):
        sii_flags.append(1)
    if ((sig_sii_out > 600)&(sig_sii_out < 1000)):
        sii_flags.append(2)
    if (sig_sii_1comp > 450):
        sii_flags.append(3)
        
    sii_flags = np.array(sii_flags)
    
    if (len(sii_flags) == 0):
        sii_flag = 0
    else:
        sii_flag = sum(2**sii_flags)
    
    ## 5-sigma confidence of an extra component
    if ((p_val <= 3e-7)&(sig_sii >= 35)&(~default_cond|(sig_sii_1comp > 450))):
        sii_bestfit = gfit_2comp
        n_dof = 8
    else:
        sii_bestfit = gfit_1comp
        n_dof = 5
        
    return (sii_bestfit, n_dof, sii_flag)

####################################################################################################
####################################################################################################

def find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii):
    """
    Find the best fit for [OIII]4959,5007 doublet.
    The code fits both one-component and two-component fits and picks the best version.
    The two-component fit is picked if the p-value for chi2 distribution is < 3e-7 --> 
    5-sigma confidence for an extra component statistically.
    
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
    oiii_bestfit : Astropy model
        Best-fit 1 component or 2 component model
    
    n_dof : int
        Number of degrees of freedom
    """
    
    ## Single component fit
    gfit_1comp = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Two component fit
    gfit_2comp = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, ivar_oiii)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_oiii, gfit_1comp(lam_oiii), ivar_oiii)
    chi2_2comp = mfit.calculate_chi2(flam_oiii, gfit_2comp(lam_oiii), ivar_oiii)
    
    ## Statistical check for the second component
    df = 7-4
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Criterion to have min([OIII]) in two-component model to be 35 km/s
    sig_oiii = mfit.lamspace_to_velspace(gfit_2comp['oiii5007'].stddev.value, \
                                       gfit_2comp['oiii5007'].mean.value)
    
    ## Criterion for defaulting back to one-component model
    ## Sigma ([OIII]out) > 1000 km/s
    sig_oiii_out = mfit.lamspace_to_velspace(gfit_2comp['oiii5007_out'].stddev.value, \
                                            gfit_2comp['oiii5007_out'].mean.value)
    
    default_cond = (sig_oiii_out > 1000)
    
    ## 5-sigma confidence of an extra component
    if ((p_val <= 3e-7)&(sig_oiii >= 35)&(~default_cond)):
        oiii_bestfit = gfit_2comp
        n_dof = 7
    else:
        oiii_bestfit = gfit_1comp
        n_dof = 4
        
    return (oiii_bestfit, n_dof)
    
####################################################################################################
####################################################################################################

class nii_ha_fit:
    """
    This class contains functions related to [NII]+Ha Fitting:
        1) free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit)
        2) fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit)
        3) fixed_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit)
    """
    def free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):    
        """
        Find bestfit for [NII]+Ha emission-lines while keeping Ha is free to vary.
        [NII] is kept fixed to [SII], and all the narrow lines have a single component.
        
        The code fits both broad and non-broad component fits and picks the best version.
        The broad-component fit is picked if the p-value for chi2 distribution is < 3e-7 --> 
        5-sigma confidence for an extra component statistically.    
        
        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            
        n_dof : int
            Number of degrees of freedom
            
        psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
        """
    
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                     ivar_nii_ha, sii_bestfit, \
                                                                     broad_comp = False)

        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                     ivar_nii_ha, sii_bestfit, \
                                                                     priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2)]
            
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 8-5
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow Ha flux is zero, but broad Ha flux is not zero
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
        
        default_cond = (ha_n_flux == 0)&(ha_b_flux != 0)
        
        if (default_cond):
            nii_ha_flag = 1
        else:
            nii_ha_flag = 0

        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)&(~default_cond)):
            nii_ha_bestfit = gfit_b
            n_dof = 8
            psel = psel
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 5
            psel = []
            
        return (nii_ha_bestfit, n_dof, psel, nii_ha_flag)
    
####################################################################################################

    def fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
        """
        Find bestfit for [NII]+Ha emission-lines while keeping Ha fixed to [SII].
        [NII] is kept fixed to [SII], and all the narrow lines have a single component.
        
        The code fits both broad and non-broad component fits and picks the best version.
        The broad-component fit is picked if the p-value for chi2 distribution is < 3e-7 --> 
        5-sigma confidence for an extra component statistically.    
        
        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            
        n_dof : int
            Number of degrees of freedom
            
        psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
        """
        
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                 ivar_nii_ha, sii_bestfit, \
                                                                 broad_comp = False)
        
        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                     ivar_nii_ha, sii_bestfit, \
                                                                     priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2)]

        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 7-4
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow Ha flux is zero, but broad Ha flux is not zero
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
        
        default_cond = (ha_n_flux == 0)&(ha_b_flux != 0)
        
        if (default_cond):
            nii_ha_flag = 1
        else:
            nii_ha_flag = 0

        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)&(~default_cond)):
            nii_ha_bestfit = gfit_b
            n_dof = 7
            psel = psel
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 4
            psel = []

        return (nii_ha_bestfit, n_dof, psel, nii_ha_flag)
    
####################################################################################################
    
    def fixed_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
        """
        Find bestfit for [NII]+Ha emission-lines while keeping Ha fixed to [SII].
        [NII] is kept fixed to [SII], and all the narrow lines have two components.
        
        The code fits both broad and non-broad component fits and picks the best version.
        The broad-component fit is picked if the p-value for chi2 distribution is < 3e-7 --> 
        5-sigma confidence for an extra component statistically.    
        
        Parameters
        ----------
        lam_nii_ha : numpy array
            Wavelength array of the [NII]+Ha region where the fits need to be performed.

        flam_nii_ha : numpy array
            Flux array of the spectra in the [NII]+Ha region.

        ivar_nii_ha : numpy array
            Inverse variance array of the spectra in the [NII]+Ha region.
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        Returns
        -------
        gfit : Astropy model
            Best-fit "without-broad" or "with-broad" component
            
        n_dof : int
            Number of degrees of freedom
            
        psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
        """
        
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                  ivar_nii_ha, sii_bestfit, \
                                                                  broad_comp = False)

        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                 ivar_nii_ha, sii_bestfit, \
                                                                 priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2)]

        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 9-6
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow/outflow Ha flux is zero, but broad Ha flux is not zero
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
        ha_out_flux = mfit.compute_emline_flux(gfit_b['ha_out'].amplitude.value, \
                                              gfit_b['ha_out'].stddev.value)
        
        default_cond = ((ha_n_flux == 0)|(ha_out_flux == 0))&(ha_b_flux != 0)
        
        if (default_cond):
            nii_ha_flag = 1
        else:
            nii_ha_flag = 0

        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)&(~default_cond)):
            nii_ha_bestfit = gfit_b
            n_dof = 9
            psel = psel
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 6
            psel = []

        return (nii_ha_bestfit, n_dof, psel, nii_ha_flag)
    
####################################################################################################
####################################################################################################

def find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
    """
    Find the best fit for [NII]+Ha emission lines.
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value of the chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of [NII] and Ha is same as [SII].
    
    Parameters
    ----------
    lam_nii_ha : numpy array
        Wavelength array of the [NII]+Ha region where the fits need to be performed.

    flam_nii_ha : numpy array
        Flux array of the spectra in the [NII]+Ha region.

    ivar_nii_ha : numpy array
        Inverse variance array of the spectra in the [NII]+Ha region.

    sii_bestfit : Astropy model
        Best fit model for the [SII] emission-lines.
        
    Returns
    -------
    nii_ha_bestfit : Astropy model
        Best-fit model for [NII]+Ha emission lines.

    n_dof : int
        Number of degrees of freedom
        
    psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
    """
    
    ## Functions change depending on the number of components in [SII]
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## First try free Ha version
        nii_ha_bestfit, n_dof, psel, nii_ha_flag = nii_ha_fit.free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, \
                                                                       sii_bestfit)
        
        ## How does Ha width compare to [SII] width?
        sig_sii = mfit.lamspace_to_velspace(sii_bestfit['sii6716'].stddev.value, \
                                           sii_bestfit['sii6716'].mean.value)
        
        sig_ha = mfit.lamspace_to_velspace(nii_ha_bestfit['ha_n'].stddev.value, \
                                         nii_ha_bestfit['ha_n'].mean.value)
        
        per_diff = (sig_ha - sig_sii)*100/sig_sii
        
        
        if ((per_diff < -30)|(per_diff >= 30)|(nii_ha_bestfit['ha_n'].amplitude.value == 0)):
            ## If sigma (Ha) is not within 30% of sigma ([SII]) -- used fixed Ha version
            nii_ha_bestfit, n_dof, psel, nii_ha_flag = nii_ha_fit.fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, \
                                                                            sii_bestfit)    
    else:
        nii_ha_bestfit, n_dof, psel, nii_ha_flag = nii_ha_fit.fixed_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                         ivar_nii_ha, sii_bestfit)
    return (nii_ha_bestfit, n_dof, psel, nii_ha_flag)        

####################################################################################################
####################################################################################################

def find_hb_best_fit(lam_hb, flam_hb, ivar_hb, nii_ha_bestfit):
    """
    Find the best fit for Hb emission-line. The number of components of Hb is same as Ha.
    The width of narrow/outflow/broad component of Hb is fixed to the 
    width of narrow/outflow/broad component of Ha.
    
    Parameters
    ----------
    lam_hb : numpy array
        Wavelength array of the Hb region
    
    flam_hb : numpy array
        Flux array of the spectra in the Hb region
        
    ivar_hb : numpy array
        Inverse variance array of the spectra in the Hb region
        
    nii_ha_bestfit : Astropy model
        Best fit model for the [NII]+Ha region
        
    Returns
    -------
    hb_bestfit : Astropy model
        Best-fit model for the Hb emission line
        
    n_dof : int
        Number of degrees of freedom
    """
    
    ha_models = nii_ha_bestfit.submodel_names
    
    if ('ha_out' not in ha_models):
        ## Single component Fit
        hb_bestfit = fl.fit_hb_line.fit_hb_one_component(lam_hb, flam_hb, \
                                                        ivar_hb, nii_ha_bestfit)
        
        if ('hb_b' not in hb_bestfit.submodel_names):
            n_dof = 2
        else:
            n_dof = 3
            
    else:
        ## Two components fit
        hb_bestfit = fl.fit_hb_line.fit_hb_two_components(lam_hb, flam_hb, \
                                                         ivar_hb, nii_ha_bestfit)
        
        if ('hb_b' not in hb_bestfit.submodel_names):
            n_dof = 3
        else:
            n_dof = 4
            
    ## Returns the bestfit
    return (hb_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_nii_ha_sii_best_fit(lam_nii_ha_sii, flam_nii_ha_sii, ivar_nii_ha_sii):
    """
    Find the bestfit for [NII]+Ha+[SII]region. This is for the case of 
    extreme broadline (quasar-like) sources. 
    
    Parameters
    ----------
    lam_nii_ha_sii : numpy array
        Wavelength array of the [NII]+Ha+[SII] region.

    flam_nii_ha_sii : numpy array
        Flux array of the spectra in the [NII]+Ha+[SII] region.

    ivar_nii_ha_sii : numpy array
        Inverse variance array of the spectra in the [NII]+Ha+[SII] region.
        
    Returns
    -------
    nii_ha_sii_bestfit : Astropy model
        Best-fit model for the [NII]+Ha+[SII] region with a broad component   
        
    n_dof : int
        Number of degrees of freedom
        
    psel : list
            Selected prior for the broad component
    """
    
    ## Test with different priors and select the one with the least chi2
    priors_list = [[3,6], [5,8]]
    gfits = []
    chi2s = []
    
    for p in priors_list:
        gfit = fl.fit_extreme_broadline_sources.fit_nii_ha_sii(lam_nii_ha_sii,\
                                                               flam_nii_ha_sii, \
                                                               ivar_nii_ha_sii, priors = p)
        chi2_fit = mfit.calculate_chi2(flam_nii_ha_sii, gfit(lam_nii_ha_sii), ivar_nii_ha_sii)
        gfits.append(gfit)
        chi2s.append(chi2_fit)
        
    ## Select the fit with the minimum chi2
    nii_ha_sii_bestfit = gfits[np.argmin(chi2s)]
    n_dof = 10
    
    ## Select the prior that leads to the bestfit
    psel = priors_list[np.argmin(chi2)]
    
    return (nii_ha_sii_bestfit, n_dof, psel)

####################################################################################################
####################################################################################################

def find_hb_oiii_bestfit(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, nii_ha_sii_bestfit):
    """
    Find the bestfit for the Hb+[OIII] region. This is for the case of 
    extreme broadline (quasar-like) sources.
    The code fits both one-component and two-component fits for [OIII] doublet and 
    picks the best version.
    The two-component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    
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
    hb_oiii_bestfit : Astropy model
        Best-fit model for the Hb+[OIII] region with a broad component 
        
    n_dof : int
        Number of degrees of freedom
    """

    ## Single component fit
    gfit_1comp = fl.fit_extreme_broadline_sources.fit_hb_oiii_1comp(lam_hb_oiii, \
                                                                    flam_hb_oiii, \
                                                                    ivar_hb_oiii, \
                                                                    nii_ha_sii_bestfit)
    
    ## Two component fit
    gfit_2comp = fl.fit_extreme_broadline_sources.fit_hb_oiii_2comp(lam_hb_oiii, \
                                                                    flam_hb_oiii, \
                                                                    ivar_hb_oiii, \
                                                                    nii_ha_sii_bestfit)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_hb_oiii, gfit_1comp(lam_hb_oiii), ivar_hb_oiii)
    chi2_2comp = mfit.calculate_chi2(flam_hb_oiii, gfit_2comp(lam_hb_oiii), ivar_hb_oiii)
    
    ## Statistical check for the second component
    df = 9-6 
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Criterion for defaulting back to one-component model
    ## Sigma ([OIII]out) > 1000 km/s
    sig_oiii_out = mfit.lamspace_to_velspace(gfit_2comp['oiii5007_out'].stddev.value, \
                                            gfit_2comp['oiii5007_out'].mean.value)
    
    default_cond = (sig_oiii_out > 1000)
    
    ## 5-sigma confidence of an extra component
    if ((p_val <= 3e-7)&(~default_cond)):
        hb_oiii_bestfit = gfit_2comp
        n_dof = 9
    else:
        hb_oiii_bestfit = gfit_1comp
        n_dof = 6
        
    return (hb_oiii_bestfit, n_dof)

####################################################################################################
####################################################################################################
