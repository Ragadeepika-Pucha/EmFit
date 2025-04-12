"""
This script consists of functions for finding the bestfit for the emission-lines.
It consists of the following functions:
    1) find_sii_best_fit(lam_sii, flam_sii, ivar_sii, rsig_sii)
    2) find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii)
    3) nii_ha_fit.free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                        sii_bestfit, rsig_sii)
    4) nii_ha_fit.fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                        sii_bestfit, rsig_sii)
    5) nii_ha_fit.fixed_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                        sii_bestfit, rsig_sii)
    6) find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                            sii_bestfit, rsig_sii)
    7) find_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb, nii_ha_bestfit, rsig_nii_ha)
    8) find_free_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb)
    9) find_nii_ha_sii_best_fit(lam_nii_ha_sii, flam_nii_ha_sii, ivar_nii_ha_sii, \
                                rsig_nii_ha_sii)
    10) find_hb_oiii_best_fit(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, rsig_hb_oiii, \
                            nii_ha_sii_bestfit, rsig_nii_ha_sii)
    11) highz_fit.find_free_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb)
    12) highz_fit.find_fixed_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                                        oiii_bestfit, rsig_oiii)
    13) find_nev_best_fit(lam_nev, flam_nev, ivar_nev, rsig_nev, \
                                        sii_bestfit, rsig_sii)

Author : Ragadeepika Pucha
Version : 2025, April 11
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import measure_fits as mfit
import fit_lines as fl

from scipy.stats import chi2

###################################################################################################

def find_sii_best_fit(lam_sii, flam_sii, ivar_sii, rsig_sii):
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
        
    rsig_sii : float
        Median Resolution element in the [SII] region.

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
    gfit_1comp = fl.fit_sii_lines.fit_one_component(lam_sii, flam_sii, ivar_sii, rsig_sii)
    
    ## Two-component fit
    gfit_2comp = fl.fit_sii_lines.fit_two_components(lam_sii, flam_sii, ivar_sii, rsig_sii)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_sii, gfit_1comp(lam_sii), ivar_sii)
    chi2_2comp = mfit.calculate_chi2(flam_sii, gfit_2comp(lam_sii), ivar_sii)
    
    ## Statistical check for the second component
    df = 8-5
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Criterion for two-component model --> narrow [SII] is resolved
    res_cond = (gfit_2comp['sii6716'].stddev.value > rsig_sii)&\
    (gfit_2comp['sii6731'].stddev.value > rsig_sii)
    
    ## Criterion for defaulting back to one-component model
    ## rel-redshift > 450 km/s or < -450 km/s
    ## [SII]outflow sigma > 600 km/s 
    mean_sii = gfit_2comp['sii6716'].mean.value
    mean_sii_out = gfit_2comp['sii6716_out'].mean.value
    sig_sii_out, _ = mfit.correct_for_rsigma(gfit_2comp['sii6716_out'].mean.value, \
                                            gfit_2comp['sii6716_out'].stddev.value, \
                                            rsig_sii)
    
    delz_sii = (mean_sii_out - mean_sii)*3e+5/6718.294
    
    ## If the amplitude ratio of (outflow/narrow) > 2
    ## default to one-component model
    amp_ratio = gfit_2comp['sii6716_out'].amplitude.value/gfit_2comp['sii6716'].amplitude.value

    default_cond = (delz_sii < -450)|(delz_sii > 450)|(sig_sii_out > 600)|(amp_ratio > 2)
    
    ## If the sigma ([SII]) > 450 km/s in a single-component model
    ## Default back to two-component model
    sig_sii_1comp, _ = mfit.correct_for_rsigma(gfit_1comp['sii6716'].mean.value, \
                                               gfit_1comp['sii6716'].stddev.value, \
                                               rsig_sii)
        
    ## 5-sigma confidence of an extra component
    if ((p_val <= 3e-7)&(res_cond)&(~default_cond|(sig_sii_1comp > 450))):
        sii_bestfit = gfit_2comp
        n_dof = 8
    else:
        sii_bestfit = gfit_1comp
        n_dof = 5
        
    return (sii_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii):
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
        
    rsig_oiii : float
        Median Resolution element in the [OIII] region.

    Returns
    -------
    oiii_bestfit : Astropy model
        Best-fit 1 component or 2 component model
    
    n_dof : int
        Number of degrees of freedom
    """
    
    ## Single component fit
    gfit_1comp = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii)
    
    ## Two component fit
    gfit_2comp = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, ivar_oiii, rsig_oiii)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_oiii, gfit_1comp(lam_oiii), ivar_oiii)
    chi2_2comp = mfit.calculate_chi2(flam_oiii, gfit_2comp(lam_oiii), ivar_oiii)
    
    ## Statistical check for the second component
    df = 7-4
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Criterion for two-component model --> narrow [OIII] is resolved
    res_cond = (gfit_2comp['oiii4959'].stddev.value > rsig_oiii)&\
    (gfit_2comp['oiii5007'].stddev.value > rsig_oiii)
        
    ## Criterion for defaulting back to one-component model
    ## Sigma ([OIII]out) > 1000 km/s
    sig_oiii_out, _ = mfit.correct_for_rsigma(gfit_2comp['oiii5007'].mean.value, \
                                             gfit_2comp['oiii5007'].stddev.value, \
                                             rsig_oiii)
    
    ## If the amplitude ratio of (outflow/narrow) > 2
    ## default to one-component model
    amp_ratio = gfit_2comp['oiii5007_out'].amplitude.value/gfit_2comp['oiii5007'].amplitude.value
    
    default_cond = (sig_oiii_out > 1000)|(amp_ratio > 2)
    
    ## 5-sigma confidence of an extra component
    if ((p_val <= 3e-7)&(res_cond)&(~default_cond)):
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
        1) free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                sii_bestfit, rsig_sii)
        2) fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                 sii_bestfit, rsig_sii)
        3) fixed_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, 
                                  sii_bestfit, rsig_sii)
    """
    def free_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                              sii_bestfit, rsig_sii):    
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
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
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
                                                                      ivar_nii_ha, rsig_nii_ha, \
                                                                      sii_bestfit, rsig_sii, \
                                                                      broad_comp = False)

        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                     ivar_nii_ha, rsig_nii_ha, \
                                                                     sii_bestfit, rsig_sii, \
                                                                     priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2s)]
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 8-5
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
    
        ## Broad Ha width
        ha_b_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_b'].mean.value, \
                                          gfit_b['ha_b'].stddev.value, \
                                          rsig_nii_ha)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow Ha flux is zero, but broad Ha flux is not zero
        ## If broad Hb flux = 0, then also default to no broad fit
        ## If sigma (narrow Ha) < sigma (narrow [SII]), then also default to no broad fit
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
    
        ha_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_n'].mean.value, \
                                           gfit_b['ha_n'].stddev.value, \
                                           rsig_nii_ha)
        nii_sig, _ = mfit.correct_for_rsigma(gfit_b['nii6583'].mean.value, \
                                            gfit_b['nii6583'].stddev.value, \
                                            rsig_nii_ha)
        
        ## Default conditions based on velocity offset of broad Ha
        ## Velocity offset of broad Ha
        ha_b_offset = (gfit_b['ha_n'].mean.value - gfit_b['ha_b'].mean.value)*3e+5/6564.312
        ha_b_ratio = ha_b_offset/ha_b_sig
        
        off_cond = (ha_b_fwhm < 1000)&((ha_b_ratio > 0.8)|(ha_b_ratio < -0.8))
    
        ## Default conditions
        cond1 = ((ha_n_flux == 0)&(ha_b_flux != 0))
        cond2 = (ha_b_flux == 0)
        cond3 = (ha_sig < nii_sig)&(~np.isclose(ha_sig, nii_sig))
        
        default_cond = cond1|cond2|cond3|off_cond

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
            
        return (nii_ha_bestfit, n_dof, psel)
    
####################################################################################################

    def fixed_ha_one_component(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                               sii_bestfit, rsig_sii):
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
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
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
                                                                 ivar_nii_ha, rsig_nii_ha, \
                                                                 sii_bestfit, rsig_sii, \
                                                                 broad_comp = False)
        
        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                ivar_nii_ha, rsig_nii_ha, \
                                                                sii_bestfit, rsig_sii, \
                                                                priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2s)]

        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 7-4
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Ha width
        ha_b_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_b'].mean.value, \
                                          gfit_b['ha_b'].stddev.value, \
                                          rsig_nii_ha)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow Ha flux is zero, but broad Ha flux is not zero
        ## If broad Hb flux = 0, then also default to no broad fit
        ## If sigma (narrow Ha) < sigma (narrow [SII]), then also default to no broad fit
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
    
        ha_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_n'].mean.value, \
                                           gfit_b['ha_n'].stddev.value, \
                                           rsig_nii_ha)
        nii_sig, _ = mfit.correct_for_rsigma(gfit_b['nii6583'].mean.value, \
                                            gfit_b['nii6583'].stddev.value, \
                                            rsig_nii_ha)
        ## Default conditions
        cond1 = ((ha_n_flux == 0)&(ha_b_flux != 0))
        cond2 = (ha_b_flux == 0)
        cond3 = (ha_sig < nii_sig)&(~np.isclose(ha_sig, nii_sig))
        
        ## Default conditions based on velocity offset of broad Ha
        ## Velocity offset of broad Ha
        ha_b_offset = (gfit_b['ha_n'].mean.value - gfit_b['ha_b'].mean.value)*3e+5/6564.312
        ha_b_ratio = ha_b_offset/ha_b_sig
        
        off_cond = (ha_b_fwhm < 1000)&((ha_b_ratio > 0.8)|(ha_b_ratio < -0.8))
        
        default_cond = cond1|cond2|cond3|off_cond

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

        return (nii_ha_bestfit, n_dof, psel)
    
####################################################################################################
    
    def fixed_ha_two_components(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                                sii_bestfit, rsig_sii):
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
            
        rsig_nii_ha : float
            Median resolution element in the [NII]+Ha region
            
        sii_bestfit : Astropy model
            Best fit model for the [SII] emission-lines.
            
        rsig_sii : float
            Median resolution element in the [SII] region.
            
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
                                                                  ivar_nii_ha, rsig_nii_ha, \
                                                                  sii_bestfit, rsig_sii, \
                                                                  broad_comp = False)

        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                 ivar_nii_ha, rsig_nii_ha, \
                                                                 sii_bestfit, rsig_sii, \
                                                                 priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_nii_ha, gfit(lam_nii_ha), ivar_nii_ha)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2s)]

        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)

        ## Statistical check for a broad component
        df = 9-6
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Ha width
        ha_b_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_b'].mean.value, \
                                             gfit_b['ha_b'].stddev.value, \
                                             rsig_nii_ha)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## If narrow/outflow Ha flux is zero, but broad Ha flux is not zero
        ## If broad Hb flux = 0, then also default to no broad fit
        ## If sigma (narrow Ha) < sigma (narrow [NII]), then also default to no broad fit
        ## If sigma (outflow Ha) < sigma (outflow [NII]), then also default to no broad fit
        ## Default to no broad fit
        ha_b_flux = mfit.compute_emline_flux(gfit_b['ha_b'].amplitude.value, \
                                            gfit_b['ha_b'].stddev.value)
        ha_n_flux = mfit.compute_emline_flux(gfit_b['ha_n'].amplitude.value, \
                                            gfit_b['ha_n'].stddev.value)
        ha_out_flux = mfit.compute_emline_flux(gfit_b['ha_out'].amplitude.value, \
                                              gfit_b['ha_out'].stddev.value)
        
        ha_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_n'].mean.value, \
                                           gfit_b['ha_n'].stddev.value, \
                                           rsig_nii_ha)
        ha_out_sig, _ = mfit.correct_for_rsigma(gfit_b['ha_out'].mean.value, \
                                               gfit_b['ha_out'].stddev.value, \
                                               rsig_nii_ha)
        nii_sig, _ = mfit.correct_for_rsigma(gfit_b['nii6583'].mean.value, \
                                            gfit_b['nii6583'].stddev.value, \
                                            rsig_nii_ha)
        nii_out_sig, _ = mfit.correct_for_rsigma(gfit_b['nii6583_out'].mean.value, \
                                                gfit_b['nii6583_out'].stddev.value, \
                                                rsig_nii_ha)
        
        ## Default conditions
        cond1 = (((ha_n_flux == 0)|(ha_out_flux == 0))&(ha_b_flux != 0))
        cond2 = (ha_b_flux == 0)
        cond3 = (ha_sig < nii_sig)&(~np.isclose(ha_sig, nii_sig))
        cond4 = (ha_out_sig < nii_out_sig)&(~np.isclose(ha_out_sig, nii_out_sig))
        
        ## Default conditions based on velocity offset of broad Ha
        ## Velocity offset of broad Ha
        ha_b_offset = (gfit_b['ha_n'].mean.value - gfit_b['ha_b'].mean.value)*3e+5/6564.312
        ha_b_ratio = ha_b_offset/ha_b_sig
        
        off_cond = (ha_b_fwhm < 1000)&((ha_b_ratio > 0.8)|(ha_b_ratio < -0.8))
        
        default_cond = cond1|cond2|cond3|cond4|off_cond

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

        return (nii_ha_bestfit, n_dof, psel)
    
####################################################################################################
####################################################################################################

def find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, rsig_nii_ha, \
                         sii_bestfit, rsig_sii):
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
        
    rsig_nii_ha : float
        Median resolution element in the [NII]+Ha region.

    sii_bestfit : Astropy model
        Best fit model for the [SII] emission-lines.
        
    rsig_sii : float
        Median resolution element in the [SII] region.
        
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
        nii_ha_bestfit, n_dof, psel = nii_ha_fit.free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                       ivar_nii_ha, rsig_nii_ha, \
                                                                       sii_bestfit, rsig_sii)
        
        ## How does Ha width compare to [SII] width?
        sig_sii, _ = mfit.correct_for_rsigma(sii_bestfit['sii6716'].mean.value, \
                                             sii_bestfit['sii6716'].stddev.value, \
                                             rsig_sii)
        
        sig_ha, _ = mfit.correct_for_rsigma(nii_ha_bestfit['ha_n'].mean.value, \
                                           nii_ha_bestfit['ha_n'].stddev.value, \
                                           rsig_nii_ha)
        
        per_diff = (sig_ha - sig_sii)*100/sig_sii
                
        if ((per_diff < 0)|(per_diff >= 30)|(nii_ha_bestfit['ha_n'].amplitude.value == 0)):
            ## If sigma (Ha) is less than sigma ([SII]) or increases more then 30% of sigma ([SII])
            ## Use fixed version
            nii_ha_bestfit, n_dof, psel = nii_ha_fit.fixed_ha_one_component(lam_nii_ha, \
                                                                            flam_nii_ha, \
                                                                            ivar_nii_ha, \
                                                                            rsig_nii_ha, \
                                                                            sii_bestfit, \
                                                                            rsig_sii)    
    else:
        nii_ha_bestfit, n_dof, psel = nii_ha_fit.fixed_ha_two_components(lam_nii_ha, \
                                                                         flam_nii_ha, \
                                                                         ivar_nii_ha, \
                                                                         rsig_nii_ha, \
                                                                         sii_bestfit, \
                                                                         rsig_sii)
    return (nii_ha_bestfit, n_dof, psel)        

####################################################################################################
####################################################################################################

def find_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                     nii_ha_bestfit, rsig_nii_ha):
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
        
    rsig_hb : float
        Median resolution element in the Hb region
        
    nii_ha_bestfit : Astropy model
        Best fit model for the [NII]+Ha region
        
    rsig_nii_ha : float
        Median resolution element in the [NII]+Ha region.
        
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
                                                         ivar_hb, rsig_hb, \
                                                         nii_ha_bestfit, rsig_nii_ha)
        
        if ('hb_b' not in hb_bestfit.submodel_names):
            n_dof = 2
        else:
            n_dof = 3
            
    else:
        ## Two components fit
        hb_bestfit = fl.fit_hb_line.fit_hb_two_components(lam_hb, flam_hb, \
                                                          ivar_hb, rsig_hb, \
                                                          nii_ha_bestfit, rsig_nii_ha)
        
        if ('hb_b' not in hb_bestfit.submodel_names):
            n_dof = 3
        else:
            n_dof = 4
            
    ## Returns the bestfit
    return (hb_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_nii_ha_sii_best_fit(lam_nii_ha_sii, flam_nii_ha_sii, ivar_nii_ha_sii, \
                            rsig_nii_ha_sii):
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
        
    rsig_nii_ha_sii : float
        Median resolution element in the [NII]+Ha+[SII]region.
        
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
                                                               ivar_nii_ha_sii, \
                                                               rsig_nii_ha_sii, \
                                                               priors = p)
        chi2_fit = mfit.calculate_chi2(flam_nii_ha_sii, gfit(lam_nii_ha_sii), ivar_nii_ha_sii)
        gfits.append(gfit)
        chi2s.append(chi2_fit)
        
    ## Select the fit with the minimum chi2
    nii_ha_sii_bestfit = gfits[np.argmin(chi2s)]
    n_dof = 10
    
    ## Select the prior that leads to the bestfit
    psel = priors_list[np.argmin(chi2s)]
    
    return (nii_ha_sii_bestfit, n_dof, psel)

####################################################################################################
####################################################################################################

def find_hb_oiii_best_fit(lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii, rsig_hb_oiii, \
                         nii_ha_sii_bestfit, rsig_nii_ha_sii):
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
        
    rsig_hb_oiii : float
        Median resolution element in the Hb+[OIII] region.

    nii_ha_sii_bestfit : Astropy model
        Best fit model for the [NII]+Ha+[SII] emission-lines.
        
    rsig_nii_ha_sii : float
        Median resolution element in the [NII]+Ha+[SII] region.

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
                                                                    rsig_hb_oiii, \
                                                                    nii_ha_sii_bestfit, \
                                                                    rsig_nii_ha_sii)
    
    ## Two component fit
    gfit_2comp = fl.fit_extreme_broadline_sources.fit_hb_oiii_2comp(lam_hb_oiii, \
                                                                    flam_hb_oiii, \
                                                                    ivar_hb_oiii, \
                                                                    rsig_hb_oiii, \
                                                                    nii_ha_sii_bestfit, \
                                                                    rsig_nii_ha_sii)
    
    ## Chi2 values for both the fits
    chi2_1comp = mfit.calculate_chi2(flam_hb_oiii, gfit_1comp(lam_hb_oiii), ivar_hb_oiii)
    chi2_2comp = mfit.calculate_chi2(flam_hb_oiii, gfit_2comp(lam_hb_oiii), ivar_hb_oiii)
    
    ## Statistical check for the second component
    df = 9-6 
    del_chi2 = chi2_1comp - chi2_2comp
    p_val = chi2.sf(del_chi2, df)
    
    ## Criterion for defaulting back to one-component model
    ## Sigma ([OIII]out) > 1000 km/s
    sig_oiii_out, _ = mfit.correct_for_rsigma(gfit_2comp['oiii5007_out'].mean.value, \
                                          gfit_2comp['oiii5007_out'].stddev.value, \
                                          rsig_hb_oiii)
    ## If the amplitude ratio of (outflow/narrow) > 1.5
    ## default to one-component model
    amp_ratio = gfit_2comp['oiii5007_out'].amplitude.value/gfit_2comp['oiii5007'].amplitude.value
    
    default_cond = (sig_oiii_out > 1000)|(amp_ratio > 1.5)
    
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

class highz_hb_fit:
    """
    This class contains functions related to High-redshift Hb Fit (Fixed Version)
        1) fixed_hb_one_component(lam_hb, flam_hb, ivar_hb, rsig_hb, oiii_bestfit, rsig_oiii)
        2) fixed_hb_two_components(lam_hb, flam_hb, ivar_hb, rsig_hb, oiii_bestfit, rsig_oiii)
    """
    def fixed_hb_one_component(lam_hb, flam_hb, ivar_hb, rsig_hb, oiii_bestfit, rsig_oiii):
        """
        Find bestfit for the sigma of Hb is tied to [OIII].
        This is for high-redshift galaxies when [SII] and Ha are not available.

        The code fits both broad and non-broad component fits and picks the best version.
        The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
        5-sigma confidence for an extra component statistically.

        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region

        flam_hb : numpy array
            Flux array of the spectra in the Hb region

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        rsig_hb : float
            Median resolution element in the Hb region.
            
        oiii_bestfit: Astropy model
            Bestfit for [OIII] emission-lines
            
        rsig_oiii : float
            Median Resolution element in the [OIII] region

        Returns
        -------
        hb_bestfit : Astropy model
            Best-fit "without-broad" or "with-broad" component

        n_dof : int
            Number of degrees of freedom

        psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
        """

        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_highz_hb_line.fit_fixed_hb_one_component(lam_hb, flam_hb, ivar_hb, \
                                                                    rsig_hb, oiii_bestfit, \
                                                                    rsig_oiii, broad_comp = False)
        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []

        for p in priors_list:
            gfit = fl.fit_highz_hb_line.fit_fixed_hb_one_component(lam_hb, flam_hb, ivar_hb, \
                                                                   rsig_hb, oiii_bestfit, rsig_oiii, \
                                                                   priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_hb, gfit(lam_hb), ivar_hb)
            gfits.append(gfit)
            chi2s.append(chi2_fit)

        ## Select the broad-component fit with the minimum chi2s
        gfit_b = gfits[np.argmin(chi2s)]

        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2s)]

        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)

        ## Statistical check for the broad component
        df = 5-2
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)

        ## Broad Hb width
        hb_b_sig, _ = mfit.correct_for_rsigma(gfit_b['hb_b'].mean.value, \
                                              gfit_b['hb_b'].stddev.value, \
                                              rsig_hb)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)

        ## If narrow Hb flux is zero, but broad Hb is not zero
        ## If broad Hb flux = 0
        ## Default to no broad fit in these cases
        hb_n_flux = mfit.compute_emline_flux(gfit_b['hb_n'].amplitude.value, \
                                            gfit_b['hb_n'].stddev.value)
        hb_b_flux = mfit.compute_emline_flux(gfit_b['hb_b'].amplitude.value, \
                                            gfit_b['hb_b'].stddev.value)

        default_cond = ((hb_n_flux == 0)&(hb_b_flux != 0))|(hb_b_flux == 0)

        ## Conditions for selecting a broad component
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)&(~default_cond)):
            hb_bestfit = gfit_b
            n_dof = 5
            psel = psel
        else:
            hb_bestfit = gfit_no_b
            n_dof = 2
            psel = []

        return (hb_bestfit, n_dof, psel)
    
####################################################################################################

    def fixed_hb_two_components(lam_hb, flam_hb, ivar_hb, rsig_hb, oiii_bestfit, rsig_oiii):
        """
        Find bestfit for the sigma of Hb and outflow component is tied to [OIII] components.
        This is for high-redshift galaxies when [SII] and Ha are not available.

        The code fits both broad and non-broad component fits and picks the best version.
        The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
        5-sigma confidence for an extra component statistically.

        Parameters
        ----------
        lam_hb : numpy array
            Wavelength array of the Hb region

        flam_hb : numpy array
            Flux array of the spectra in the Hb region

        ivar_hb : numpy array
            Inverse variance array of the spectra in the Hb region.

        rsig_hb : float
            Median resolution element in the Hb region.
            
        oiii_bestfit: Astropy model
            Bestfit for [OIII] emission-lines
            
        rsig_oiii : float
            Median Resolution element in the [OIII] region

        Returns
        -------
        hb_bestfit : Astropy model
            Best-fit "without-broad" or "with-broad" component

        n_dof : int
            Number of degrees of freedom

        psel : list
            Selected prior if the bestmodel fit has a broad component
            psel = [] if there is no broad component
        """
        
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_highz_hb_line.fit_fixed_hb_two_components(lam_hb, flam_hb, ivar_hb, \
                                                                     rsig_hb, oiii_bestfit, \
                                                                     rsig_oiii, broad_comp = False)
        
        ## With broad component
        ## Test with different priors and select the one with the least chi2
        priors_list = [[4,5], [3,6], [5,8]]
        gfits = []
        chi2s = []
        
        for p in priors_list:
            gfit = fl.fit_highz_hb_line.fit_fixed_hb_two_components(lam_hb, flam_hb, ivar_hb, \
                                                                   rsig_hb, oiii_bestfit, rsig_oiii, \
                                                                   priors = p, broad_comp = True)
            chi2_fit = mfit.calculate_chi2(flam_hb, gfit(lam_hb), ivar_hb)
            gfits.append(gfit)
            chi2s.append(chi2_fit)
            
        ## Select the broad-component fit with the minimum chi2
        gfit_b = gfits[np.argmin(chi2s)]
        
        ## Select the prior that leads to the bestfit
        psel = priors_list[np.argmin(chi2s)]
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)
        
        ## Statistical check for a broad component
        df = 7-4
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Hb width
        hb_b_sig, _ = mfit.correct_for_rsigma(gfit_b['hb_b'].mean.value, \
                                             gfit_b['hb_b'].stddev.value, \
                                             rsig_hb)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)
        
        ## If narrow/outflow Hb flux is zero, but broad Hb is not zero
        ## If broad Hb flux = 0
        ## Default to no broad fit in these cases
        
        hb_b_flux = mfit.compute_emline_flux(gfit_b['hb_b'].amplitude.value, \
                                            gfit_b['hb_b'].stddev.value)
        hb_n_flux = mfit.compute_emline_flux(gfit_b['hb_n'].amplitude.value, \
                                            gfit_b['hb_n'].stddev.value)
        hb_out_flux = mfit.compute_emline_flux(gfit_b['hb_out'].amplitude.value, \
                                              gfit_b['hb_out'].stddev.value)
        
        ## Default conditions
        cond1 = (((hb_n_flux == 0)|(hb_out_flux == 0))&(hb_b_flux != 0))
        cond2 = (hb_b_flux == 0)
        
        default_cond = cond1|cond2
        
        ## Conditions for selecting a broad component
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)&(~default_cond)):
            hb_bestfit = gfit_b
            n_dof = 7
            psel = psel
        else:
            hb_bestfit = gfit_no_b
            n_dof = 4
            psel = []
            
        return (hb_bestfit, n_dof, psel)
    
####################################################################################################
####################################################################################################

def find_highz_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb, oiii_bestfit, rsig_oiii):
    """
    Find the best fit for the high-z Hb emission line.
    This is for the sources with z >= 0.45, where [SII] and Ha are not available.
    
    The code fits both broad and non-broad component fits and picks the best version.
    The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.

    Parameters
    ----------
    lam_hb : numpy array
        Wavelength array of the Hb region

    flam_hb : numpy array
        Flux array of the spectra in the Hb region

    ivar_hb : numpy array
        Inverse variance array of the spectra in the Hb region.

    rsig_hb : float
        Median resolution element in the Hb region.

    oiii_bestfit: Astropy model
        Bestfit for [OIII] emission-lines

    rsig_oiii : float
        Median Resolution element in the [OIII] region

    Returns
    -------
    hb_bestfit : Astropy model
        Best-fit "without-broad" or "with-broad" component

    n_dof : int
        Number of degrees of freedom

    psel : list
        Selected prior if the bestmodel fit has a broad component
        psel = [] if there is no broad component
    """
    
    ## Functions change depending on the number of components in [OIII]
    oiii_models = oiii_bestfit.submodel_names
    
    if (('oiii4959_out' not in oiii_models)&('oiii5007_out' not in oiii_models)):
        ## One-Component Fit
        hb_bestfit, n_dof, psel = highz_hb_fit.fixed_hb_one_component(lam_hb, flam_hb, ivar_hb, \
                                                                     rsig_hb, oiii_bestfit, \
                                                                     rsig_oiii)
    else:
        ## Two-Component Fit
        hb_bestfit, n_dof, psel = highz_hb_fit.fixed_hb_two_components(lam_hb, flam_hb, ivar_hb, \
                                                                      rsig_hb, oiii_bestfit, \
                                                                      rsig_oiii)
        
    return (hb_bestfit, n_dof, psel)
    
####################################################################################################
####################################################################################################

def find_nev_best_fit(lam_nev, flam_nev, ivar_nev, rsig_nev, sii_bestfit, rsig_sii):
    """
    Find the best fit for [NeV] emission lines.
    The code picks one or two-component fits depending on the number of components of the 
    [SII] emission line. The number of components is same as [SII].

    Parameters
    ----------
    lam_nev : numpy array
        Wavelength array of the [NeV] region where the fits need to be perfomed.

    flam_nev : numpy array
        Flux array for the spectra in the [NeV] region.

    ivar_nev : numpy array
        Inverse variance array of the spectra in the [NeV] region.

    rsig_nev : float
        Median resolution element in the [NeV] region.

    sii_bestfit: Astropy model
        Best fit model for the [SII] emission-lines.

    rsig_sii : float
        Median resolution element in the [SII] region.

    Returns
    -------
    nev_bestfit : Astropy model
        Best-fit model for the [NeV] emission lines.

    n_dof : int
        Number of degrees of freedom.
    """

    ## Functions change depending on the number of components in [SII]
    sii_models = sii_bestfit.submodel_names

    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## One-component model
        gfit = fl.fit_nev_lines.fit_one_component(lam_nev, flam_nev, ivar_nev, rsig_nev, \
                                                 sii_bestfit, rsig_sii)
        n_dof = 4
    else:
        ## Two-component model
        gfit = fl.fit_nev_lines.fit_two_components(lam_nev, flam_nev, ivar_nev, rsig_nev, \
                                                  sii_bestfit, rsig_sii)
        n_dof = 6


    return (gfit, n_dof)

####################################################################################################
####################################################################################################
    
    


# class highz_fit:
#     """
#     This class contains functions related to High-z Hb and [OIII] Fitting:
#         1) find_free_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb)
#         2) find_fixed_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb)
#     """
#     def find_free_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb):
#         """
#         Find bestfit for Hb when fit freely. This is for high-redshift galaxies 
#         when [SII] and Ha are not available.

#         The code fits both broad and non-broad component fits and picks the best version.
#         The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
#         5-sigma confidence for an extra component statistically.

#         Parameters
#         ----------
#         lam_hb : numpy array
#             Wavelength array of the Hb region

#         flam_hb : numpy array
#             Flux array of the spectra in the Hb region

#         ivar_hb : numpy array
#             Inverse variance array of the spectra in the Hb region.

#         rsig_hb : float
#             Median resolution element in the Hb region.

#         Returns
#         -------
#         hb_bestfit : Astropy model
#             Best-fit "without-broad" or "with-broad" component

#         n_dof : int
#             Number of degrees of freedom

#         psel : list
#             Selected prior if the bestmodel fit has a broad component
#             psel = [] if there is no broad component
#         """

#         ## Single component model
#         ## Without broad component
#         gfit_no_b = fl.fit_highz_hb_oiii_lines.fit_free_hb(lam_hb, flam_hb, ivar_hb, \
#                                                            rsig_hb, broad_comp = False)

#         ## With broad component
#         ## Test with different priors and select the one with the least chi2
#         priors_list = [[4,5], [3,6], [5,8]]
#         gfits = []
#         chi2s = []

#         for p in priors_list:
#             gfit = fl.fit_highz_hb_oiii_lines.fit_free_hb(lam_hb, flam_hb, ivar_hb, rsig_hb, \
#                                                           priors = p, broad_comp = True)
#             chi2_fit = mfit.calculate_chi2(flam_hb, gfit(lam_hb), ivar_hb)
#             gfits.append(gfit)
#             chi2s.append(chi2_fit)

#         ## Select the broad-component fit with the minimum chi2s
#         gfit_b = gfits[np.argmin(chi2s)]

#         ## Select the prior that leads to the bestfit
#         psel = priors_list[np.argmin(chi2s)]

#         ## Chi2 values for both the fits
#         chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
#         chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)

#         ## Statistical check for the broad component
#         df = 6-3
#         del_chi2 = chi2_no_b - chi2_b
#         p_val = chi2.sf(del_chi2, df)

#         ## Broad Ha width
#         hb_b_sig, _ = mfit.correct_for_rsigma(gfit_b['hb_b'].mean.value, \
#                                               gfit_b['hb_b'].stddev.value, \
#                                               rsig_hb)
#         hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)

#         ## If narrow Hb flux is zero, but broad Hb is not zero
#         ## If broad Hb flux = 0
#         ## Default to no broad fit in these cases
#         hb_n_flux = mfit.compute_emline_flux(gfit_b['hb_n'].amplitude.value, \
#                                             gfit_b['hb_n'].stddev.value)
#         hb_b_flux = mfit.compute_emline_flux(gfit_b['hb_b'].amplitude.value, \
#                                             gfit_b['hb_b'].stddev.value)

#         default_cond = ((hb_n_flux == 0)&(hb_b_flux != 0))|(hb_b_flux == 0)

#         ## Conditions for selecting a broad component
#         ## 5-sigma confidence of an extra component is satisfied
#         ## Broad component FWHM > 300 km/s
#         if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)&(~default_cond)):
#             hb_bestfit = gfit_b
#             n_dof = 6
#             psel = psel
#         else:
#             hb_bestfit = gfit_no_b
#             n_dof = 3
#             psel = []

#         return (hb_bestfit, n_dof, psel)
    


    
    
    
    

# ####################################################################################################
# ####################################################################################################
