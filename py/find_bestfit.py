"""
This script consists of functions for fitting emission-lines. 
The different functions are divided into different classes for different emission lines.

Author : Ragadeepika Pucha
Version : 2024, February 2
"""

###################################################################################################

import numpy as np

from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D, Polynomial1D

import fit_utils
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
    
    ## 5-sigma confidence of an extra component
    if (p_val <= 3e-7):
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
        
        sii_bestfit = gfit_2comp
        n_dof = 8
    else:
        sii_bestfit = gfit_1comp
        n_dof = 5
        
    return (sii_bestfit, n_dof)

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
    
    ## 5-sigma confidence of an extra component
    if (p_val <= 3e-7):
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
            gfit_2comp = cont + gfit_oiii4959 + gfit_oiii5007 + gfit_oiii4959_out + gfit_oiii5007_out
        
        oiii_bestfit = gfit_2comp
        n_dof = 7
    else:
        oiii_bestfit = gfit_1comp
        n_dof = 4
        
    return (oiii_bestfit, n_dof)
    
####################################################################################################
####################################################################################################

def find_hb_free_best_fit(lam_hb, flam_hb, ivar_hb, sii_bestfit):
    """
    Find the best fit for Hb emission-line, allowing width of Hb to vary freely.
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of Hb is same as [SII].
    
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
        
    Returns
    -------
    hb_bestfit : Astropy model
        Best-fit model for Hb emission line
        
    n_dof : int
        Number of degrees of freedom
    """
    
    ## Functions change depending on the number of components in [SII]
    
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_hb_line.fit_free_one_component(lam_hb, flam_hb, ivar_hb, \
                                                      sii_bestfit, broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_hb_line.fit_free_one_component(lam_hb, flam_hb, ivar_hb, \
                                                      sii_bestfit, broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)
        
        ## Statistical check for a broad component
        df = 7-4
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Hb width
        hb_b_sig = mfit.lamspace_to_velspace(gfit_b['hb_b'].stddev.value, \
                                            gfit_b['hb_b'].mean.value)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)):
            hb_bestfit = gfit_b
            n_dof = 7
        else:
            hb_bestfit = gfit_no_b
            n_dof = 4
            
    else:
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_hb_line.fit_free_two_components(lam_hb, flam_hb, ivar_hb, \
                                                          sii_bestfit, broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_hb_line.fit_free_two_components(lam_hb, flam_hb, ivar_hb, \
                                                        sii_bestfit, broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)
        
        ## Statistical check for a broad component
        df = 10-7
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Hb width
        hb_b_sig = mfit.lamspace_to_velspace(gfit_b['hb_b'].stddev.value, \
                                            gfit_b['hb_b'].mean.value)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)):
            hb_bestfit = gfit_b
            n_dof = 10
        else:
            hb_bestfit = gfit_no_b
            n_dof = 7
            
    return (hb_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_hb_fixed_best_fit(lam_hb, flam_hb, ivar_hb, sii_bestfit):
    """
    Find the best fit for Hb emission-line, fixing Hb width to [SII].
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value for chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of Hb is same as [SII].
    
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
        
    Returns
    -------
    hb_bestfit : Astropy model
        Best-fit model for Hb emission line
        
    n_dof : int
        Number of degrees of freedom
    """
    
    ## Functions change depending on the number of components in [SII]
    
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_hb_line.fit_fixed_one_component(lam_hb, flam_hb, ivar_hb, \
                                                      sii_bestfit, broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_hb_line.fit_fixed_one_component(lam_hb, flam_hb, ivar_hb, \
                                                      sii_bestfit, broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)
        
        ## Statistical check for a broad component
        df = 6-3
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Hb width
        hb_b_sig = mfit.lamspace_to_velspace(gfit_b['hb_b'].stddev.value, \
                                            gfit_b['hb_b'].mean.value)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)):
            hb_bestfit = gfit_b
            n_dof = 6
        else:
            hb_bestfit = gfit_no_b
            n_dof = 3
            
    else:
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_hb_line.fit_fixed_two_components(lam_hb, flam_hb, ivar_hb, \
                                                          sii_bestfit, broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_hb_line.fit_fixed_two_components(lam_hb, flam_hb, ivar_hb, \
                                                        sii_bestfit, broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_hb, gfit_no_b(lam_hb), ivar_hb)
        chi2_b = mfit.calculate_chi2(flam_hb, gfit_b(lam_hb), ivar_hb)
        
        ## Statistical check for a broad component
        df = 8-5
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Hb width
        hb_b_sig = mfit.lamspace_to_velspace(gfit_b['hb_b'].stddev.value, \
                                            gfit_b['hb_b'].mean.value)
        hb_b_fwhm = mfit.sigma_to_fwhm(hb_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(hb_b_fwhm >= 300)):
            hb_bestfit = gfit_b
            n_dof = 8
        else:
            hb_bestfit = gfit_no_b
            n_dof = 5
    
    return (hb_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_nii_free_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
    """
    Find the best fit for [NII]+Ha emission lines.
    [NII] is fixed to [SII] and Ha is allowed to vary freely.
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value of the chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of [NII] and Ha is same as [SII]
    
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
    """
    
    ## Functions change depending on the number of components in [SII]
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                     ivar_nii_ha, sii_bestfit, \
                                                                     broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                   ivar_nii_ha, sii_bestfit, \
                                                                   broad_comp = True)
        
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
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 9
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 6
            
    else:
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_free_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                       ivar_nii_ha, sii_bestfit, \
                                                                       broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_nii_free_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                    ivar_nii_ha, sii_bestfit, \
                                                                    broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)
        
        ## Statistical check for a broad component
        df = 14-11
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 14
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 11
            
    return (nii_ha_bestfit, n_dof)
        
####################################################################################################
####################################################################################################

def find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
    """
    Find the best fit for [NII]+Ha emission lines.
    Both [NII] and Ha are fixed to [SII].
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value of the chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of [NII] and Ha is same as [SII]
    
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
    """
    
    ## Functions change depending on the number of components in [SII]
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                 ivar_nii_ha, sii_bestfit, \
                                                                 broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                              ivar_nii_ha, sii_bestfit, \
                                                              broad_comp = True)
        
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
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 8
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 5
            
    else:
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                  ivar_nii_ha, sii_bestfit, \
                                                                  broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                               ivar_nii_ha, sii_bestfit, \
                                                               broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)
        
        ## Statistical check for a broad component
        df = 12-9
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 12
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 9
            
    return (nii_ha_bestfit, n_dof)

####################################################################################################
####################################################################################################

def find_free_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, ivar_nii_ha, sii_bestfit):
    """
    Find the best fit for [NII]+Ha emission lines.
    Both [NII] and Ha are allowed to vary.
    The code fits both without and with broad component fits and picks the best version.
    The broad component fit is picked if the p-value of the chi2 distribution is < 3e-7 -->
    5-sigma confidence for an extra component statistically.
    The number of components of [NII] and Ha is same as [SII]
    
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
    """
    
    ## Functions change depending on the number of components in [SII]
    sii_models = sii_bestfit.submodel_names
    
    if (('sii6716_out' not in sii_models)&('sii6731_out' not in sii_models)):
        ## Single component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_free_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                      ivar_nii_ha, sii_bestfit, \
                                                                      broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_free_nii_ha_one_component(lam_nii_ha, flam_nii_ha, \
                                                                   ivar_nii_ha, sii_bestfit, \
                                                                   broad_comp = True)
        
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
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 9
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 6
            
    else:
        ## Two component model
        ## Without broad component
        gfit_no_b = fl.fit_nii_ha_lines.fit_free_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                       ivar_nii_ha, sii_bestfit, \
                                                                       broad_comp = False)
        
        ## With broad component
        gfit_b = fl.fit_nii_ha_lines.fit_free_nii_ha_two_components(lam_nii_ha, flam_nii_ha, \
                                                                    ivar_nii_ha, sii_bestfit, \
                                                                    broad_comp = True)
        
        ## Chi2 values for both the fits
        chi2_no_b = mfit.calculate_chi2(flam_nii_ha, gfit_no_b(lam_nii_ha), ivar_nii_ha)
        chi2_b = mfit.calculate_chi2(flam_nii_ha, gfit_b(lam_nii_ha), ivar_nii_ha)
        
        ## Statistical check for a broad component
        df = 14-11
        del_chi2 = chi2_no_b - chi2_b
        p_val = chi2.sf(del_chi2, df)
        
        ## Broad Ha width
        ha_b_sig = mfit.lamspace_to_velspace(gfit_b['ha_b'].stddev.value, \
                                            gfit_b['ha_b'].mean.value)
        ha_b_fwhm = mfit.sigma_to_fwhm(ha_b_sig)
        
        ## Conditions for selecting a broad component:
        ## 5-sigma confidence of an extra component is satisfied
        ## Broad component FWHM > 300 km/s
        if ((p_val <= 3e-7)&(ha_b_fwhm >= 300)):
            nii_ha_bestfit = gfit_b
            n_dof = 14
        else:
            nii_ha_bestfit = gfit_no_b
            n_dof = 11
            
    return (nii_ha_bestfit, n_dof)

####################################################################################################
####################################################################################################