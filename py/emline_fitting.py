"""
This script consists of functions related to fitting the emission line spectra, 
and plotting the models and residuals.

Author : Ragadeepika Pucha
Version : 2024, March 5
"""

####################################################################################################

import numpy as np

from astropy.table import Table, vstack
from astropy.modeling.models import Gaussian1D, Const1D

import spec_utils, plot_utils
import fit_lines
import measure_fits as mfit
import emline_params as emp
import find_bestfit

import matplotlib.pyplot as plt
import random

####################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':18,
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

#################################################################################################### 

def fit_spectra(specprod, survey, program, healpix, targetid, z):
    """
    Function to fit Hb, [OIII], [NII], Ha, and [SII] emission-lines for a given target.
    The code separates "normal" vs "extreme" broadline sources and fits them accordingly.
    
    Parameters
    ----------
    specprod : str
        Spectral Production Pipeline name fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : int
        Healpix number of the target
        
    z : float
        Redshift of the target
        
    Returns
    -------
    t_params : Astropy Table
        Table of parameters from the fits
    """
    
    ## Rest-frame emission-line spectra
    coadd_spec, lam_rest, \
    flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                         healpix, targetid, z, rest_frame = True)
    
    ## Fit [SII] lines first
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest, em_line = 'sii')
    sii_fit, _ = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
    sii_diff, sii_frac = mfit.measure_sii_difference(lam_sii, flam_sii)
    
    ## Conditions for separating extreme broadline sources
    sii_frac_cond = (np.abs(sii_frac) >= 5.0)
    sii_diff_cond = (sii_diff >= 0.5)
        
    if ('sii6716_out' in sii_fit.submodel_names):
        sii_out_sig = mfit.lamspace_to_velspace(sii_fit['sii6716_out'].stddev.value, \
                                               sii_fit['sii6716_out'].mean.value)
    else:
        sii_out_sig = 0.0
        
    sii_out_cond = (sii_out_sig >= 1000)
    
    ext_cond = ((sii_frac_cond)&(sii_diff_cond))|(sii_out_cond)
    
    if ext_cond:
        ## Fit using extreme-line fitting code
        hb_params, oiii_params, \
        nii_ha_params, sii_params = fit_spectra_extreme(coadd_spec, lam_rest, flam_rest, ivar_rest)
    else:
        ## Fit using normal fitting code
        hb_params, oiii_params, \
        nii_ha_params, sii_params = fit_spectra_normal(coadd_spec, lam_rest, flam_rest, ivar_rest)
        
    ######## Combine everything
    tgt = {}
    tgt['targetid'] = [targetid]
    tgt['specprod'] = [specprod]
    tgt['survey'] = [survey]
    tgt['program'] = [program]
    tgt['healpix'] = [healpix]
    tgt['z'] = [z]
    
    t_params = Table(tgt|hb_params|oiii_params|nii_ha_params|sii_params)
    for col in t_params.colnames:
        t_params.rename_column(col, col.upper())
    
    return (t_params)

####################################################################################################
####################################################################################################

def fit_spectra_normal(coadd_spec, lam_rest, flam_rest, ivar_rest):
    """
    Function to fit the given spectra with four different fitting windows - 
    Hb, [OIII], [NII]+Ha, and [SII].
    
    Parameters
    ----------
    coadd_spec : obj
        Coadded Spectra object (coadded across the cameras) of the given target
        
    lam_rest : numpy array
        Rest-frame wavelength array of the spectra
        
    flam_rest : numpy array
        Rest-frame flux array of the spectra
        
    ivar_rest : numpy array
        Rest-frame inverse variance of the spectra
        
    Returns
    -------
    hb_params : dict
        List of parameters related to Hb Fitting
        
    oiii_params : dict
        List of parameters related to [OIII] Fitting
        
    nii_ha_params : dict
        List of parameters related to [NII]+Ha Fitting
        
    sii_params : dict
        List of parameters related to [SII] Fitting
    """
    
    ## Fitting windows for the different emission-lines
    lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                         ivar_rest, em_line = 'hb')

    lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                               ivar_rest, em_line = 'oiii')
    lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                     ivar_rest, em_line = 'nii_ha')
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                            ivar_rest, em_line = 'sii')
    
    ## Fitting routines
    ## Bestfit for [SII]
    gfit_sii, ndof_sii = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
    ## Bestfit for [OIII]
    gfit_oiii, ndof_oiii = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)
    ## Bestfit for [NII]+Ha
    gfit_nii_ha, ndof_nii_ha = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
                                                                ivar_nii_ha, gfit_sii)
    ## Bestfit for Hb
    gfit_hb, ndof_hb = find_bestfit.find_hb_best_fit(lam_hb, flam_hb, ivar_hb, \
                                                    gfit_nii_ha)
    ## Compute reduced chi2:
    rchi2_sii = mfit.calculate_chi2(flam_sii, gfit_sii(lam_sii), ivar_sii, \
                                    ndof_sii, reduced_chi2 = True)
    rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                     ndof_oiii, reduced_chi2 = True)
    rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                   ndof_hb, reduced_chi2 = True)
    rchi2_nii_ha = mfit.calculate_chi2(flam_nii_ha, gfit_nii_ha(lam_nii_ha), \
                                           ivar_nii_ha, ndof_nii_ha, reduced_chi2 = True)
    
    ## Parameters for the fit
    hb_models = ['hb_n', 'hb_out', 'hb_b']
    oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
    nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
    sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']
    
    hb_params = emp.get_parameters(gfit_hb, hb_models)
    oiii_params = emp.get_parameters(gfit_oiii, oiii_models)
    nii_ha_params = emp.get_parameters(gfit_nii_ha, nii_ha_models)
    sii_params = emp.get_parameters(gfit_sii, sii_models)
    
    ## Continuum
    hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
    oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
    nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha['nii_ha_cont'].amplitude.value]
    sii_params['sii_continuum'] = [gfit_sii['sii_cont'].amplitude.value]
    
    ## NOISE
    hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
    oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'oiii')
    nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'nii_ha')
    sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'sii')
    
    hb_params['hb_noise'] = [hb_noise]
    oiii_params['oiii_noise'] = [oiii_noise]
    nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
    sii_params['sii_noise'] = [sii_noise]
    
    ## N(DOF)
    hb_params['hb_ndof'] = [ndof_hb]
    oiii_params['oiii_ndof'] = [ndof_oiii]
    oiii_params['hb_oiii_ndof'] = [int(0)]
    nii_ha_params['nii_ha_ndof'] = [ndof_nii_ha]
    sii_params['sii_ndof'] = [ndof_sii]
    sii_params['nii_ha_sii_ndof'] = [int(0)]

    ## Reduced chi2
    hb_params['hb_rchi2'] = [rchi2_hb]
    oiii_params['oiii_rchi2'] = [rchi2_oiii]
    oiii_params['hb_oiii_rchi2'] = [0.0]
    nii_ha_params['nii_ha_rchi2'] = [rchi2_nii_ha]
    sii_params['sii_rchi2'] = [rchi2_sii]
    sii_params['nii_ha_sii_rchi2'] = [0.0]
    
    return (hb_params, oiii_params, nii_ha_params, sii_params)

####################################################################################################
####################################################################################################

def fit_spectra_extreme(coadd_spec, lam_rest, flam_rest, ivar_rest):
    """
    Function to fit the given spectra with two different fitting windows - 
    Hb+[OIII] and [NII]+Ha+[SII] --> Fitting extreme broadline sources.
    
    Parameters
    ----------
    coadd_spec : obj
        Coadded Spectra object (coadded across the cameras) of the given target
        
    lam_rest : numpy array
        Rest-frame wavelength array of the spectra
        
    flam_rest : numpy array
        Rest-frame flux array of the spectra
        
    ivar_rest : numpy array
        Rest-frame inverse variance of the spectra
        
    Returns
    -------
    hb_params : dict
        List of parameters related to Hb Fitting
        
    oiii_params : dict
        List of parameters related to [OIII] Fitting
        
    nii_ha_params : dict
        List of parameters related to [NII]+Ha Fitting
        
    sii_params : dict
        List of parameters related to [SII] Fitting
    
    """
    
    ## Fitting windows for the different emission-line regions
    lam_nii_ha_sii, flam_nii_ha_sii, \
    ivar_nii_ha_sii = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest, 'nii_ha_sii')
    
    lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                       ivar_rest, 'hb_oiii')
    
    gfit_nii_ha_sii, ndof_nii_ha_sii = find_bestfit.find_nii_ha_sii_best_fit(lam_nii_ha_sii, \
                                                                            flam_nii_ha_sii, \
                                                                            ivar_nii_ha_sii)
    gfit_hb_oiii, ndof_hb_oiii = find_bestfit.find_hb_oiii_bestfit(lam_hb_oiii, flam_hb_oiii, \
                                                                  ivar_hb_oiii, gfit_nii_ha_sii)
    
    ## Compute reduced_chi2
    rchi2_nii_ha_sii = mfit.calculate_chi2(flam_nii_ha_sii, gfit_nii_ha_sii(lam_nii_ha_sii), ivar_nii_ha_sii, \
                                          ndof_nii_ha_sii, reduced_chi2 = True)
    rchi2_hb_oiii = mfit.calculate_chi2(flam_hb_oiii, gfit_hb_oiii(lam_hb_oiii), ivar_hb_oiii, \
                                       ndof_hb_oiii, reduced_chi2 = True)
    
    ## Parameters for the fit
    hb_models = ['hb_n', 'hb_out', 'hb_b']
    oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
    nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
    sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']
    
    hb_params = emp.get_parameters(gfit_hb_oiii, hb_models)
    oiii_params = emp.get_parameters(gfit_hb_oiii, oiii_models)
    nii_ha_params = emp.get_parameters(gfit_nii_ha_sii, nii_ha_models)
    sii_params = emp.get_parameters(gfit_nii_ha_sii, sii_models)
    
    ## Continuum
    hb_params['hb_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
    oiii_params['oiii_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
    nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]
    sii_params['sii_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]
    
    ## NOISE
    hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
    oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'oiii')
    nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'nii_ha')
    sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'sii')
    
    hb_params['hb_noise'] = [hb_noise]
    oiii_params['oiii_noise'] = [oiii_noise]
    nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
    sii_params['sii_noise'] = [sii_noise]
    
    ## N(DOF)
    hb_params['hb_ndof'] = [int(0)]
    oiii_params['oiii_ndof'] = [int(0)]
    oiii_params['hb_oiii_ndof'] = [ndof_hb_oiii]
    nii_ha_params['nii_ha_ndof'] = [int(0)]
    sii_params['sii_ndof'] = [int(0)]
    sii_params['nii_ha_sii_ndof'] = [ndof_nii_ha_sii]
    
    ## Reduced chi2 values
    hb_params['hb_rchi2'] = [0.0]
    oiii_params['oiii_rchi2'] = [0.0]
    oiii_params['hb_oiii_rchi2'] = [rchi2_hb_oiii]
    nii_ha_params['nii_ha_rchi2'] = [0.0]
    sii_params['sii_rchi2'] = [0.0]
    sii_params['nii_ha_sii_rchi2'] = [rchi2_nii_ha_sii]
        
    return (hb_params, oiii_params, nii_ha_params, sii_params)

####################################################################################################
####################################################################################################

def construct_normal_fits(t, index):
    """
    Construct fits of a particular source from the table of parameters. This is for normal fitting sources.
    
    Parameters 
    ----------
    t : Astropy Table
        Table of fit parameters
        
    index : int
        Index number of the source
        
    Returns
    -------
    fits : list
        List of [Hb, [OIII], [NII]+Ha, [SII]] fits
        
    """
    
    ######################################################################################
    ## Hbeta model
    hb_models = []
    
    ## Hb continuum model
    hb_cont = Const1D(amplitude = t['HB_CONTINUUM'].data[index], name = 'hb_cont')

    ## Gaussian model for the narrow component
    gfit_hb_n = Gaussian1D(amplitude = t['HB_N_AMPLITUDE'].data[index], \
                          mean = t['HB_N_MEAN'].data[index], \
                          stddev = t['HB_N_STD'].data[index], name = 'hb_n')

    gfit_hb = hb_cont + gfit_hb_n

    if (t['HB_OUT_MEAN'].data[index] != 0):
        ## Gaussian model for the outflow component if available
        gfit_hb_out = Gaussian1D(amplitude = t['HB_OUT_AMPLITUDE'].data[index], \
                                mean = t['HB_OUT_MEAN'].data[index], \
                                stddev = t['HB_OUT_STD'].data[index], name = 'hb_out')
        hb_models.append(gfit_hb_out)


    if (t['HB_B_MEAN'].data[index] != 0):
        ## Gaussian model for the broad component if available
        gfit_hb_b = Gaussian1D(amplitude = t['HB_B_AMPLITUDE'].data[index], \
                              mean = t['HB_B_MEAN'].data[index], \
                              stddev = t['HB_B_STD'].data[index], name = 'hb_b')

        hb_models.append(gfit_hb_b)

    ## Total Hb model
    for model in hb_models:
        gfit_hb = gfit_hb + model
        
    ######################################################################################
    ######################################################################################
    ## [OIII] model
    
    ## [OIII] continuum model
    oiii_cont = Const1D(amplitude = t['OIII_CONTINUUM'].data[index], name = 'oiii_cont')
    ## Gaussian model for [OIII]4959 narrow component
    gfit_oiii4959 = Gaussian1D(amplitude = t['OIII4959_AMPLITUDE'].data[index], \
                              mean = t['OIII4959_MEAN'].data[index], \
                               stddev = t['OIII4959_STD'].data[index], name = 'oiii4959')
    ## Gaussian model for [OIII]5007 narrow component
    gfit_oiii5007 = Gaussian1D(amplitude = t['OIII5007_AMPLITUDE'].data[index], \
                              mean = t['OIII5007_MEAN'].data[index], \
                              stddev = t['OIII5007_STD'].data[index], name = 'oiii5007')

    gfit_oiii = oiii_cont + gfit_oiii4959 + gfit_oiii5007

    oiii_models = []

    if (t['OIII5007_OUT_MEAN'].data[index] != 0):
        ## Gaussian model for [OIII]4959 outflow component if available
        gfit_oiii4959_out = Gaussian1D(amplitude = t['OIII4959_OUT_AMPLITUDE'].data[index], \
                                      mean = t['OIII4959_OUT_MEAN'].data[index], \
                                      stddev = t['OIII4959_OUT_STD'].data[index], \
                                       name = 'oiii4959_out')
        ## Gaussian model for [OIII]5007 outflow component if available
        gfit_oiii5007_out = Gaussian1D(amplitude = t['OIII5007_OUT_AMPLITUDE'].data[index], \
                                      mean = t['OIII5007_OUT_MEAN'].data[index], \
                                      stddev = t['OIII5007_OUT_STD'].data[index], \
                                       name = 'oiii5007_out')

        oiii_models.append(gfit_oiii4959_out)
        oiii_models.append(gfit_oiii5007_out)
    ## Total [OIII] model
    for model in oiii_models:
        gfit_oiii = gfit_oiii + model
    
    ######################################################################################
    ######################################################################################
    ## [NII] + Ha model
    
    ## [NII]+Ha continuum model
    nii_ha_cont = Const1D(amplitude = t['NII_HA_CONTINUUM'].data[index], name = 'nii_ha_cont')
    ## Gaussian model for [NII]6548 narrow component
    gfit_nii6548 = Gaussian1D(amplitude = t['NII6548_AMPLITUDE'].data[index], \
                             mean = t['NII6548_MEAN'].data[index], \
                             stddev = t['NII6548_STD'].data[index], name = 'nii6548')
    ## Gaussian model for [NII]6583 narrow component
    gfit_nii6583 = Gaussian1D(amplitude = t['NII6583_AMPLITUDE'].data[index], \
                             mean = t['NII6583_MEAN'].data[index], \
                             stddev = t['NII6583_STD'].data[index], name = 'nii6583')
    ## Gaussian model for Ha narrow component
    gfit_ha = Gaussian1D(amplitude = t['HA_N_AMPLITUDE'].data[index], \
                        mean = t['HA_N_MEAN'].data[index], \
                        stddev = t['HA_N_STD'].data[index], name = 'ha_n')

    gfit_nii_ha = nii_ha_cont + gfit_nii6548 + gfit_nii6583 + gfit_ha

    nii_ha_models = []

    if (t['NII6548_OUT_MEAN'].data[index] != 0):
        ## Gaussian model for [NII]6548 outflow component if available
        gfit_nii6548_out = Gaussian1D(amplitude = t['NII6548_OUT_AMPLITUDE'].data[index], \
                                      mean = t['NII6548_OUT_MEAN'].data[index], \
                                      stddev = t['NII6548_OUT_STD'].data[index], \
                                      name = 'nii6548_out')
        ## Gaussian model for [NII]6583 outflow component if available
        gfit_nii6583_out = Gaussian1D(amplitude = t['NII6583_OUT_AMPLITUDE'].data[index], \
                                      mean = t['NII6583_OUT_MEAN'].data[index], \
                                      stddev = t['NII6583_OUT_STD'].data[index], \
                                      name = 'nii6583_out')
        ## Gaussian model for Ha outflow component if available
        gfit_ha_out = Gaussian1D(amplitude = t['HA_OUT_AMPLITUDE'].data[index], \
                                mean = t['HA_OUT_MEAN'].data[index], \
                                stddev = t['HA_OUT_STD'].data[index], \
                                 name = 'ha_out')

        nii_ha_models.append(gfit_nii6548_out)
        nii_ha_models.append(gfit_nii6583_out)
        nii_ha_models.append(gfit_ha_out)

    if (t['HA_B_MEAN'].data[index] != 0):
        ## Gaussian model for Hb broad component if available
        gfit_ha_b = Gaussian1D(amplitude = t['HA_B_AMPLITUDE'].data[index], \
                              mean = t['HA_B_MEAN'].data[index], \
                              stddev = t['HA_B_STD'].data[index], \
                               name = 'ha_b')
        nii_ha_models.append(gfit_ha_b)
    
    ## Total [NII]+Ha model
    for model in nii_ha_models:
        gfit_nii_ha = gfit_nii_ha + model
        
    ######################################################################################
    ######################################################################################
    ## [SII] model
    
    ## [SII] continuum model
    sii_cont = Const1D(amplitude = t['SII_CONTINUUM'].data[index], name = 'sii_cont')
    ## Gaussian model for [SII]6716 narrow component
    gfit_sii6716 = Gaussian1D(amplitude = t['SII6716_AMPLITUDE'].data[index], \
                             mean = t['SII6716_MEAN'].data[index], \
                             stddev = t['SII6716_STD'].data[index], name = 'sii6716')
    ## Gaussian model for [SII]6731 narrow component
    gfit_sii6731 = Gaussian1D(amplitude = t['SII6731_AMPLITUDE'].data[index], \
                             mean = t['SII6731_MEAN'].data[index], \
                             stddev = t['SII6731_STD'].data[index], name = 'sii6731')

    gfit_sii = sii_cont + gfit_sii6716 + gfit_sii6731

    sii_models = []

    if (t['SII6716_OUT_MEAN'].data[index] != 0):
        ## Gaussian model for [SII]6716 outflow component if available
        gfit_sii6716_out = Gaussian1D(amplitude = t['SII6716_OUT_AMPLITUDE'].data[index], \
                                     mean = t['SII6716_OUT_MEAN'].data[index], \
                                     stddev = t['SII6716_OUT_STD'].data[index], \
                                      name = 'sii6716_out')
        ## Gaussian model for [SII]6731 outflow component if available
        gfit_sii6731_out = Gaussian1D(amplitude = t['SII6731_OUT_AMPLITUDE'].data[index], \
                                     mean = t['SII6731_OUT_MEAN'].data[index], \
                                     stddev = t['SII6731_OUT_STD'].data[index], \
                                      name = 'sii6731_out')

        sii_models.append(gfit_sii6716_out)
        sii_models.append(gfit_sii6731_out)
    
    ## Total [SII] model
    for model in sii_models:
        gfit_sii = gfit_sii + model

    fits_tab = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
    
    return (fits_tab)

####################################################################################################
####################################################################################################

def construct_extreme_fits(t, index):
    """
    Construct fits of a particular source from the table of parameters.
    This is for extreme broadline fitting sources.
    
    Parameters
    ----------
    t : Astropy Table 
        Table of fit parameters
        
    index : int
        Index number of the source
        
    Returns
    -------
    fits : list
        List of [Hb+[OIII] and [NII]+Ha+[SII]] fits
    """
    
    ######################################################################################
    ## Hbeta + [OIII] models
    hb_oiii_models = []
    
    ## Gaussian model for the narrow component
    gfit_hb_n = Gaussian1D(amplitude = t['HB_N_AMPLITUDE'].data[index], \
                          mean = t['HB_N_MEAN'].data[index], \
                          stddev = t['HB_N_STD'].data[index], name = 'hb_n')
    
    ## Gaussian model for the [OIII]4959,5007 narrow components
    gfit_oiii4959 = Gaussian1D(amplitude = t['OIII4959_AMPLITUDE'].data[index], \
                              mean = t['OIII4959_MEAN'].data[index], \
                              stddev = t['OIII4959_STD'].data[index], name = 'oiii4959')
    
    gfit_oiii5007 = Gaussian1D(amplitude = t['OIII5007_AMPLITUDE'].data[index], \
                              mean = t['OIII5007_MEAN'].data[index], \
                              stddev = t['OIII5007_STD'].data[index], name = 'oiii5007')
    
    ## Continuum
    hb_oiii_cont = Const1D(amplitude = t['HB_CONTINUUM'].data[index], \
                           name = 'hb_oiii_cont')
    
    
    gfit_hb_oiii = hb_oiii_cont + gfit_hb_n + gfit_oiii4959 + gfit_oiii5007
        
    if (t['HB_B_MEAN'].data[index] != 0):
        ## Gaussian model for the broad component if available
        gfit_hb_b = Gaussian1D(amplitude = t['HB_B_AMPLITUDE'].data[index], \
                              mean = t['HB_B_MEAN'].data[index], \
                              stddev = t['HB_B_STD'].data[index], name = 'hb_b')
        hb_oiii_models.append(gfit_hb_b)
        
    if (t['OIII4959_OUT_MEAN'].data[index] != 0):
        gfit_oiii4959_out = Gaussian1D(amplitude = t['OIII4959_OUT_AMPLITUDE'].data[index], \
                                      mean = t['OIII4959_OUT_MEAN'].data[index], \
                                      stddev = t['OIII4959_OUT_STD'].data[index], \
                                      name = 'oiii4959_out')
        gfit_oiii5007_out = Gaussian1D(amplitude = t['OIII5007_OUT_AMPLITUDE'].data[index], \
                                      mean = t['OIII5007_OUT_MEAN'].data[index], \
                                      stddev = t['OIII5007_OUT_STD'].data[index], \
                                      name = 'oiii5007_out')
        
        hb_oiii_models.append(gfit_oiii4959_out)
        hb_oiii_models.append(gfit_oiii5007_out)
        
    ## Total Hb+[OIII] model
    for model in hb_oiii_models:
        gfit_hb_oiii = gfit_hb_oiii + model
        
    ######################################################################################
    ######################################################################################
    ## [NII]+Ha+[SII] models
    
    ## Continuum
    nii_ha_sii_cont = Const1D(amplitude = t['SII_CONTINUUM'].data[index], \
                              name = 'nii_ha_sii_cont')
    
    ## [NII]6548,6583 models
    gfit_nii6548 = Gaussian1D(amplitude = t['NII6548_AMPLITUDE'].data[index], \
                              mean = t['NII6548_MEAN'].data[index], \
                              stddev = t['NII6548_STD'].data[index], \
                              name = 'nii6548')
    gfit_nii6583 = Gaussian1D(amplitude = t['NII6583_AMPLITUDE'].data[index], \
                             mean = t['NII6583_MEAN'].data[index], \
                             stddev = t['NII6583_STD'].data[index], \
                             name = 'nii6583')
    
    ## [SII]6716,6731 models
    gfit_sii6716 = Gaussian1D(amplitude = t['SII6716_AMPLITUDE'].data[index], \
                             mean = t['SII6716_MEAN'].data[index], \
                             stddev = t['SII6716_STD'].data[index], \
                             name = 'sii6716')
    gfit_sii6731 = Gaussian1D(amplitude = t['SII6731_AMPLITUDE'].data[index], \
                             mean = t['SII6731_MEAN'].data[index], \
                             stddev = t['SII6716_STD'].data[index], \
                             name = 'sii6731')
    
    ## Narrow and Broad Ha
    gfit_ha_n = Gaussian1D(amplitude = t['HA_N_AMPLITUDE'].data[index], \
                          mean = t['HA_N_MEAN'].data[index], \
                          stddev = t['HA_N_STD'].data[index], \
                          name = 'ha_n')
    gfit_ha = gfit_ha_n
    
    if (t['HA_B_MEAN'].data[index] != 0):
        ## Gaussian model for broad Ha, if available
        gfit_ha_b = Gaussian1D(amplitude = t['HA_B_AMPLITUDE'].data[index], \
                              mean = t['HA_B_MEAN'].data[index], \
                              stddev = t['HA_B_STD'].data[index], \
                              name = 'ha_b')
        gfit_ha = gfit_ha + gfit_ha_b
    
    ## Total [NII]+Ha+[SII] model
    gfit_nii_ha_sii = nii_ha_sii_cont + gfit_nii6548 + gfit_nii6583 + \
    gfit_ha + gfit_sii6716 + gfit_sii6731

    ## Fits list
    fits_tab = [gfit_hb_oiii, gfit_nii_ha_sii]
    
    return (fits_tab)

####################################################################################################
####################################################################################################

# def fit_emline_spectra(specprod, survey, program, healpix, targetid, z, fit_cont = True):
#     """
#     Fit [SII], Hb, [OIII], [NII]+Ha emission lines for a given emission line spectra.
#     The code runs 1000 iterations and returns the parameter values and errors.
    
#     Parameters
#     ----------
#     specprod : str
#         Spectral Production Pipeline name fuji|guadalupe|...
        
#     survey : str
#         Survey name for the spectra
        
#     program : str
#         Program name for the spectra
        
#     healpix : str
#         Healpix number of the target
        
#     targetid : int64
#         The unique TARGETID associated with the target
        
#     z : float
#         Redshift of the target

#     Returns
#     -------
#     t_params : astropy table
#         Table of output parameters for the fit, along with the errors.

#     """
    
#     ## Rest-frame emission-line spectra
#     lam_rest, flam_rest, ivar_rest, res_matrix = spec_utils.get_emline_spectra(specprod, survey, program, \
#                                                                   healpix, targetid, z, rest_frame = True, \
#                                                                   plot_continuum = False)
    
#     fits_orig, rchi2_orig, t_params = fit_spectra_iteration(lam_rest, flam_rest, ivar_rest, fit_cont = fit_cont)
    
#     ## Error spectra
#     err_rest = 1/np.sqrt(ivar_rest)
#     err_rest[~np.isfinite(err_rest)] = 0.0
    
#     ## List of tables of different iterations
#     tables = []
#     n_sii = fits_orig[-1].n_submodels
#     n_oiii = fits_orig[1].n_submodels

#     for kk in range(100):
#         noise_spec = random.gauss(0, err_rest)
#         to_add_spec = res_matrix.dot(noise_spec)
#         flam_new = flam_rest + to_add_spec
#         fits, rchi2s, t_params = fit_spectra_iteration(lam_rest, flam_new, \
#                                                        ivar_rest, n_sii, n_oiii, \
#                                                       fit_cont = fit_cont)
#         tables.append(t_params)

#     t_fits = vstack(tables)
#     #t_fits.write(f'iterations/iter-{targetid}-1000.fits')
    
#     ## Percent of iterations with broad Hb detected
#     per_hb = len(t_fits[t_fits['hb_b_flux'] > 0])*100/len(t_fits)
    
#     ## Percent of iterations with broad Ha detected
#     per_ha = len(t_fits[t_fits['ha_b_flux'] > 0])*100/len(t_fits)
    
#     per = {}
#     per['percent_hb_b'] = [per_hb]
#     per['percent_ha_b'] = [per_ha]
    
#     tgt = {}
#     tgt['targetid'] = [targetid]
#     tgt['specprod'] = [specprod]
#     tgt['survey'] = [survey]
#     tgt['program'] = [program]
#     tgt['healpix'] = [healpix]
#     tgt['z'] = [z]
    
#     hb_models = ['hb_n', 'hb_out', 'hb_b']
#     oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
#     nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
#     sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']
    
#     hb_params = emp.get_bestfit_parameters(t_fits, hb_models, emline = 'hb')
#     oiii_params = emp.get_bestfit_parameters(t_fits, oiii_models, emline = 'oiii')
#     nii_ha_params = emp.get_bestfit_parameters(t_fits, nii_ha_models, emline = 'nii_ha')
#     sii_params = emp.get_bestfit_parameters(t_fits, sii_models, emline = 'sii')
    
#     ####################################################################
#     ## Compute rchi2 -- first get number of degrees of freedom
    
#     sii_dof = t_fits['sii_dof'].data[0]
#     oiii_dof = t_fits['oiii_dof'].data[0]
    
#     if (np.all(t_fits['hb_b_flux'].data != 0)):
#         tsel = t_fits[t_fits['hb_b_flux'] != 0]
#         hb_dof = np.bincount(tsel['hb_dof'].data).argmax()
#     else:
#         hb_dof = np.bincount(t_fits['hb_dof'].data).argmax()
        
#     if (np.all(t_fits['ha_b_flux'].data != 0)):
#         tsel = t_fits[t_fits['ha_b_flux'] != 0]
#         nii_ha_dof = np.bincount(tsel['nii_ha_dof'].data).argmax()
#     else:
#         nii_ha_dof = np.bincount(t_fits['nii_ha_dof'].data).argmax()
        
#     #####################################################################
    
#     ## Final fit
#     t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)
#     for col in t_params.colnames:
#         t_params.rename_column(col, col.upper())
    
#     fits_final = construct_fits(t_params, 0)
#     dof = [hb_dof, oiii_dof, nii_ha_dof, sii_dof]
    
#     hb_rchi2, oiii_rchi2, \
#     nii_ha_rchi2, sii_rchi2 = emp.compute_final_rchi2(lam_rest, flam_rest, ivar_rest, \
#                                                       fits_final, dof)
    
#     hb_params['hb_rchi2'] = [hb_rchi2]
#     oiii_params['oiii_rchi2'] = [oiii_rchi2]
#     nii_ha_params['nii_ha_rchi2'] = [nii_ha_rchi2]
#     sii_params['sii_rchi2'] = [sii_rchi2]
    
#     total = tgt|hb_params|oiii_params|nii_ha_params|sii_params|per
#     tfinal = Table(total)
    
#     for col in tfinal.colnames:
#         tfinal.rename_column(col, col.upper())
        
#     tfinal.write(f'output/single_files/emfit-{targetid}.fits')
    
#     return(tfinal)

# ####################################################################################################
    
# def fit_spectra_iteration(lam, flam, ivar, n_sii = None, n_oiii = None, fit_cont = True):
#     """
#     Fit spectra for a given iteration of flux values.
    
#     Parameters
#     ----------
#     lam : numpy array
#         Rest-frame wavelength array of the spectra
        
#     flam : numpy array
#         Rest-frame flux array of the spectra (within 1-sigma errors)
        
#     ivar : numpy array
#         Rest-frame inverse variance array of the spectra
        
#     n_sii : int
#         Number of submodels in the [SII] fit. Default is None.
#         If n_sii = 2: Single-component [SII] fit
#         If n_sii = 4: Two-component [SII] fit
#         If n_sii = None: Find the best-fit for [SII]
        
#     n_oiii : int
#         Number of submodels in the [OIII] fit. Default is None.
#         If n_oiii = 2: Single-component [OIII] fit
#         If n_oiii = 4: Two-component [OIII] fit
#         If n_oiii = None: Find the best-fit for [OIII]
        
#     Returns
#     -------
#     fits : List
#         List of best-fits in the order - [Hb, [OIII], [NII]+Ha, [SII]]
    
#     rchi2s : List
#         List of Reduced chi2 values in the order - [Hb, [OIII], [NII]+Ha, [SII]]
        
#     t_params : Astropy Table
#         Table of output parameters for the fit
    
#     """
#     ######################################################################################
#     ## Fitting windows for the different emission-lines.
    
#     lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam, flam, \
#                                                          ivar, em_line = 'hb')

#     lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam, flam, \
#                                                                ivar, em_line = 'oiii')
#     lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam, flam, \
#                                                                      ivar, em_line = 'nii_ha')
#     lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam, flam, \
#                                                             ivar, em_line = 'sii')    
    
#     ######################################################################################
#     ## [SII] fit
#     if ((n_sii == 2)|(n_sii == 3)):
#         ## n_sii = 2 or 3 -- single component fits
        
#         gfit_sii, rchi2_sii = fit_lines.fit_sii_lines.fit_one_component(lam_sii, flam_sii, ivar_sii, \
#                                                                        fit_cont = fit_cont)
        
#         if (fit_cont == True):
#             sii_dof = 5
#         else:
#             sii_dof = 4
        
#     elif ((n_sii == 4)|(n_sii == 5)):
#         ## n_sii =  4 or 5 -- two component fits
#         gfit_sii, rchi2_sii = fit_lines.fit_sii_lines.fit_two_components(lam_sii, flam_sii, ivar_sii, \
#                                                                         fit_cont = fit_cont)
        
#         if (fit_cont == True):
#             sii_dof = 8
#         else:
#             sii_dof = 7
        
#     elif (n_sii == None):
#         ## Find the best fit
#         gfit_sii, rchi2_sii, _, sii_dof = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii, \
#                                                                   fit_cont = fit_cont)
    
#     ## [OIII] fit
#     if ((n_oiii == 2)|(n_oiii == 3)):
#         ## n_oiii = 2 or 3 -- single component fits
#         gfit_oiii, rchi2_oiii = fit_lines.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, ivar_oiii, \
#                                                                           fit_cont = fit_cont)
        
#         if (fit_cont == True):
#             oiii_dof = 4
#         else:
#             oiii_dof = 3
        
#     elif ((n_oiii == 4)|(n_oiii == 5)):
#         ## n_oiii = 4 or 5 -- two component fits
#         gfit_oiii, rchi2_oiii = fit_lines.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, ivar_oiii, \
#                                                                            fit_cont = fit_cont)
        
#         if (fit_cont == True):
#             oiii_dof = 7
#         else:
#             oiii_dof = 6
        
#     elif (n_oiii == None):
#         ## Find the best fit
#         gfit_oiii, rchi2_oiii, _, oiii_dof = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii, \
#                                                                      fit_cont = fit_cont)
        
#     ## Hb fit
#     gfit_hb, rchi2_hb, _, hb_dof = find_bestfit.find_hb_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii, \
#                                                            fit_cont = fit_cont)
    
#     ## [NII] + Ha fit
#     gfit_nii_ha, rchi2_nii_ha, _, nii_ha_dof = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
#                                                                         ivar_nii_ha, gfit_sii, \
#                                                                         fit_cont = fit_cont)
    
#     rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    
#     ######################################################################################
#     ## Parameters from the fit
    
#     hb_models = ['hb_n', 'hb_out', 'hb_b']
#     oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
#     nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
#     sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

#     hb_params = emp.get_parameters(gfit_hb, hb_models)
#     oiii_params = emp.get_parameters(gfit_oiii, oiii_models)
#     nii_ha_params = emp.get_parameters(gfit_nii_ha, nii_ha_models)
#     sii_params = emp.get_parameters(gfit_sii, sii_models)
    
#     if (fit_cont == False):
#         hb_cont = Const1D(amplitude = 0.0, name = 'hb_cont')
#         oiii_cont = Const1D(amplitude = 0.0, name = 'oiii_cont')
#         nii_ha_cont = Const1D(amplitude = 0.0, name = 'nii_ha_cont')
#         sii_cont = Const1D(amplitude = 0.0, name = 'sii_cont')
        
#         gfit_hb = hb_cont + gfit_hb
#         gfit_oiii = oiii_cont + gfit_oiii
#         gfit_nii_ha = nii_ha_cont + gfit_nii_ha
#         gfit_sii = sii_cont + gfit_sii
        
#     hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
#     oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
#     nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha['nii_ha_cont'].amplitude.value]
#     sii_params['sii_continuum'] = [gfit_sii['sii_cont'].amplitude.value]
    
#     hb_params['hb_dof'] = [hb_dof]
#     oiii_params['oiii_dof'] = [oiii_dof]
#     nii_ha_params['nii_ha_dof'] = [nii_ha_dof]
#     sii_params['sii_dof'] = [sii_dof]
    
#     fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
#     #n_dof = [hb_dof, oiii_dof, nii_ha_dof, sii_dof]
#     params = hb_params|oiii_params|nii_ha_params|sii_params    
    
#     ## Convert dictionary to table
#     t_params = Table(params)
    
#     return (fits, rchi2s, t_params)

# ####################################################################################################

# def check_fits(table, index):
#     specprod = table['SPECPROD'].astype(str).data[index]
#     targetid = table['TARGETID'].data[index]
#     survey = table['SURVEY'].astype(str).data[index]
#     program = table['PROGRAM'].astype(str).data[index]
#     healpix = table['HEALPIX'].data[index]
#     z = table['Z'].data[index]
#     logmass = table['logM'].data[index]
    
#     if ('Version_NII_Ha' in table.colnames):
#         version = table['Version_NII_Ha'].astype(str).data[index]

#         if (version == 'both'):
#             ver = 'v1'
#         else:
#             ver = version
#     else:
#         ver = 'v1'
    
    
#     lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
#                                                                    healpix, targetid, z, \
#                                                                    rest_frame = True, \
#                                                                    plot_continuum = False)

#     lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                          ivar_rest, em_line = 'hb')

#     lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                                ivar_rest, em_line = 'oiii')
#     lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                                      ivar_rest, em_line = 'nii_ha')
#     lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                             ivar_rest, em_line = 'sii')
    
#     gfit_sii, rchi2_sii, sii_bits, _ = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
#     gfit_oiii, rchi2_oiii, oiii_bits, _ = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)
#     gfit_hb, rchi2_hb, hb_bits, _ = find_bestfit.find_hb_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii)
#     gfit_nii_ha, rchi2_nii_ha, nii_ha_bits, _ = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
#                                                                                ivar_nii_ha, gfit_sii, ver = ver)
    
#     fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
#     rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
#     title = f'TARGETID: {targetid}; z: {round(z, 3)}; logmass: {round(logmass, 2)}\n'+ \
#     f'https://www.legacysurvey.org/viewer-desi/desi-spectrum/daily/targetid{targetid}'
    
#     fig = plot_utils.plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s, title = title)
    
#     return (fig)    

# ####################################################################################################

# def fit_spectra(specprod, survey, program, healpix, targetid, z, free_balmer = False):
#     """
#     Fit Hb, [OIII], [NII]+Ha, and [SII] emission lines for a given emission-line spectra
    
#     Parameters
#     ----------
#     specprod : str
#         Spectral Production Pipeline name fuji|guadalupe|...
        
#     survey : str
#         Survey name for the spectra
        
#     program : str
#         Program name for the spectra
        
#     healpix : str
#         Healpix number of the target
        
#     targetid : int64
#         The unique TARGETID associated with the target
        
#     z : float
#         Redshift of the target
        
#     free_balmer : bool
#         Whether or not to let sigma of balmer lines be free for the fit

#     Returns
#     -------
#     t_params : astropy table
#         Table of output parameters for the fit
#     """
    
#     ## Rest-frame emission-line spectra
#     coadd_spec, lam_rest, \
#     flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
#                                                          healpix, targetid, z, rest_frame = True)
    
#     ## Fitting windows for the different emission-lines.
    
#     lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                          ivar_rest, em_line = 'hb')

#     lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                                ivar_rest, em_line = 'oiii')
#     lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                                      ivar_rest, em_line = 'nii_ha')
#     lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
#                                                             ivar_rest, em_line = 'sii')
    
#     ## Fitting Routines
    
#     ## Bestfit for [SII]
#     gfit_sii, ndof_sii = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
#     ## Bestfit for [OIII]
#     gfit_oiii, ndof_oiii = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)

#     if (free_balmer == True):
#         ## Bestfit for Hb with sigma (Hb) varying freely
#         gfit_hb, ndof_hb = find_bestfit.find_hb_free_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii)
#         ## Bestfit for [NII]+Ha with sigma (Ha) varying freely
#         gfit_nii_ha, ndof_nii_ha = find_bestfit.find_nii_free_ha_best_fit(lam_nii_ha, flam_nii_ha, \
#                                                                          ivar_nii_ha, gfit_sii)
#     else:
#         ## Bestfit for Hb with sigma (Hb) fixed to [SII]
#         gfit_hb, ndof_hb = find_bestfit.find_hb_fixed_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii)
#         ## Bestfit for [NII]+Ha with sigma (Ha) fixed to [SII]
#         gfit_nii_ha, ndof_nii_ha = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
#                                                                     ivar_nii_ha, gfit_sii)
        
#     ## Compute reduced chi2:
#     rchi2_sii = mfit.calculate_chi2(flam_sii, gfit_sii(lam_sii), ivar_sii, \
#                                     ndof_sii, reduced_chi2 = True)
#     rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
#                                      ndof_oiii, reduced_chi2 = True)
#     rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
#                                    ndof_hb, reduced_chi2 = True)
#     rchi2_nii_ha = mfit.calculate_chi2(flam_nii_ha, gfit_nii_ha(lam_nii_ha), \
#                                            ivar_nii_ha, ndof_nii_ha, reduced_chi2 = True)
    
#     ### Parameters for the fit
#     hb_models = ['hb_n', 'hb_out', 'hb_b']
#     oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
#     nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
#     sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']
    
#     hb_params = emp.get_parameters(gfit_hb, hb_models)
#     oiii_params = emp.get_parameters(gfit_oiii, oiii_models)
#     nii_ha_params = emp.get_parameters(gfit_nii_ha, nii_ha_models)
#     sii_params = emp.get_parameters(gfit_sii, sii_models)
    
#     ## Continuum
#     hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
#     oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
#     nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha['nii_ha_cont'].amplitude.value]
#     sii_params['sii_continuum'] = [gfit_sii['sii_cont'].amplitude.value]
    
#     ## NOISE
#     hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
#     oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'oiii')
#     nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'nii_ha')
#     sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'sii')
    
#     hb_params['hb_noise'] = [hb_noise]
#     oiii_params['oiii_noise'] = [oiii_noise]
#     nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
#     sii_params['sii_noise'] = [sii_noise]
    
#     ## N(DOF)
#     hb_params['hb_ndof'] = [ndof_hb]
#     oiii_params['oiii_ndof'] = [ndof_oiii]
#     nii_ha_params['nii_ha_ndof'] = [ndof_nii_ha]
#     sii_params['sii_ndof'] = [ndof_sii]

#     ## Reduced chi2
#     hb_params['hb_rchi2'] = [rchi2_hb]
#     oiii_params['oiii_rchi2'] = [rchi2_oiii]
#     nii_ha_params['nii_ha_rchi2'] = [rchi2_nii_ha]
#     sii_params['sii_rchi2'] = [rchi2_sii]
    
#     ######## Combine everything
#     tgt = {}
#     tgt['targetid'] = [targetid]
#     tgt['specprod'] = [specprod]
#     tgt['survey'] = [survey]
#     tgt['program'] = [program]
#     tgt['healpix'] = [healpix]
#     tgt['z'] = [z]
    
#     t_params = Table(tgt|hb_params|oiii_params|nii_ha_params|sii_params)
#     for col in t_params.colnames:
#         t_params.rename_column(col, col.upper())
        
#     return (t_params)




