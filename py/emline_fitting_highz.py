"""
This script consists of functions related to fitting the emission line spectra for high-z spectra.
It consists of the following functions:
    1) fit_highz_hb(specprod, survey, program, healpix, targetid, z)
    2) fit_original_hb(lam_rest, flam_rest, ivar_rest, rsigma)
    3) fit_iteration_hb(lam_rest, flam_new, ivar_rest, rsigma, fit_orig, psel)


Author : Ragadeepika Pucha
Version : 2024, April 2022
"""

####################################################################################################

import numpy as np

from astropy.table import Table, vstack, hstack
from astropy.modeling.models import Gaussian1D, Const1D

import spec_utils, plot_utils
import fit_lines as fl
import measure_fits as mfit
import emline_params as emp
import find_bestfit

from desiutil.dust import dust_transmission

import matplotlib.pyplot as plt
import random

####################################################################################################
####################################################################################################

def fit_highz_hb(specprod, survey, program, healpix, targetid, z):
    """
    Function to fit a single spectrum for Hb alone.
    
    Parameters 
    ----------
    specprod : str
        Spectral Production Pipeline name 
        fuji|guadalupe|...
        
    survey : str
        Survey name for the spectrum
        
    program : str
        Program name for the spectrum
        
    healpix : str
        Healpix number of the target
        
    targets : int64
        Unique TARGETID of the target
        
    z : float
        Redshift of the target
        
    Returns
    -------
    t_final : astropy table
        Table of fit parameters
    
    """
    
    ## Rest-frame emission-line spectra
    coadd_spec, lam_rest, \
    flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, healpix,\
                                                         targetid, z, rest_frame = True)
    
    ## 1D resolution array
    rsigma = spec_utils.compute_resolution_sigma(coadd_spec)[0]
    t_orig, gfit_hb, ndof_hb, psel = fit_original_hb(lam_rest, flam_rest, ivar_rest, rsigma)
   
    err_rest = 1/np.sqrt(ivar_rest)
    err_rest[~np.isfinite(err_rest)] = 0.0
    res_matrix = coadd_spec.R['brz'][0]
    
    tables = []
    tables.append(t_orig)
    
    for kk in range(100):
        noise_spec = random.gauss(0, err_rest)
        to_add_spec = res_matrix.dot(noise_spec)
        flam_new = flam_rest + to_add_spec
        t_fit = fit_iteration_hb(lam_rest, flam_new, ivar_rest, rsigma, gfit_hb, psel)
        
        tables.append(t_fit)
        
    t_fits = vstack(tables)
        
    hb_models = ['hb_n', 'hb_b']
    hb_params = emp.get_bestfit_parameters(t_fits, hb_models, 'hb')
    
    ## TARGET Information
    tgt = {}
    tgt['targetid'] = [targetid]
    tgt['specprod'] = [specprod]
    tgt['survey'] = [survey]
    tgt['program'] = [program]
    tgt['healpix'] = [healpix]
    tgt['z'] = [z]
    
    lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                  ivar_rest, rsigma, \
                                                                  em_line = 'hb')
    
    ## Compute Reduced chi2
    rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                  ndof_hb, reduced_chi2 = True)
    
    ## Add reduced chi2 and ndof
    hb_params['hb_ndof'] = [ndof_hb]
    hb_params['hb_rchi2'] = [rchi2_hb]
    
    t_final = Table(tgt|hb_params)
    
    for col in t_final.colnames:
        t_final.rename_column(col, col.upper())

    return (t_final)

####################################################################################################
####################################################################################################

def fit_original_hb(lam_rest, flam_rest, ivar_rest, rsigma):
    """
    Function to fit the original spectra for Hb alone.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame Wavelength array of the spectra

    flam_rest : numpy array
        Rest-frame Flux array of the spectra

    ivar_rest : numpy array
        Rest-frame Inverse Variance array of the spectra

    rsigma : numpy array
        1D array of Intrumental resolution elements

    Returns
    -------
    t_orig : astropy table
        Table of the output parameters

    gfit_hb : Astropy model
        Bestfit model for Hb

    ndof_hb : int
        Number of degrees of freedom for Hb model

    psel : list
        Prior selected for the Hb bestfit.
    """
    
    ## Fitting window for Hb
    lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest,\
                                                                 rsigma, em_line = 'hb')
    
    ## Fit
    gfit_hb, ndof_hb, psel = find_bestfit.find_free_hb_best_fit(lam_hb, flam_hb, ivar_hb, rsig_hb)
    
    hb_models = ['hb_n', 'hb_b']
    hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
    hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
    
    ## Noise
    hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
    hb_params['hb_noise'] = [hb_noise]
    
    t_orig = Table(hb_params)
    
    return (t_orig, gfit_hb, ndof_hb, psel)

####################################################################################################
####################################################################################################

def fit_iteration_hb(lam_rest, flam_new, ivar_rest, rsigma, fit_orig, psel):
    """
    Function to fit an iteration of the spectra fit for Hb alone.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame Wavelength array of the spectra

    flam_new : numpy array
        Rest-frame Flux array after adding noise within error bars.

    ivar_rest : numpy array
        Rest-frame Inverse Variance array of the spectra

    rsigma : numpy array
        1D array of Intrumental resolution elements

    fit_orig : Astropy model
        Original fit for Hb

    psel : list
        Prior selected for the Hb bestfit

    Returns
    -------
    t_params : astropy table
        Table of the fit parameters
    """
    
    
    ## Fitting window for Hb
    lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_new, ivar_rest, \
                                                                     rsigma, em_line = 'hb')
    
    if ('hb_b' in fit_orig.submodel_names):
        gfit_hb = fl.fit_hb_line.fit_free_hb_one_component(lam_hb, flam_hb, \
                                                                  ivar_hb, rsig_hb, \
                                                                  priors = psel, \
                                                                  broad_comp = True)
    else:
        gfit_hb = fl.fit_hb_line.fit_free_hb_one_component(lam_hb, flam_hb, \
                                                                  ivar_hb, rsig_hb, \
                                                                  priors = psel, \
                                                                  broad_comp = False)
    hb_models = ['hb_n', 'hb_b']
    hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
    hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
    t_fit = Table(hb_params)
    
    return (t_fit)

####################################################################################################
####################################################################################################

