"""
This script consists of functions related to fitting the emission line spectra, 
and plotting the models and residuals.

Author : Ragadeepika Pucha
Version : 2024, March 21
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
    
    ## Rest-frame emission-line spectra
    coadd_spec, lam_rest, \
    flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                         healpix, targetid, z, rest_frame = True)
    
    ## Fit [SII] lines first
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                            ivar_rest, em_line = 'sii')
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
    
    ## Original Fits
    if ext_cond:
        ## Fit using extreme-line fitting code
        t_orig, fits_orig, \
        ndofs_orig, psel = fit_original_spectra.extreme_fit(lam_rest, flam_rest, ivar_rest)
    else:
        ## Fit using the normal source fitting code
        t_orig, fits_orig, \
        ndofs_orig, psel = fit_original_spectra.normal_fit(lam_rest, flam_rest, ivar_rest)
        
    ## Error spectra
    err_rest = 1/np.sqrt(ivar_rest) 
    err_rest[~np.isfinite(err_rest)] = 0.0
    res_matrix = coadd_spec.R['brz'][0]

    tables = []
    tables.append(t_orig)
    
    for kk in range(100):
        noise_spec = random.gauss(0, err_rest)
        to_add_spec = res_matrix.dot(noise_spec)
        flam_new = flam_rest + to_add_spec
        
        if ext_cond:
            ## Extreme-line fitting code
            t_params = fit_spectra_iteration.extreme_fit(lam_rest, flam_new, ivar_rest, \
                                                         fits_orig, psel)
        else:
            ## Normal source fitting code
            t_params = fit_spectra_iteration.normal_fit(lam_rest, flam_new, ivar_rest, \
                                                        fits_orig, psel)
            
        tables.append(t_params)
        
    t_fits = vstack(tables)
    
    per_ha = len(t_fits[t_fits['ha_b_flux'].data != 0])*100/len(t_fits)
    
    tgt = {}
    tgt['targetid'] = [targetid]
    tgt['specprod'] = [specprod]
    tgt['survey'] = [survey]
    tgt['program'] = [program]
    tgt['healpix'] = [healpix]
    tgt['z'] = [z]
    tgt['per_broad'] = [per_ha]

    ## Get bestfit parameters
    if ext_cond:
        ## Extreme-line fitting
        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allbestfit_params.extreme_fit(t_fits, ndofs_orig, \
                                                                          lam_rest, flam_rest, \
                                                                          ivar_rest)
    else:
        ## Normal source fitting
        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allbestfit_params.normal_fit(t_fits, ndofs_orig, \
                                                                         lam_rest, flam_rest, \
                                                                         ivar_rest)
    
    t_final = Table(tgt|hb_params|oiii_params|nii_ha_params|sii_params)
    
    for col in t_final.colnames:
        t_final.rename_column(col, col.upper())
    
    return (t_final)

####################################################################################################

class fit_original_spectra:
    """
    Functions to fit the original spectra for "normal" source fitting and 
    extreme broadline source fitting:
        1) normal_fit(lam_rest, flam_rest, ivar_rest)
        2) extreme_fit(lam_rest, flam_rest, ivar_rest)
    """
    
    def normal_fit(lam_rest, flam_rest, ivar_rest):
        """
        Function to fit the original "normal" source spectra.
        
        Parameters
        ----------
        lam_rest : numpy array
            Rest-frame Wavelength array of the spectra
            
        flam_rest : numpy array
            Rest-frame Flux array of the spectra
            
        ivar_rest : numpy array
            Rest-frame Inverse Variance array of the spectra
            
        Returns
        -------
        t_params : astropy table
            Table of the output parameters
            
        fits : list
            List of [Hb, [OIII], [NII]+Ha, and [SII]] fits
            
        ndofs : list
            List of number of degrees of freedom in [Hb, [OIII], [NII]+Ha, and [SII]] fits
            
        prior_sel : list
            Prior selected for the [NII]+Ha bestfit.
        """

        ## Fitting windows for the different emission-lines
        lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest,\
                                                             ivar_rest, em_line = 'hb')
        lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest,\
                                                                   ivar_rest, em_line = 'oiii')
        lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                         ivar_rest, \
                                                                         em_line = 'nii_ha')
        lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                ivar_rest, em_line = 'sii')

        ## Fits
        gfit_sii, ndof_sii = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
        gfit_oiii, ndof_oiii = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)
        gfit_nii_ha, ndof_nii_ha, prior_sel = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, \
                                                                                flam_nii_ha, \
                                                                                ivar_nii_ha, \
                                                                                gfit_sii)
        gfit_hb, ndof_hb = find_bestfit.find_hb_best_fit(lam_hb, \
                                                         flam_hb, \
                                                         ivar_hb, \
                                                         gfit_nii_ha)

        fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
        ndofs = [ndof_hb, ndof_oiii, ndof_nii_ha, ndof_sii]
        
        ## Compute reduced chi2:
        rchi2_sii = mfit.calculate_chi2(flam_sii, gfit_sii(lam_sii), ivar_sii, \
                                        ndof_sii, reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                         ndof_oiii, reduced_chi2 = True)
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                       ndof_hb, reduced_chi2 = True)
        rchi2_nii_ha = mfit.calculate_chi2(flam_nii_ha, gfit_nii_ha(lam_nii_ha), \
                                               ivar_nii_ha, ndof_nii_ha, reduced_chi2 = True)
        
        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allfit_params.normal_fit(fits, lam_rest, flam_rest)
        
        hb_params['hb_rchi2'] = [rchi2_hb]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]
        nii_ha_params['nii_ha_rchi2'] = [rchi2_nii_ha]
        sii_params['sii_rchi2'] = [rchi2_sii]
        
        oiii_params['hb_oiii_rchi2'] = [0.0]
        sii_params['nii_ha_sii_rchi2'] = [0.0]

        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)

        return (t_params, fits, ndofs, prior_sel)
    
####################################################################################################

    def extreme_fit(lam_rest, flam_rest, ivar_rest):
        """
        Function to fit the original extreme broadline source spectra.
        
        Parameters
        ----------
        lam_rest : numpy array
            Rest-frame Wavelength array of the spectra
            
        flam_rest : numpy array
            Rest-frame Flux array of the spectra
            
        ivar_rest : numpy array
            Rest-frame Inverse Variance array of the spectra
            
        Returns
        -------
        t_params : astropy table
            Table of the output parameters
            
        fits : list
            List of [Hb+[OIII] and [NII]+Ha+[SII]] fits
            
        ndofs : list
            List of number of degrees of freedom in [Hb+[OIII] and [NII]+Ha+[SII]] fits
            
        prior_sel : list
            Prior selected for the [NII]+Ha+[SII] bestfit.
        """
        
        ## Fitting windows for the different emission-line regions
        lam_nii_ha_sii, flam_nii_ha_sii, \
        ivar_nii_ha_sii = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest, 'nii_ha_sii')

        lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                           ivar_rest, 'hb_oiii')

        ## Fits
        gfit_nii_ha_sii, \
        ndof_nii_ha_sii, prior_sel = find_bestfit.find_nii_ha_sii_best_fit(lam_nii_ha_sii, \
                                                                           flam_nii_ha_sii, \
                                                                           ivar_nii_ha_sii)
        gfit_hb_oiii, ndof_hb_oiii = find_bestfit.find_hb_oiii_bestfit(lam_hb_oiii, \
                                                                       flam_hb_oiii, \
                                                                       ivar_hb_oiii, \
                                                                       gfit_nii_ha_sii)

        fits = [gfit_hb_oiii, gfit_nii_ha_sii]
        ndofs = [ndof_hb_oiii, ndof_nii_ha_sii]
        
        ## Compute reduced chi2
        rchi2_nii_ha_sii = mfit.calculate_chi2(flam_nii_ha_sii, gfit_nii_ha_sii(lam_nii_ha_sii), \
                                              ivar_nii_ha_sii, ndof_nii_ha_sii, reduced_chi2 = True)
        rchi2_hb_oiii = mfit.calculate_chi2(flam_hb_oiii, gfit_hb_oiii(lam_hb_oiii), \
                                           ivar_hb_oiii, ndof_hb_oiii, reduced_chi2 = True)

        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allfit_params.extreme_fit(fits, lam_rest, flam_rest)
        
        
        hb_params['hb_rchi2'] = [0.0]
        oiii_params['oiii_rchi2'] = [0.0]
        nii_ha_params['nii_ha_rchi2'] = [0.0]
        sii_params['sii_rchi2'] = [0.0]
        
        oiii_params['hb_oiii_rchi2'] = [rchi2_hb_oiii]
        sii_params['nii_ha_sii_rchi2'] = [rchi2_nii_ha_sii]

        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)

        return (t_params, fits, ndofs, prior_sel)

####################################################################################################
####################################################################################################

class fit_spectra_iteration:
    """
    Functions to fit a Monte Carlo iteration of the spectra.
        1) normal_fit(lam_rest, flam_new, ivar_rest, fits_orig, psel)
        2) extreme_fit(lam_rest, flam_rest, ivar_rest, fits_orig, psel)
    """
    
    def normal_fit(lam_rest, flam_new, ivar_rest, fits_orig, psel):
        """
        Function to fit an iteration of the "normal" source fit.
        
        Parameters
        ----------
        lam_rest : numpy array
            Rest-frame Wavelength array of the spectra
            
        flam_new : numpy array
            Rest-frame Flux array after adding noise within error bars.
            
        ivar_rest : numpy array
            Rest-frame Inverse Variance array of the spectra
            
        fits_orig : list
            List of original fits in the order - [Hb, [OIII], [NII]+Ha, [SII]]
            
        psel : list
            Prior selected for the [NII]+Ha bestfit
        
        Returns
        -------
        t_params : astropy table
            Table of the fit parameters
        """
    
        ## Original Fits
        hb_orig, oiii_orig, nii_ha_orig, sii_orig = fits_orig

        ## Fitting windows for the different emission-lines
        lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                             ivar_rest, em_line = 'hb')
        lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                   ivar_rest, em_line = 'oiii')
        lam_nii_ha, flam_nii_ha, \
        ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                ivar_rest, em_line = 'nii_ha')
        lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                ivar_rest, em_line = 'sii')

        ## [SII] models in the originial fit
        sii_models = sii_orig.submodel_names

        #################################### [SII] Fitting #########################################
        ## Fit [SII] 
        ## If [SII] has one component -- repeat with one-component fits
        ## If [SII] has two components -- repeat with two-component fits
        ## Check the existence of second component by looking for sii6716_out in submodels

        if ('sii6716_out' not in sii_models):
            ## one-component model
            gfit_sii = fl.fit_sii_lines.fit_one_component(lam_sii, flam_sii, ivar_sii)
        else:
            ## two-component model
            gfit_sii = fl.fit_sii_lines.fit_two_components(lam_sii, flam_sii, ivar_sii)

        ################################### [OIII] Fitting #########################################
        ## Fit [OIII]
        ## If [OIII] has one component -- repeat with one-component fits
        ## If [OIII] has two components -- repeat with two-component fits
        ## Check the existence of second component by looking for oiii5007_out in submodels

        if ('oiii5007_out' not in oiii_orig.submodel_names):
            ## one-component model
            gfit_oiii = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, ivar_oiii)
        else:
            ## two-component model
            gfit_oiii = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, ivar_oiii)

        ################################### [NII]+Ha Fitting #######################################
        ## Fit [NII]+Ha
        ## If [SII] has two components -- two component model
        ## If [SII] has one component -- one component model
        ## If sig (Ha) and sig([SII]) are not close or sig (Ha) < sig ([SII]) -- fixed version
        ## If sig([Ha]) and sig([SII]) are close -- fixed version
        ## If 'ha_b' in submodels -- broad_comp = True

        if ('sii6716_out' not in sii_models):
            ## one-component model
            sig_ha = mfit.lamspace_to_velspace(nii_ha_orig['ha_n'].stddev.value, \
                                               nii_ha_orig['ha_n'].mean.value)
            sig_sii = mfit.lamspace_to_velspace(sii_orig['sii6716'].stddev.value, \
                                                sii_orig['sii6716'].mean.value)

            if ((sig_ha > sig_sii)&(~np.isclose(sig_ha, sig_sii))):
                ## Free version
                if ('ha_b' in nii_ha_orig.submodel_names):
                    ## Broad component exists
                    gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, \
                                                                                flam_nii_ha, \
                                                                                ivar_nii_ha, \
                                                                                gfit_sii, \
                                                                                priors = psel, \
                                                                                broad_comp = True)
                else:
                    ## No broad component
                    gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_free_ha_one_component(lam_nii_ha, \
                                                                                flam_nii_ha, \
                                                                                ivar_nii_ha, \
                                                                                gfit_sii, \
                                                                                broad_comp = False)
            else:
                ## Fixed version
                if ('ha_b' in nii_ha_orig.submodel_names):
                    ## Broad component exists
                    gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, \
                                                                            flam_nii_ha, \
                                                                            ivar_nii_ha, \
                                                                            gfit_sii, \
                                                                            priors = psel, \
                                                                            broad_comp = True)
                else:
                    ## No broad component
                    gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_ha_one_component(lam_nii_ha, \
                                                                            flam_nii_ha, \
                                                                            ivar_nii_ha, \
                                                                            gfit_sii, \
                                                                            broad_comp = False)
        else:
            ## two-component model

            if ('ha_b' in nii_ha_orig.submodel_names):
                ## Broad component exists
                gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, \
                                                                        flam_nii_ha, \
                                                                        ivar_nii_ha, \
                                                                        gfit_sii, \
                                                                        priors = psel, \
                                                                        broad_comp = True)
            else:
                ## No broad component
                gfit_nii_ha = fl.fit_nii_ha_lines.fit_nii_ha_two_components(lam_nii_ha, \
                                                                        flam_nii_ha, \
                                                                        ivar_nii_ha, \
                                                                        gfit_sii, \
                                                                        broad_comp = False)  

        ####################################### Hb Fitting #########################################
        ## Fit Hb
        ## If [SII] has one component -- One component model
        ## If [SII] has two components -- Two component model

        if ('sii6716_out' not in sii_models):
            ## one-component model
            gfit_hb = fl.fit_hb_line.fit_hb_one_component(lam_hb, flam_hb, ivar_hb, gfit_nii_ha)
        else:
            ## two-component model
            gfit_hb = fl.fit_hb_line.fit_hb_two_components(lam_hb, flam_hb, ivar_hb, gfit_nii_ha)

        ############################################################################################

        fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]

        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allfit_params.normal_fit(fits, lam_rest, flam_new)

        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)

        return (t_params)
    
####################################################################################################

    def extreme_fit(lam_rest, flam_new, ivar_rest, fits_orig, psel):
        """
        Function to fit an iteration of the extreme-broadline source fit.
        
        Parameters
        ----------
        lam_rest : numpy array
            Rest-frame Wavelength array of the spectra
            
        flam_new : numpy array
            Rest-frame Flux array after adding noise within error bars.
            
        ivar_rest : numpy array
            Rest-frame Inverse Variance array of the spectra
            
        fits_orig : list
            List of original fits in the order - [Hb+[OIII], [NII]+Ha+[SII]]
            
        psel : list
            Prior selected for the [NII]+Ha+[SII] bestfit
        
        Returns
        -------
        t_params : astropy table
            Table of the fit parameters
        """
        
        ## Original Fits
        hb_oiii_orig, nii_ha_sii_orig = fits_orig

        ## Fitting windows for the different emission-line regions
        lam_nii_ha_sii, flam_nii_ha_sii, \
        ivar_nii_ha_sii = spec_utils.get_fit_window(lam_rest, flam_new, ivar_rest, 'nii_ha_sii')

        lam_hb_oiii, flam_hb_oiii, ivar_hb_oiii = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                           ivar_rest, 'hb_oiii')

        ####################################### [NII]+Ha+[SII] Fitting #############################

        ## [NII]+Ha+[SII] Fit
        ## Repeat with the same function and prior
        gfit_nii_ha_sii = fl.fit_extreme_broadline_sources.fit_nii_ha_sii(lam_nii_ha_sii, \
                                                                          flam_nii_ha_sii, \
                                                                          ivar_nii_ha_sii, \
                                                                          priors = psel)

        ############################## Hb + [OIII] Fitting #########################################

        ## Hb+[OIII] Fit
        ## If [OIII] has one component - one-component fit
        ## If [OIII] has two components - two-components fit

        if ('oiii5007_out' not in hb_oiii_orig.submodel_names):
            ## One component model
            gfit_hb_oiii = fl.fit_extreme_broadline_sources.fit_hb_oiii_1comp(lam_hb_oiii, \
                                                                              flam_hb_oiii, \
                                                                              ivar_hb_oiii, \
                                                                              gfit_nii_ha_sii)
        else:
            ## two component model
            gfit_hb_oiii = fl.fit_extreme_broadline_sources.fit_hb_oiii_2comp(lam_hb_oiii, \
                                                                              flam_hb_oiii, \
                                                                              ivar_hb_oiii, \
                                                                              gfit_nii_ha_sii)

        ############################################################################################

        fits = [gfit_hb_oiii, gfit_nii_ha_sii]

        hb_params, oiii_params, \
        nii_ha_params, sii_params = emp.get_allfit_params.extreme_fit(fits, lam_rest, flam_new)

        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)

        return (t_params)

####################################################################################################
####################################################################################################

class construct_fits_from_table:
    """
    Includes functions to construct fits from the table for a given source.
    Two functions:
        1) normal_fit(t, index)
        2) extreme_fit(t, index)
    """

    def normal_fit(t, index):
        """
        Construct fits of a particular source from the table of parameters. 
        This is for normal fitting sources.

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

    def extreme_fit(t, index):
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






