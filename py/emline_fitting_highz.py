"""
This script consists of functions related to fitting the emission line spectra for high-z spectra.
It consists of the following functions:
    1) fit_highz_spectra(specprod, survey, program, healpix, targetid, z)
    2) fit_original(lam_rest, flam_rest, ivar_rest, rsigma)
    3) fit_iteration(lam_rest, flam_new, ivar_rest, rsigma, fit_orig, psel)

Author : Ragadeepika Pucha
Version : 2024, August 30
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

class fit_highz_spectra_free_hb:
    """
    Functions related to fitting High-z spectra where Hb is fit independently:
        1) fit_highz_spectra(specprod, survey, program, healpix, target, z)
        2) fit_original(lam_rest, flam_rest, ivar_rest, rsigma)
        3) fit_iteration(lam_rest, flam_new, ivar_rest, rsigma, fits_orig, psel)
    """

    def fit_highz_spectra(specprod, survey, program, healpix, targetid, z):
        """
        Function to fit a single spectrum for Hb and [OIII].
        Hb is fit freely.

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
        t_orig, fits_orig, ndofs_orig, psel = fit_highz_spectra_free_hb.fit_original(lam_rest, flam_rest, \
                                                                                     ivar_rest, rsigma)

        err_rest = 1/np.sqrt(ivar_rest)
        err_rest[~np.isfinite(err_rest)] = 0.0
        res_matrix = coadd_spec.R['brz'][0]

        tables = []
        tables.append(t_orig)

        for kk in range(100):
            noise_spec = random.gauss(0, err_rest)
            to_add_spec = res_matrix.dot(noise_spec)
            flam_new = flam_rest + to_add_spec
            t_fit = fit_highz_spectra_free_hb.fit_iteration(lam_rest, flam_new, ivar_rest,\
                                                            rsigma, fits_orig, psel)

            tables.append(t_fit)

        t_fits = vstack(tables)

        ## Percentage of broad Hb
        per_hb = len(t_fits[t_fits['hb_b_flux'].data != 0])*100/len(t_fits)

        ## Bestfit Parameters
        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_bestfit_parameters(t_fits, hb_models, 'hb')

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_bestfit_parameters(t_fits, oiii_models, 'oiii')

        ## TARGET Information
        tgt = {}
        tgt['targetid'] = [targetid]
        tgt['specprod'] = [specprod]
        tgt['survey'] = [survey]
        tgt['program'] = [program]
        tgt['healpix'] = [healpix]
        tgt['z'] = [z]
        tgt['per_broad'] = [per_hb]

        gfit_hb, gfit_oiii = fits_orig
        ndof_hb, ndof_oiii = ndofs_orig

        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                      ivar_rest, rsigma, \
                                                                      em_line = 'hb')

        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ## Compute Reduced chi2
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                      ndof_hb, reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                        ndof_oiii, reduced_chi2 = True)

        ## Add reduced chi2 and ndof
        hb_params['hb_ndof'] = [ndof_hb]
        hb_params['hb_rchi2'] = [rchi2_hb]

        oiii_params['oiii_ndof'] = [ndof_oiii]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]

        t_final = Table(tgt|hb_params|oiii_params)

        for col in t_final.colnames:
            t_final.rename_column(col, col.upper())

        ## Check sigma values for unresolved cases
        ## t_final = emp.fix_sigma(t_final)

        return (t_final)

    ####################################################################################################
    ####################################################################################################

    def fit_original(lam_rest, flam_rest, ivar_rest, rsigma):
        """
        Function to fit the original spectra for Hb and [OIII].
        Both emission-line are fit independently.

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

        fits : list
            List of [Hb and [OIII]] fits

        ndofs : list
            List of number of degrees of freedom in [Hb and [OIII]] fits

        prior_sel : list
            Prior selected for the Hb bestfit.
        """

        ## Fitting window for Hb
        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest,\
                                                                     rsigma, em_line = 'hb')

        ## Fitting window for [OIII]
        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ## [OIII] Fit 
        gfit_oiii, ndof_oiii = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii,\
                                                               ivar_oiii, rsig_oiii)

        ## Free Hb Fit
        gfit_hb, ndof_hb, prior_sel = find_bestfit.highz_fit.find_free_hb_best_fit(lam_hb, flam_hb, \
                                                                                   ivar_hb, rsig_hb)

        fits = [gfit_hb, gfit_oiii]
        ndofs = [ndof_hb, ndof_oiii]

        ## Compute reduced chi2
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                      ndof_hb, reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                        ndof_oiii, reduced_chi2 = True)

        ## Parameters
        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        ## Noise
        hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
        hb_params['hb_noise'] = [hb_noise]
        hb_params['hb_rchi2'] = [rchi2_hb]

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_parameters(gfit_oiii, oiii_models, rsig_oiii)
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'oiii')
        oiii_params['oiii_noise'] = [oiii_noise]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]

        t_orig = Table(hb_params|oiii_params)

        return (t_orig, fits, ndofs, prior_sel)

    ####################################################################################################
    ####################################################################################################

    def fit_iteration(lam_rest, flam_new, ivar_rest, rsigma, fits_orig, psel):
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

        fitsorig : list
            Original fit for [Hb and [OIII]] emission lines.

        psel : list
            Prior selected for the Hb bestfit

        Returns
        -------
        t_fit : astropy table
            Table of the fit parameters
        """

        ## Original Fits
        hb_orig, oiii_orig = fits_orig


        ## Fitting window for Hb
        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                      ivar_rest, rsigma, \
                                                                      em_line = 'hb')

        ## Fitting window for [OIII]
        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ####################################### [OIII] Fitting #########################################
        ## Fit [OIII]
        ## If [OIII] has one component -- repeat with one-component fit
        ## If [OIII] has two components -- repeat with two-component fits
        ## Check the existence of second component by looking for oiii5007_out in submodels

        if ('oiii5007_out' not in oiii_orig.submodel_names):
            ## One-component model
            gfit_oiii = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, \
                                                           ivar_oiii, rsig_oiii)
        else:
            ## Two-component model
            gfit_oiii = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, \
                                                            ivar_oiii, rsig_oiii)

        ###################################### Hbeta Fitting ###########################################
        ## Fit Hb
        ## If Hb has a broad component -- repeat with broad-component fit
        ## If Hb has only narrow component -- repeat with no-broad-component fit
        ## Check the existence of the broad component by looking for hb_b in submodels

        if ('hb_b' in hb_orig.submodel_names):
            gfit_hb = fl.fit_highz_hb_oiii_lines.fit_free_hb(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                                                             priors = psel, broad_comp = True)
        else:
            gfit_hb = fl.fit_highz_hb_oiii_lines.fit_free_hb(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                                                             priors = psel, broad_comp = False)

        ##############################################################################################
        ## Parameters

        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        ## Noise
        hb_noise = mfit.compute_noise_emline(lam_rest, flam_new, 'hb')
        hb_params['hb_noise'] = [hb_noise]

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_parameters(gfit_oiii, oiii_models, rsig_oiii)
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        oiii_noise = mfit.compute_noise_emline(lam_rest, flam_new, 'oiii')
        oiii_params['oiii_noise'] = [oiii_noise]    

        t_fit = Table(hb_params|oiii_params)

        return (t_fit)

####################################################################################################
####################################################################################################

class fit_highz_spectra_fixed_hb:
    """
    Functions related to fitting High-z spectra where Hb is tied to [OIII]:
        1) fit_highz_spectra(specprod, survey, program, healpix, target, z)
        2) fit_original(lam_rest, flam_rest, ivar_rest, rsigma)
        3) fit_iteration(lam_rest, flam_new, ivar_rest, rsigma, fits_orig, psel)
    """
    
    def fit_highz_spectra(specprod, survey, program, healpix, targetid, z):
        """
        Function to fit a single spectrum for Hb and [OIII].
        Hb is fit freely.

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
        t_orig, fits_orig, ndofs_orig, psel = fit_highz_spectra_fixed_hb.fit_original(lam_rest, flam_rest, \
                                                                                      ivar_rest, rsigma)

        err_rest = 1/np.sqrt(ivar_rest)
        err_rest[~np.isfinite(err_rest)] = 0.0
        res_matrix = coadd_spec.R['brz'][0]

        tables = []
        tables.append(t_orig)

        for kk in range(100):
            noise_spec = random.gauss(0, err_rest)
            to_add_spec = res_matrix.dot(noise_spec)
            flam_new = flam_rest + to_add_spec
            t_fit = fit_highz_spectra_fixed_hb.fit_iteration(lam_rest, flam_new, ivar_rest,\
                                                             rsigma, fits_orig, psel)

            tables.append(t_fit)

        t_fits = vstack(tables)

        ## Percentage of broad Hb
        per_hb = len(t_fits[t_fits['hb_b_flux'].data != 0])*100/len(t_fits)

        ## Bestfit Parameters
        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_bestfit_parameters(t_fits, hb_models, 'hb')

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_bestfit_parameters(t_fits, oiii_models, 'oiii')

        ## TARGET Information
        tgt = {}
        tgt['targetid'] = [targetid]
        tgt['specprod'] = [specprod]
        tgt['survey'] = [survey]
        tgt['program'] = [program]
        tgt['healpix'] = [healpix]
        tgt['z'] = [z]
        tgt['per_broad'] = [per_hb]

        gfit_hb, gfit_oiii = fits_orig
        ndof_hb, ndof_oiii = ndofs_orig

        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                      ivar_rest, rsigma, \
                                                                      em_line = 'hb')

        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ## Compute Reduced chi2
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                      ndof_hb, reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                        ndof_oiii, reduced_chi2 = True)

        ## Add reduced chi2 and ndof
        hb_params['hb_ndof'] = [ndof_hb]
        hb_params['hb_rchi2'] = [rchi2_hb]

        oiii_params['oiii_ndof'] = [ndof_oiii]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]

        t_final = Table(tgt|hb_params|oiii_params)

        for col in t_final.colnames:
            t_final.rename_column(col, col.upper())

        ## Check sigma values for unresolved cases
        ## t_final = emp.fix_sigma(t_final)
        
        return (t_final)

    ####################################################################################################
    ####################################################################################################

    def fit_original(lam_rest, flam_rest, ivar_rest, rsigma):
        """
        Function to fit the original spectra for Hb and [OIII].
        Both emission-line are fit independently.

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

        fits : list
            List of [Hb and [OIII]] fits

        ndofs : list
            List of number of degrees of freedom in [Hb and [OIII]] fits

        prior_sel : list
            Prior selected for the Hb bestfit.
        """

        ## Fitting window for Hb
        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest,\
                                                                     rsigma, em_line = 'hb')

        ## Fitting window for [OIII]
        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ## [OIII] Fit 
        gfit_oiii, ndof_oiii = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii,\
                                                               ivar_oiii, rsig_oiii)

        ## Free Hb Fit
        gfit_hb, ndof_hb, prior_sel = find_bestfit.highz_fit.find_fixed_hb_best_fit(lam_hb, flam_hb, \
                                                                                    ivar_hb, rsig_hb, \
                                                                                   gfit_oiii, rsig_oiii)

        fits = [gfit_hb, gfit_oiii]
        ndofs = [ndof_hb, ndof_oiii]

        ## Compute reduced chi2
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, \
                                      ndof_hb, reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, \
                                        ndof_oiii, reduced_chi2 = True)

        ## Parameters
        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        ## Noise
        hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'hb')
        hb_params['hb_noise'] = [hb_noise]
        hb_params['hb_rchi2'] = [rchi2_hb]

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_parameters(gfit_oiii, oiii_models, rsig_oiii)
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, 'oiii')
        oiii_params['oiii_noise'] = [oiii_noise]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]

        t_orig = Table(hb_params|oiii_params)

        return (t_orig, fits, ndofs, prior_sel)

    ####################################################################################################
    ####################################################################################################

    def fit_iteration(lam_rest, flam_new, ivar_rest, rsigma, fits_orig, psel):
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

        fitsorig : list
            Original fit for [Hb and [OIII]] emission lines.

        psel : list
            Prior selected for the Hb bestfit

        Returns
        -------
        t_fit : astropy table
            Table of the fit parameters
        """

        ## Original Fits
        hb_orig, oiii_orig = fits_orig


        ## Fitting window for Hb
        lam_hb, flam_hb, ivar_hb, rsig_hb = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                      ivar_rest, rsigma, \
                                                                      em_line = 'hb')

        ## Fitting window for [OIII]
        lam_oiii, flam_oiii, ivar_oiii, rsig_oiii = spec_utils.get_fit_window(lam_rest, flam_new, \
                                                                             ivar_rest, rsigma, \
                                                                             em_line = 'oiii')

        ####################################### [OIII] Fitting #########################################
        ## Fit [OIII]
        ## If [OIII] has one component -- repeat with one-component fit
        ## If [OIII] has two components -- repeat with two-component fits
        ## Check the existence of second component by looking for oiii5007_out in submodels

        if ('oiii5007_out' not in oiii_orig.submodel_names):
            ## One-component model
            gfit_oiii = fl.fit_oiii_lines.fit_one_component(lam_oiii, flam_oiii, \
                                                           ivar_oiii, rsig_oiii)
        else:
            ## Two-component model
            gfit_oiii = fl.fit_oiii_lines.fit_two_components(lam_oiii, flam_oiii, \
                                                            ivar_oiii, rsig_oiii)

        ###################################### Hbeta Fitting ###########################################
        ## Fit Hb
        ## If Hb has a broad component -- repeat with broad-component fit
        ## If Hb has only narrow component -- repeat with no-broad-component fit
        ## Check the existence of the broad component by looking for hb_b in submodels

        if ('hb_b' in hb_orig.submodel_names):
            gfit_hb = fl.fit_highz_hb_oiii_lines.fit_fixed_hb(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                                                              gfit_oiii, rsig_oiii,
                                                              priors = psel, broad_comp = True)
        else:
            gfit_hb = fl.fit_highz_hb_oiii_lines.fit_fixed_hb(lam_hb, flam_hb, ivar_hb, rsig_hb, \
                                                              gfit_oiii, rsig_oiii,
                                                              priors = psel, broad_comp = False)

        ##############################################################################################
        ## Parameters

        hb_models = ['hb_n', 'hb_b']
        hb_params = emp.get_parameters(gfit_hb, hb_models, rsig_hb)
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        ## Noise
        hb_noise = mfit.compute_noise_emline(lam_rest, flam_new, 'hb')
        hb_params['hb_noise'] = [hb_noise]

        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        oiii_params = emp.get_parameters(gfit_oiii, oiii_models, rsig_oiii)
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        oiii_noise = mfit.compute_noise_emline(lam_rest, flam_new, 'oiii')
        oiii_params['oiii_noise'] = [oiii_noise]    

        t_fit = Table(hb_params|oiii_params)

        return (t_fit)

####################################################################################################
####################################################################################################
