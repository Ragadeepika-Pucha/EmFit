"""
This script consists of functions for computing the parameters of the emission-line fits.
It consists of the following functions:
    1) get_parameters(gfit, models)
    2) get_bestfit_parameters(table, models, emline)
    3) get_allfit_params.normal_fit(fits, lam, flam)
    4) get_allfit_params.extreme_fit(fits, lam, flam)
    
Author : Ragadeepika Pucha
Version : 2024, March 14
"""

###################################################################################################
from astropy.table import Table
import numpy as np

import measure_fits as mfit
import emline_fitting as emfit
import spec_utils
from astropy.stats import sigma_clipped_stats
###################################################################################################

def get_parameters(gfit, models):
    """
    Function to get amplitude, mean, standard deviation, sigma, and flux for each of 
    model components in a given emission-line model.
    
    Parameters
    ----------
    gfit : Astropy model
        Compound model for the emission-line
        
    models : list
        List of total submodels expected from a given emission-line fitting.
        
    Returns
    -------
    params : dict
        Dictionary with the parameter values
    """
    
    params = {}
    n = gfit.n_submodels
    
    if (n > 1):
        names = gfit.submodel_names
    else:
        names = gfit.name
    
    for model in models:
        if (model in names):
            if (n == 1):
                amp, mean, std = gfit.parameters
            else:
                amp, mean, std = gfit[model].parameters
            sig = mfit.lamspace_to_velspace(std, mean)
            flux = mfit.compute_emline_flux(amp, std)
            
            params[f'{model}_amplitude'] = [amp]
            params[f'{model}_mean'] = [mean]
            params[f'{model}_std'] = [std]
            params[f'{model}_sigma'] = [sig]
            params[f'{model}_flux'] = [flux]
        else:
            params[f'{model}_amplitude'] = [0.0]
            params[f'{model}_mean'] = [0.0]
            params[f'{model}_std'] = [0.0]
            params[f'{model}_sigma'] = [0.0]
            params[f'{model}_flux'] = [0.0]
            
    return (params)
    
###################################################################################################
###################################################################################################

def get_bestfit_parameters(table, models, emline):
    """
    Function to get the bestfit parameters from the table of iterations.
    If the model component is not available, then the bestfit parameters is set to zero.
    Otherwise, the sigma clipped median and standard deviation are taken as the value and error
    for a given parameter.
    
    Parameters
    ----------
    table : Astropy Table
        Table of iteration parameters
        
    models : list
        List of Gaussian models for a given emission-line fit
        
    emline : str
        Emission-line name of the models
        Can be "hb", "oiii", "nii_ha", or "sii"
        
    Returns
    -------
    params : dict
        Dictionary of bestfit parameters
    """
    params = {}
    
    for model in models:
        amplitude_arr = table[f'{model}_amplitude'].data
        mean_arr = table[f'{model}_mean'].data
        std_arr = table[f'{model}_std'].data
        var_arr = std_arr**2
        flux_arr = table[f'{model}_flux'].data
        sigma_arr = table[f'{model}_sigma'].data
        
        amp_zero = (np.all(np.isclose(amplitude_arr, 0.0)))
        mean_zero = (np.all(np.isclose(mean_arr, 0.0)))
        std_zero = (np.all(np.isclose(std_arr, 0.0)))
        
        allzero = amp_zero&mean_zero&std_zero
        
        if (allzero):
            ## When the model is not available
            amp = 0.0
            amp_err = 0.0
            mean = 0.0
            mean_err = 0.0
            std = 0.0
            std_err = 0.0
            flux = 0.0
            flux_err = 0.0
            sigma = 0.0
            sigma_err = 0.0
            flux_fits = 0.0
            flux_err_fits = 0.0
            sigma_fits = 0.0
            sigma_err_fits = 0.0
            
        else:
            ## Sigma-clipped stats
            # _, amp, amp_err = sigma_clipped_stats(amplitude_arr)
            # _, mean, mean_err = sigma_clipped_stats(mean_arr)
            # _, std, std_err = np.sqrt(sigma_clipped_stats(var_arr))
            # _, flux_fits, flux_err_fits = sigma_clipped_stats(flux_arr)
            # _, sigma_fits, sigma_err_fits = sigma_clipped_stats(sigma_arr)
            
            ## Version 2 
            cond = (amplitude_arr > 0)&(mean_arr > 0)&(std_arr > 0)
            
            amp = np.nanmean(np.where(~cond, np.nan, amplitude_arr))
            amp_err = np.std(amplitude_arr)
            mean = np.nanmean(np.where(~cond, np.nan, mean_arr))
            mean_err = np.std(mean_arr)
            std = np.sqrt(np.nanmean(np.where(~cond, np.nan, var_arr)))
            std_err = np.std(std_arr)
            flux_fits = np.nanmean(np.where(~cond, np.nan, flux_arr))
            flux_err_fits = np.std(flux_arr)
            sigma_fits = np.nanmean(np.where(~cond, np.nan, sigma_arr))
            sigma_err_fits = np.std(sigma_arr)
            flux, flux_err = mfit.compute_emline_flux(amp, std, amp_err, std_err)
            sigma, sigma_err = mfit.compute_emline_flux(std, mean, std_err, mean_err)
            
            # if ((amp == 0)|(std == 0)):
            #     flux, flux_err = 0.0, 0.0
            #     sigma, sigma_err = 0.0, 0.0
            # else:
            #     flux, flux_err = mfit.compute_emline_flux(amp, std, amp_err, std_err)
            #     sigma, sigma_err = mfit.lamspace_to_velspace(std, mean, std_err, mean_err)
                        
        params[f'{model}_amplitude'] = [amp]
        params[f'{model}_amplitude_err'] = [amp_err]
        params[f'{model}_mean'] = [mean]
        params[f'{model}_mean_err'] = [mean_err]
        params[f'{model}_std'] = [std]
        params[f'{model}_std_err'] = [std_err]
        params[f'{model}_flux'] = [flux]
        params[f'{model}_flux_err'] = [flux_err]
        params[f'{model}_sigma'] = [sigma]
        params[f'{model}_sigma_err'] = [sigma_err]
        params[f'{model}_flux_fits'] = [flux_fits]
        params[f'{model}_flux_fits_err'] = [flux_err_fits]
        params[f'{model}_sigma_fits'] = [sigma_fits]
        params[f'{model}_sigma_fits_err'] = [sigma_err_fits]
    
    ## Continuum computation
    cont_col = table[f'{emline}_continuum'].data
    if (np.all(np.isclose(cont_col, 0.0))):
        cont = 0.0
        cont_err = 0.0
    else:
        _, cont, cont_err = sigma_clipped_stats(cont_col)
        
    params[f'{emline}_continuum'] = [cont]
    params[f'{emline}_continuum_err'] = [cont_err]
    
    ## Noise computation
    noise_col = table[f'{emline}_noise'].data
    
    if (np.all(np.isclose(noise_col, 0.0))):
        noise = 0.0
    else:
        noise = np.median(noise_col)
            
    params[f'{emline}_noise'] = [noise]
        
    return (params)

###################################################################################################
###################################################################################################

class get_allfit_params:
    """
    Functions to get all the parameters together.
        1) normal_fit(fits, lam, flam)
        2) extreme_fit(fits, lam, flam)
    """
    
    def normal_fit(fits, lam, flam):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] fits.
        This is for the normal source fitting method.
        
        Parameters
        ----------
        fits : list
            List of [Hb, [OIII], [NII]+Ha, [SII]] bestfits
            
        lam : array
            Wavelength array of the spectra
            
        flam : array
            Flux array of the spectra
            
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb fit, 
            followed by continuum and noise measurements.
            
        oiii_params : dict
            Gaussian parameters of the [OIII] fit, 
            followed by continuum and noise measurements.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha fit, 
            followed by continuum and noise measurements.
            
        sii_params : dict
            Gaussian parameters of the [SII] fit,
            followed by continuum and noise measurements.
        """
        
        
        gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii = fits

        ## Parameters for the fit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_parameters(gfit_hb, hb_models)
        oiii_params = get_parameters(gfit_oiii, oiii_models)
        nii_ha_params = get_parameters(gfit_nii_ha, nii_ha_models)
        sii_params = get_parameters(gfit_sii, sii_models)

        ## Continuum
        hb_params['hb_continuum'] = [gfit_hb['hb_cont'].amplitude.value]
        oiii_params['oiii_continuum'] = [gfit_oiii['oiii_cont'].amplitude.value]
        nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha['nii_ha_cont'].amplitude.value]
        sii_params['sii_continuum'] = [gfit_sii['sii_cont'].amplitude.value]

        ## NOISE
        hb_noise = mfit.compute_noise_emline(lam, flam, 'hb')
        oiii_noise = mfit.compute_noise_emline(lam, flam, 'oiii')
        nii_ha_noise = mfit.compute_noise_emline(lam, flam, 'nii_ha')
        sii_noise = mfit.compute_noise_emline(lam, flam, 'sii')

        hb_params['hb_noise'] = [hb_noise]
        oiii_params['oiii_noise'] = [oiii_noise]
        nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
        sii_params['sii_noise'] = [sii_noise]

        return (hb_params, oiii_params, nii_ha_params, sii_params)
    
###################################################################################################

    def extreme_fit(fits, lam, flam):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] fits.
        This is for the extreme broad-line source fitting.
        
        Parameters
        ----------
        fits : list
            List of [Hb+[OIII], [NII]+Ha+[SII]] bestfits
            
        lam : array
            Wavelength array of the spectra
            
        flam : array
            Flux array of the spectra
            
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb fit, 
            followed by continuum and noise measurements.
            
        oiii_params : dict
            Gaussian parameters of the [OIII] fit, 
            followed by continuum and noise measurements.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha fit, 
            followed by continuum and noise measurements.
            
        sii_params : dict
            Gaussian parameters of the [SII] fit,
            followed by continuum and noise measurements.
        """
        
        gfit_hb_oiii, gfit_nii_ha_sii = fits

        ## Parameters for the fit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_parameters(gfit_hb_oiii, hb_models)
        oiii_params = get_parameters(gfit_hb_oiii, oiii_models)
        nii_ha_params = get_parameters(gfit_nii_ha_sii, nii_ha_models)
        sii_params = get_parameters(gfit_nii_ha_sii, sii_models)

        ## Continuum
        hb_params['hb_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
        oiii_params['oiii_continuum'] = [gfit_hb_oiii['hb_oiii_cont'].amplitude.value]
        nii_ha_params['nii_ha_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]
        sii_params['sii_continuum'] = [gfit_nii_ha_sii['nii_ha_sii_cont'].amplitude.value]

        ## NOISE
        hb_noise = mfit.compute_noise_emline(lam, flam, 'hb')
        oiii_noise = mfit.compute_noise_emline(lam, flam, 'oiii')
        nii_ha_noise = mfit.compute_noise_emline(lam, flam, 'nii_ha')
        sii_noise = mfit.compute_noise_emline(lam, flam, 'sii')

        hb_params['hb_noise'] = [hb_noise]
        oiii_params['oiii_noise'] = [oiii_noise]
        nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
        sii_params['sii_noise'] = [sii_noise]

        return (hb_params, oiii_params, nii_ha_params, sii_params)

###################################################################################################
###################################################################################################

class get_allbestfit_params:
    """
    Functions to get all the parameters for the bestfit.
        1) normal_fit(t_fits, ndofs_list, lam_rest, flam_rest, ivar_rest)
        2) extreme_fit(t_fits, ndofs_list, lam_rest, flam_rest, ivar_rest)
    """
    
    def normal_fit(t_fits, ndofs_list, lam_rest, flam_rest, ivar_rest):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] 
        bestfits from the table of parameters of iterations.
        This is for the normal source fitting method.
        
        Parameters
        ----------
        t_fits : Astropy Table
            Table of fit parameters of all the iterations
            
        ndofs_list : List
            List of N(DOFs) for the fits
            
        lam_rest : numpy array
            Rest-Frame Wavelength array of the spectra
            
        flam_rest : numpy array
            Rest-Frame Flux array of the spectra
            
        ivar_rest : numpy array
            Rest-Frame Inverse Variance array of the spectra
        
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb bestfit, 
            followed by NDOF and reduced chi2.
            
            
        oiii_params : dict
            Gaussian parameters of the [OIII] bestfit, 
            followed by NDOF and reduced chi2.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha bestfit, 
            followed by NDOF and reduced chi2.
            
        sii_params : dict
            Gaussian parameters of the [SII] bestfit, 
            followed by NDOF and reduced chi2.
        """
    
        ## Parameters for the bestfit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_bestfit_parameters(t_fits, hb_models, 'hb')
        oiii_params = get_bestfit_parameters(t_fits, oiii_models, 'oiii')
        nii_ha_params = get_bestfit_parameters(t_fits, nii_ha_models, 'nii_ha')
        sii_params = get_bestfit_parameters(t_fits, sii_models, 'sii')
        
        ## Join into a table
        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)
        
        for col in t_params.colnames:
            t_params.rename_column(col, col.upper())
        
        ## N(DOF) of the different fits
        ndof_hb, ndof_oiii, ndof_nii_ha, ndof_sii = ndofs_list
        
        ## Gaussian Fits
        gfit_hb, gfit_oiii, \
        gfit_nii_ha, gfit_sii = emfit.construct_fits_from_table.normal_fit(t_params, 0)

        ## Reduced chi2 computation
        lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                             ivar_rest, em_line = 'hb')
        lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                   ivar_rest, em_line = 'oiii')
        lam_nii_ha, flam_nii_ha, \
        ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                ivar_rest, em_line = 'nii_ha')
        lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                ivar_rest, em_line = 'sii')
        
        
        
        rchi2_hb = mfit.calculate_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, ndof_hb, \
                                      reduced_chi2 = True)
        rchi2_oiii = mfit.calculate_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, ndof_oiii, \
                                        reduced_chi2 = True)
        rchi2_nii_ha = mfit.calculate_chi2(flam_nii_ha, gfit_nii_ha(lam_nii_ha), ivar_nii_ha, \
                                          ndof_nii_ha, reduced_chi2 = True)
        rchi2_sii = mfit.calculate_chi2(flam_sii, gfit_sii(lam_sii), ivar_sii, ndof_sii, \
                                       reduced_chi2 = True)
        
        ## Add to the params dictionary
        hb_params['hb_ndof'] = [ndof_hb]
        hb_params['hb_rchi2'] = [rchi2_hb]
        
        oiii_params['oiii_ndof'] = [ndof_oiii]
        oiii_params['oiii_rchi2'] = [rchi2_oiii]
        
        nii_ha_params['nii_ha_ndof'] = [ndof_nii_ha]
        nii_ha_params['nii_ha_rchi2'] = [rchi2_nii_ha]
        
        sii_params['sii_ndof'] = [ndof_sii]
        sii_params['sii_rchi2'] = [rchi2_sii]
        
        ## Extreme BL columns
        oiii_params['hb_oiii_ndof'] = [int(0)]
        oiii_params['hb_oiii_rchi2'] = [0.0]
        
        sii_params['nii_ha_sii_ndof'] = [int(0)]
        sii_params['nii_ha_sii_rchi2'] = [0.0]
        
        return (hb_params, oiii_params, nii_ha_params, sii_params)
    
###################################################################################################

    def extreme_fit(t_fits, ndofs_list, lam_rest, flam_rest, ivar_rest):
        """
        Function to get all the required parameters for the Hb, [OIII], [NII]+Ha, and [SII] 
        bestfits from the table of parameters of iterations.
        This is for the extreme BL source fitting method.
        
        Parameters
        ----------
        t_fits : Astropy Table
            Table of fit parameters of all the iterations
            
        ndofs_list : List
            List of N(DOFs) for the fits
            
        lam_rest : numpy array
            Rest-Frame Wavelength array of the spectra
            
        flam_rest : numpy array
            Rest-Frame Flux array of the spectra
            
        ivar_rest : numpy array
            Rest-Frame Inverse Variance array of the spectra
        
        Returns
        -------
        hb_params : dict
            Gaussian parameters of the Hb bestfit, 
            followed by NDOF and reduced chi2.
            
            
        oiii_params : dict
            Gaussian parameters of the [OIII] bestfit, 
            followed by NDOF and reduced chi2.
            
        nii_ha_params : dict
            Gaussian parameters of the [NII]+Ha bestfit, 
            followed by NDOF and reduced chi2.
            
        sii_params : dict
            Gaussian parameters of the [SII] bestfit, 
            followed by NDOF and reduced chi2.
        """
    
        ## Parameters for the bestfit
        hb_models = ['hb_n', 'hb_out', 'hb_b']
        oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
        nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', \
                         'ha_n', 'ha_out', 'ha_b']
        sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

        hb_params = get_bestfit_parameters(t_fits, hb_models, 'hb')
        oiii_params = get_bestfit_parameters(t_fits, oiii_models, 'oiii')
        nii_ha_params = get_bestfit_parameters(t_fits, nii_ha_models, 'nii_ha')
        sii_params = get_bestfit_parameters(t_fits, sii_models, 'sii')
        
        ## Join into a table
        t_params = Table(hb_params|oiii_params|nii_ha_params|sii_params)
        
        for col in t_params.colnames:
            t_params.rename_column(col, col.upper())
        
        ## N(DOF) of the different fits
        ndof_hb_oiii, ndof_nii_ha_sii = ndofs_list
        
        ## Gaussian Fits
        gfit_hb_oiii, gfit_nii_ha_sii = emfit.construct_fits_from_table.extreme_fit(t_params, 0)

        ## Reduced chi2 computation
        lam_hb_oiii, flam_hb_oiii,\
        ivar_hb_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest, 'hb_oiii')
        lam_nii_ha_sii, flam_nii_ha_sii, \
        ivar_nii_ha_sii = spec_utils.get_fit_window(lam_rest, flam_rest, ivar_rest, 'nii_ha_sii')
        
        gfit_hb_oiii, gfit_nii_ha_params = emfit.construct_fits_from_table.extreme_fit(t_params, 0)
        
        rchi2_hb_oiii = mfit.calculate_chi2(flam_hb_oiii, gfit_hb_oiii(lam_hb_oiii), \
                                            ivar_hb_oiii, ndof_hb_oiii, \
                                            reduced_chi2 = True)
        
        rchi2_nii_ha_sii = mfit.calculate_chi2(flam_nii_ha_sii, gfit_nii_ha_sii(lam_nii_ha_sii), \
                                               ivar_nii_ha_sii, ndof_nii_ha_sii, \
                                               reduced_chi2 = True)
        
        ## Normal columns
        hb_params['hb_ndof'] = [int(0)]
        hb_params['hb_rchi2'] = [0.0]
        
        oiii_params['oiii_ndof'] = [int(0)]
        oiii_params['oiii_rchi2'] = [0.0]
        
        nii_ha_params['nii_ha_ndof'] = [int(0)]
        nii_ha_params['nii_ha_rchi2'] = [0.0]
        
        sii_params['sii_ndof'] = [int(0)]
        sii_params['sii_rchi2'] = [0.0]
        
        ## Extreme BL columns
        oiii_params['hb_oiii_ndof'] = [ndof_hb_oiii]
        oiii_params['hb_oiii_rchi2'] = [rchi2_hb_oiii]
        
        sii_params['nii_ha_sii_ndof'] = [ndof_nii_ha_sii]
        sii_params['nii_ha_sii_rchi2'] = [rchi2_nii_ha_sii]
        
        return (hb_params, oiii_params, nii_ha_params, sii_params)
    
###################################################################################################
###################################################################################################
    
    


        
    
                     
        
        
        
    
        
        
        
        
    
    
    
    
        


    
    
    
        
        
        
