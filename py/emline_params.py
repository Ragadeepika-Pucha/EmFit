"""
This script consists of functions for computing the parameters of the emission-line fits.
It consists of the following functions:

Author : Ragadeepika Pucha
Version : 2024, February 8
"""

###################################################################################################
from astropy.table import Table
import numpy as np

import measure_fits as mfit
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

# def compute_final_rchi2(lam, flam, ivar, fits, dof):
    
#     gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii = fits
#     hb_dof, oiii_dof, nii_ha_dof, sii_dof = dof
    
#     ## Fitting windows for the different emission-lines.
    
#     lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam, flam, \
#                                                          ivar, em_line = 'hb')

#     lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam, flam, \
#                                                                ivar, em_line = 'oiii')
#     lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam, flam, \
#                                                                      ivar, em_line = 'nii_ha')
#     lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam, flam, \
#                                                             ivar, em_line = 'sii') 
    
    
        
#     hb_rchi2 = mfit.calculate_red_chi2(flam_hb, gfit_hb(lam_hb), ivar_hb, hb_dof)
#     oiii_rchi2 = mfit.calculate_red_chi2(flam_oiii, gfit_oiii(lam_oiii), ivar_oiii, oiii_dof)
#     nii_ha_rchi2 = mfit.calculate_red_chi2(flam_nii_ha, gfit_nii_ha(lam_nii_ha), ivar_nii_ha, nii_ha_dof)
#     sii_rchi2 = mfit.calculate_red_chi2(flam_sii, gfit_sii(lam_sii), ivar_sii, sii_dof)
    
#     return (hb_rchi2, oiii_rchi2, nii_ha_rchi2, sii_rchi2)

# ###################################################################################################


# def get_bestfit_parameters(table, models, emline):
#     """
#     Get the bestfit parameters from iterations of fits
    
#     Parameters
#     ----------
#     table : Astropy Table
#         Table of gaussian fits are different iterations
    
#     models : list
#         List of total submodels expected from a given emission-line fitting.
        
#     Returns
#     -------
#     params : dict
#         Dictionary with the parameter values
#     """
    
#     params = {}
    
#     for model in models:
#         amplitude_arr = table[f'{model}_amplitude'].data
#         mean_arr = table[f'{model}_mean'].data
#         std_arr = table[f'{model}_std'].data
#         flux_arr = table[f'{model}_flux'].data
#         sigma_arr = table[f'{model}_sigma'].data
        
#         amp_zero = (np.all(np.isclose(amplitude_arr, 0.0)))
#         mean_zero = (np.all(np.isclose(mean_arr, 0.0)))
#         std_zero = (np.all(np.isclose(std_arr, 0.0)))
        
#         allzero = amp_zero|mean_zero|std_zero
        
#         if (allzero):
#             amp = 0.0
#             amp_err = 0.0
#             mean = 0.0
#             mean_err = 0.0
#             std = 0.0
#             std_err = 0.0
#             flux = 0.0
#             flux_err = 0.0
#             sigma = 0.0
#             sigma_err = 0.0
#             flux_fits = 0.0
#             flux_err_fits = 0.0
#             sigma_fits = 0.0
#             sigma_err_fits = 0.0
#         else:
#             cond = (amplitude_arr > 0)&(mean_arr > 0)&(std_arr > 0)
            
#             amp = np.nanmean(np.where(~cond, np.nan, amplitude_arr))
#             amp_err = np.nanstd(np.where(~cond, np.nan, amplitude_arr))
#             mean = np.nanmean(np.where(~cond, np.nan, mean_arr))
#             mean_err = np.nanstd(np.where(~cond, np.nan, mean_arr))
#             var = np.where(~cond, np.nan, std_arr)**2
#             std = np.sqrt(np.nanmean(var))
#             std_err = np.sqrt(np.nanstd(var))
#             flux, flux_err = mfit.compute_emline_flux(amp, std, amp_err, std_err)
#             sigma, sigma_err = mfit.lamspace_to_velspace(std, mean, std_err, mean_err)
#             flux_fits = np.nanmean(np.where(~cond, np.nan, flux_arr))
#             flux_err_fits = np.nanstd(np.where(~cond, np.nan, flux_arr))
#             sigma_fits = np.nanmean(np.where(~cond, np.nan, sigma_arr))
#             sigma_err_fits = np.nanstd(np.where(~cond, np.nan, sigma_arr))
            
#         params[f'{model}_amplitude'] = [amp]
#         params[f'{model}_amplitude_err'] = [amp_err]
#         params[f'{model}_mean'] = [mean]
#         params[f'{model}_mean_err'] = [mean_err]
#         params[f'{model}_std'] = [std]
#         params[f'{model}_std_err'] = [std_err]
#         params[f'{model}_sigma'] = [sigma]
#         params[f'{model}_sigma_err'] = [sigma_err]
#         params[f'{model}_flux'] = [flux]
#         params[f'{model}_flux_err'] = [flux_err]
#         params[f'{model}_flux_fits'] = [flux_fits]
#         params[f'{model}_flux_err_fits'] = [flux_err_fits]
#         params[f'{model}_sigma_fits'] = [sigma_fits]
#         params[f'{model}_sigma_err_fits'] = [sigma_err_fits]
        
#         ## Continuum computation 
#         cont_col = table[f'{emline}_continuum'].data
#         if (np.all(np.isclose(cont_col, 0.0))):
#             cont = 0.0
#             cont_err = 0.0
#         else:
#             cont = np.nanmean(np.where(np.isclose(cont_col, 0.0), np.nan, cont_col))
#             cont_err = np.nanstd(np.where(np.isclose(cont_col, 0.0), np.nan, cont_col))
            
#         params[f'{emline}_continuum'] = [cont]
#         params[f'{emline}_continuum_err'] = [cont_err]
        
#     return (params)

# ####################################################################################################

# def fix_params(table):
#     emlines = ['hb', 'nii6548', 'nii6583', 'ha']
    
#     for emline in emlines:
#         if ((emline == 'hb')|(emline == 'ha')):
#             amp_n = table[f'{emline}_n_amplitude'].data[0]
#             mean_n = table[f'{emline}_n_mean'].data[0]
#             std_n = table[f'{emline}_n_std'].data[0]
#             flux_n = table[f'{emline}_n_flux'].data[0]
#             sig_n = table[f'{emline}_n_sigma'].data[0]
#         else:
#             amp_n = table[f'{emline}_amplitude'].data[0]
#             mean_n = table[f'{emline}_mean'].data[0]
#             std_n = table[f'{emline}_std'].data[0]
#             flux_n = table[f'{emline}_flux'].data[0]
#             sig_n = table[f'{emline}_sigma'].data[0]
        
#         amp_out = table[f'{emline}_out_amplitude'].data
#         mean_out = table[f'{emline}_out_mean'].data[0]
#         std_out = table[f'{emline}_out_std'].data[0]
#         flux_out = table[f'{emline}_out_flux'].data[0]
#         sig_out = table[f'{emline}_out_sigma'].data[0]
        
#         if ((amp_n == 0)&(amp_out != 0)):
#             if ((emline == 'hb')|(emline == 'ha')):
#                 table[f'{emline}_n_amplitude'][0] = amp_out
#                 table[f'{emline}_n_mean'][0] = mean_out
#                 table[f'{emline}_n_std'][0] = std_out
#                 table[f'{emline}_n_flux'][0] = flux_out
#                 table[f'{emline}_n_sigma'][0] = sig_out
#             else:
#                 table[f'{emline}_amplitude'][0] = amp_out
#                 table[f'{emline}_mean'][0] = mean_out
#                 table[f'{emline}_std'][0] = std_out
#                 table[f'{emline}_flux'][0] = flux_out
#                 table[f'{emline}_sigma'][0] = sig_out
            
#             table[f'{emline}_out_amplitude'][0] = amp_n
#             table[f'{emline}_out_mean'][0] = mean_n
#             table[f'{emline}_out_std'][0] = std_n
#             table[f'{emline}_out_flux'][0] = flux_n
#             table[f'{emline}_out_sigma'][0] = sig_n
            
    
#     return (table)

# ####################################################################################################

# def calculate_emline_noise(specprod, survey, program, healpix, targetid, z):
    
#     lam_rest, flam_rest, ivar_rest, _ = spec_utils.get_emline_spectra(specprod, survey, program, \
#                                                                    healpix, targetid, z, \
#                                                                    rest_frame = True, \
#                                                                    plot_continuum = False)
    
#     hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, em_line = 'hb')
#     oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, em_line = 'oiii')
#     nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, em_line = 'nii_ha')
#     sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, em_line = 'sii')
    
#     params = {}
#     params['TARGETID'] = [targetid]
#     params['HB_NOISE'] = [hb_noise]
#     params['OIII_NOISE'] = [oiii_noise]
#     params['NII_HA_NOISE'] = [nii_ha_noise]
#     params['SII_NOISE'] = [sii_noise]
    
#     t_params = Table(params)
    
#     #t_params.write(f'output/single_files/emfit-noise-{targetid}.fits')
    
#     return (t_params)
    
# ####################################################################################################

        
    
                     
        
        
        
    
        
        
        
        
    
    
    
    
        


    
    
    
        
        
        
