"""
This script consists of functions related to fitting the emission line spectra, 
and plotting the models and residuals.

Author : Ragadeepika Pucha
Version : 2023, May 8
"""

####################################################################################################

import numpy as np

from astropy.table import Table

import fit_utils, spec_utils, plot_utils
import fit_lines
import measure_fits as mfit
import emline_params as emp
import find_bestfit

import matplotlib.pyplot as plt

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

def fit_emline_spectra(specprod, survey, program, healpix, targetid, z):
    """
    Fit [SII], Hb, [OIII], [NII]+Ha emission lines for a given emission line spectra.
    
    Parameters
    ----------
    specprod : str
        Spectral Production Pipeline name fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : str
        Healpix number of the target
        
    targetid : int64
        The unique TARGETID associated with the target
        
    z : float
        Redshift of the target

    Returns
    -------
    t_params : astropy table
        Table of output parameters for the fit
    """
    
    lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                                   healpix, targetid, z, \
                                                                   rest_frame = True, \
                                                                   plot_continuum = False)

    lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                         ivar_rest, em_line = 'hb')

    lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                               ivar_rest, em_line = 'oiii')
    lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                     ivar_rest, em_line = 'nii_ha')
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                            ivar_rest, em_line = 'sii')
    
    gfit_sii, rchi2_sii, sii_bits, sii_delrchi2 = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
    gfit_oiii, rchi2_oiii, oiii_bits, oiii_delrchi2 = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)
    gfit_hb, rchi2_hb, hb_bits, hb_delrchi2 = find_bestfit.find_hb_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii)
    gfit_nii_ha, rchi2_nii_ha, nii_ha_bits, nii_ha_delrchi2 = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
                                                                               ivar_nii_ha, gfit_sii, ver = 'v1')
        
    hb_models = ['hb_n', 'hb_out', 'hb_b']
    oiii_models = ['oiii4959', 'oiii4959_out', 'oiii5007', 'oiii5007_out']
    nii_ha_models = ['nii6548', 'nii6548_out', 'nii6583', 'nii6583_out', 'ha_n', 'ha_out', 'ha_b']
    sii_models = ['sii6716', 'sii6716_out', 'sii6731', 'sii6731_out']

    hb_params = emp.get_parameters(gfit_hb, hb_models)
    oiii_params = emp.get_parameters(gfit_oiii, oiii_models)
    nii_ha_params = emp.get_parameters(gfit_nii_ha, nii_ha_models)
    sii_params = emp.get_parameters(gfit_sii, sii_models)
    
    hb_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                         gfit_hb, em_line = 'hb')
    oiii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                           gfit_oiii, em_line = 'oiii')
    nii_ha_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                             gfit_nii_ha, em_line = 'nii_ha')
    sii_noise = mfit.compute_noise_emline(lam_rest, flam_rest, \
                                          gfit_sii, em_line = 'sii')
    
    sii_flag = sum(2**sii_bits)
    oiii_flag = sum(2**oiii_bits)
    hb_flag = sum(2**hb_bits)
    nii_ha_flag = sum(2**nii_ha_bits)
    
    hb_params['hb_noise'] = [hb_noise]
    oiii_params['oiii_noise'] = [oiii_noise]
    nii_ha_params['nii_ha_noise'] = [nii_ha_noise]
    sii_params['sii_noise'] = [sii_noise]
    
    hb_params['hb_rchi2'] = [rchi2_hb]
    oiii_params['oiii_rchi2'] = [rchi2_oiii]
    nii_ha_params['nii_ha_rchi2'] = [rchi2_nii_ha]
    sii_params['sii_rchi2'] = [rchi2_sii]
    
    hb_params['hb_flag'] = [hb_flag]
    oiii_params['oiii_flag'] = [oiii_flag]
    nii_ha_params['nii_ha_flag'] = [nii_ha_flag]
    sii_params['sii_flag'] = [sii_flag]
    
    hb_params['hb_delta_rchi2'] = [hb_delrchi2]
    oiii_params['oiii_delta_rchi2'] = [oiii_delrchi2]
    nii_ha_params['nii_ha_delta_rchi2'] = [nii_ha_delrchi2]
    sii_params['sii_delta_rchi2'] = [sii_delrchi2]
    
    tgt = {}
    tgt['targetid'] = [targetid]
    tgt['specprod'] = [specprod]
    tgt['survey'] = [survey]
    tgt['program'] = [program]
    tgt['healpix'] = [healpix]
    tgt['z'] = [z]
    
    params = tgt|hb_params|oiii_params|nii_ha_params|sii_params    
    
    ## Convert dictionary to table
    t_params = Table(params)
    
    fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
    rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    return (t_params)

####################################################################################################
    
def emline_fit_spectra(specprod, survey, program, healpix, targetid, z):
    
    
    ## Get rest-frame emission-line spectra
    lam_rest, flam_rest, ivar_rest = fit_utils.get_emline_spectra(specprod, survey, \
                                                                  program, healpix, targetid,\
                                                                  z, rest_frame = True)
    
    ## Get the fits
    fits, rchi2s = fit_emline_spectra(lam_rest, flam_rest, ivar_rest)
    
    ## Plot the fits
    plot_spectra_fits(targetid, lam_rest, flam_rest, fits, rchi2s)
    
    return (fits, rchi2s)
    
####################################################################################################

def check_fits(table, index):
    specprod = table['SPECPROD'].astype(str).data[index]
    targetid = table['TARGETID'].data[index]
    survey = table['SURVEY'].astype(str).data[index]
    program = table['PROGRAM'].astype(str).data[index]
    healpix = table['HEALPIX'].data[index]
    z = table['Z'].data[index]
    logmass = table['logM'].data[index]
    
    if ('Version_NII_Ha' in table.colnames):
        version = table['Version_NII_Ha'].astype(str).data[index]

        if (version == 'both'):
            ver = 'v1'
        else:
            ver = version
    else:
        ver = 'v2'
    
    
    lam_rest, flam_rest, ivar_rest = spec_utils.get_emline_spectra(specprod, survey, program, \
                                                                   healpix, targetid, z, \
                                                                   rest_frame = True, \
                                                                   plot_continuum = False)

    lam_hb, flam_hb, ivar_hb = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                         ivar_rest, em_line = 'hb')

    lam_oiii, flam_oiii, ivar_oiii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                               ivar_rest, em_line = 'oiii')
    lam_nii_ha, flam_nii_ha, ivar_nii_ha = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                                     ivar_rest, em_line = 'nii_ha')
    lam_sii, flam_sii, ivar_sii = spec_utils.get_fit_window(lam_rest, flam_rest, \
                                                            ivar_rest, em_line = 'sii')
    
    gfit_sii, rchi2_sii, sii_bits, _ = find_bestfit.find_sii_best_fit(lam_sii, flam_sii, ivar_sii)
    gfit_oiii, rchi2_oiii, oiii_bits, _ = find_bestfit.find_oiii_best_fit(lam_oiii, flam_oiii, ivar_oiii)
    gfit_hb, rchi2_hb, hb_bits, _ = find_bestfit.find_hb_best_fit(lam_hb, flam_hb, ivar_hb, gfit_sii)
    gfit_nii_ha, rchi2_nii_ha, nii_ha_bits, _ = find_bestfit.find_nii_ha_best_fit(lam_nii_ha, flam_nii_ha, \
                                                                               ivar_nii_ha, gfit_sii, ver = ver)
    
    fits = [gfit_hb, gfit_oiii, gfit_nii_ha, gfit_sii]
    rchi2s = [rchi2_hb, rchi2_oiii, rchi2_nii_ha, rchi2_sii]
    
    title = f'TARGETID: {targetid}; z: {round(z, 3)}; logmass: {round(logmass, 2)}\n'+ \
    f'https://www.legacysurvey.org/viewer-desi/desi-spectrum/daily/targetid{targetid}'
    
    fig = plot_utils.plot_spectra_fits(lam_rest, flam_rest, fits, rchi2s, title = title)
    
    return (fig)    

####################################################################################################