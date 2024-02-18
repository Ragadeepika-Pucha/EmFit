"""
This script consists of spectra-related utility functions.
The following functions are available:
    1) find_coadded_spectra(specprod, survey, program, healpix, targetid)
    2) find_fastspec_models(specprod, survey, program, healpix, targetid, ver)
    3) get_emline_spectra(specprod, survey, program, healpix, targetid, \
                          z, rest_frame = False, plot_continuum = False)
    4) get_fit_window(lam_rest, flam_rest, ivar_rest, em_line)
    5) compute_resolution_sigma(coadd_spec)

Author : Ragadeepika Pucha
Version : 2024, Jan 30th
"""
###################################################################################################

import numpy as np

from astropy.table import Table
import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

import plot_utils

###################################################################################################

def find_coadded_spectra(specprod, survey, program, healpix, targetid):
    """
    This function finds the coadded spectra of a given target and returns the spectra that is 
    coadded across cameras.
    
    Parameters
    ----------
    specprod : str
        Spectral Production Pipeline name 
        fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : str
        Healpix number of the target
        
    targetid : int64
        The unique TARGETID associated with the target
        
    Returns
    -------
    coadd_spec : obj
        Coadded Spectra object (coadded across cameras) associated with the target
    """
    
    ## Targets healpix directory
    ## Since it is read only, changed the directory to dvs_ro/cfs/..
    hpx_dir =  f'/dvs_ro/cfs/cdirs/desi/spectro/redux/{specprod}/healpix'
    ## Specific healpix directory of the target
    target_dir = f'{survey}/{program}/{healpix//100}/{healpix}'
    ## Coadded data file directory of the target
    coadd_dir = f'{hpx_dir}/{target_dir}'
    ## Coadded file name
    coadd_file = f'{coadd_dir}/coadd-{survey}-{program}-{healpix}.fits'
    
    ## Get spectra
    ## Skipping HDUs that are not required for optimization 
    ## This is not working for DESI 22.5 
    ## Might update it later
    ## MASK and RESOLUTION hdus are needed
    # spec = read_spectra(coadd_file, \
    #                     skip_hdus = ('EXP_FIBERMAP', 'SCORES', \
    #                                  'EXTRA_CATALOG')).select(targets = targetid)
    
    ## Get spectra
    spec = read_spectra(coadd_file).select(targets = targetid)
    
    ## Coadd the spectra across cameras
    coadd_spec = coadd_cameras(spec)
    
    return (coadd_spec)

###################################################################################################

def find_fastspec_models(specprod, survey, program, healpix, targetid, fspec = False):
    
    """
    This function finds and returns the fastspecfit models for a given spectra.
    The version depends on the "specprod"
    
    Parameters 
    ----------
    specprod : str
        Spectral Production Pipeline name 
        fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : str
        Healpix number of the target
        
    targetid : int64
        The unique TARGETID associated with the target
         
    fspec : bool
        Whether or not to return the fastspecfit measurements row.
        Default is False

    Returns
    -------
    modelwave : numpy array
        Model wavelength array
        
    total_cont : numpy array
        Total continuum array including stellar continuum + smooth continuum models
        
    em_model : numpy array
        Emission-line model array
        
    fspec_row : Astropy row
        Fastspecfit measurements of the target. 
        Returned only if fspec = True
        
    """
    
    # ver : str
    #     Version of the fastspecfit. Default is v3.2
    #     Latest Fuji version: v3.2
    #     Latest Guadalupe version: v3.1
    #     Latest Iron version: v2.1
    
    if (specprod == 'fuji'):
        ver = 'v3.2'
    elif (specprod == 'guadalupe'):
        ver = 'v3.1'
    elif (specprod == 'iron'):
        ver = 'v2.1'

    ## Fastspecfit healpix directory
    ## Read-only file - using /dvs_ro/cfs/.. directory
    fastspec_dir = f'/dvs_ro/cfs/cdirs/desi/spectro/fastspecfit/{specprod}/{ver}/healpix'
    ## Specific healpix directory of the target
    target_dir = f'{survey}/{program}/{healpix//100}/{healpix}'
    ## Fastspecfit file directory of the target
    target_fast_dir = f'{fastspec_dir}/{target_dir}'
    ## Fastspecfit data file associated with the target
    fastfile = f'{target_fast_dir}/fastspec-{survey}-{program}-{healpix}.fits.gz'
    
    ## Metadata 
    meta = Table(fitsio.read(fastfile, 'METADATA'))
   
    ## Models
    models, hdr = fitsio.read(fastfile, 'MODELS', header = True)
    
    ## Model wavelength array
    modelwave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1']
    
    ## The specific row of the target
    row = (meta['TARGETID'] == targetid)
    
    ## Model for the target
    model = models[row]
    
    ## Continuum model
    cont_model = model[0,0,:]
    ## Smooth continuum model
    smooth_cont_model = model[0,1,:]
    ## Emission-line model
    em_model = model[0,2,:]

    ## Total continuum model
    total_cont = cont_model + smooth_cont_model
    
    if (fspec == True):
         ## Fastspecfit
        fspec = Table(fitsio.read(fastfile, 'FASTSPEC'))
        tgt = (fspec['TARGETID'] == targetid)
        fspec_row = fspec[tgt]
        
        return (modelwave, total_cont, em_model, fspec_row)

    else:
        return (modelwave, total_cont, em_model)

###################################################################################################

def get_emline_spectra(specprod, survey, program, healpix, targetid, \
                       z = None, rest_frame = False, plot_continuum = False):
    """
    This function finds the coadded spectra and stellar continuum model of a given target and
    returns the continuum-subtracted emission-line spectra
    
    Parameters
    ----------
    specprod : str
        Spectral Production Pipeline name 
        fuji|guadalupe|...
        
    survey : str
        Survey name for the spectra
        
    program : str
        Program name for the spectra
        
    healpix : str
        Healpix number of the target
        
    targetid : int64
        The unique TARGETID associated with the target
        
    z : float
        Redshift of the source. Required only if rest_frame = True
        Default = None
        
    rest_frame : bool
        Whether or not to return the emission-line spectra in the rest-frame.
        Default is False
        
    plot_continuum : bool
        Whether or not to plot spectra+continuum for the given object.
        Default is False
        
    Returns
    -------
    coadd_spec : obj
        Coadded Spectra object (coadded across cameras) associated with the target
    
    lam : numpy array
        Wavelength array of the spectra.
        Rest-frame values if rest_frame = True.
        
    emline_spec : numpy array
        Continuum subtracted spectra array.
        Rest-frame values if rest_frame = True.
        
    ivar : numpy array
        Inverse variance array of the spectra.
        Rest-frame values if rest_frame = True.
        
    """
    
    ## Coadded Spectra of the target
    coadd_spec = find_coadded_spectra(specprod, survey, program, healpix, targetid)
    bands = coadd_spec.bands[0]
    
    ## EBV
    ## Correct for MW Transmission
    ebv = coadd_spec.fibermap['EBV'].data
    
    ## MW Transmission
    mw_trans_spec = dust_transmission(coadd_spec.wave[bands], ebv)
    
    ## Wavelength, flux and inverse variance arrays
    lam = coadd_spec.wave[bands]
    flam = coadd_spec.flux[bands].flatten()/mw_trans_spec
    ivar = coadd_spec.ivar[bands].flatten()
    
    ## Stellar continuum model
    modelwave, total_cont, _ = find_fastspec_models(specprod, survey, program, healpix, targetid)
    
    ## Subtract the continuum from the flux
    ## Emission-line spectra
    emline_spec = flam - total_cont

    if (rest_frame == True)&(z is not None):
        lam = lam/(1+z)
        emline_spec = emline_spec*(1+z)
        ivar = ivar/((1+z)**2)

    if (plot_continuum == True):
        plot_utils.plot_spectra_continuum(lam, flam, total_cont)
        
    return (coadd_spec, lam, emline_spec, ivar)

###################################################################################################

def get_fit_window(lam_rest, flam_rest, ivar_rest, em_line):
    """
    Function to return the fitting windows for the different emission-lines.
    Only works for Hb, [OIII], [NII]+Ha and [SII].
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame wavelength array
        
    flam_rest : numpy array
        Rest-frame flux array
        
    ivar_rest : numpy array
        Rest-frame inverse variance array
        
    em_line : str
        Emission-line(s) which needs to be fit
        'hb' for Hb
        'oiii' for [OIII]
        'nii_ha' for [NII]+Ha
        'sii' for [SII]
        
    Returns
    -------
    lam_win : numpy array
        Wavelength array of the fit window
        
    flam_win : numpy array
        Flux array of the fit window
        
    ivar_win : numpy array
        Inverse variance array of the fit window
    """
    
    if (em_line == 'hb'):
        lam_ii = (lam_rest >= 4700)&(lam_rest <= 4930)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'oiii'):
        lam_ii = (lam_rest >= 4900)&(lam_rest <= 5100)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'nii_ha'):
        lam_ii = (lam_rest >= 6300)&(lam_rest <= 6700)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'sii'):
        lam_ii = (lam_rest >= 6650)&(lam_rest <= 6900)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'nii_ha_sii'):
        lam_ii = (lam_rest >= 6300)&(lam_rest <= 6900)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'hb_oiii'):
        lam_ii = (lam_rest >= 4700)&(lam_rest <= 5100)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    else:
        raise NameError('Emission-line not available!')
        
    return (lam_win, flam_win, ivar_win)

####################################################################################################

def compute_resolution_sigma(coadd_spec):
    """
    Function to compute wavelength-dependent "sigma" of line-spread function
    in the units of wavelength baseline (Angstrom)
    
    Adapted from Adam Bolton's code for SPARCL
    
    Parameters
    ----------
    coadd_spec : obj
        Coadded spectrum (coadded across cameras) of the object
        
    Returns
    -------
    rsigma : numpy array
        Array of 1-d resolution sigma
    
    """
    
    res_matrix = coadd_spec.resolution_data['brz']
    lam = coadd_spec.wave['brz']
    
    ## Dimensionaily of the resolution data array
    nspec, nband, npix = res_matrix.shape
    
    ## Create arrays for calculating Line Spread Function moments
    xband = np.arange(float(nband))
    xband = xband - xband.mean()
    ## Outer Product of two vectors
    xfull = np.outer(xband, np.ones(npix))  
    ## New array of nspecxnpix shape filled with zeros
    rsigma = np.full((nspec, npix), 0.)  
    
    # Loop over spectra to compute dispersion values, initially in units of pixels
    for ispec in range(nspec):
        rnorm = res_matrix[ispec].sum(0)
        rmask = rnorm > 0
        rnorm_inv = np.full(npix, 0.)
        rnorm_inv[rmask] = 1. / rnorm[rmask]
        xres = rnorm_inv * (res_matrix[ispec] * xfull).sum(0)
        x2res = rnorm_inv * (res_matrix[ispec] * xfull**2).sum(0)
        rsigma[ispec] = np.sqrt(np.abs(x2res - xres**2))
        
    # Convert from pixels to delta-wavelength (AA units)
    dwave = 0. * lam
    dwave[1:-1] = 0.5 * (lam[2:] - lam[:-2])
    dwave[0] = dwave[1]
    dwave[-1] = dwave[-2]
    rsigma *= dwave # using numpy broadcasting
    
    return (rsigma)

####################################################################################################