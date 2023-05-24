"""
This script consists of spectra-related utility functions.
The following functions are available:
    1) find_coadded_spectra(specprod, survey, program, healpix, targetid)
    2) find_stellar_continuum(specprod, survey, program, healpix, targetid)
    3) get_emline_spectra(specprod, survey, program, healpix, targetid, \
                          z, rest_frame = False, plot_continuum = False)
    4) get_fit_window(lam_rest, flam_rest, ivar_rest, em_line)

Author : Ragadeepika Pucha
Version : 2023, May 22
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
    This function finds the coadded spectra of a given target and corrects for MW transmission.
    
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
        
    Returns
    -------
    coadd_spec : obj
        Coadded Spectra object associated with the target
    """
    
    ## Targets healpix directory
    hpx_dir =  f'/global/cfs/cdirs/desi/spectro/redux/{specprod}/healpix'
    ## Specific healpix directory of the target
    target_dir = f'{survey}/{program}/{healpix//100}/{healpix}'
    ## Coadded data file directory of the target
    coadd_dir = f'{hpx_dir}/{target_dir}'
    ## Coadded file name
    coadd_file = f'{coadd_dir}/coadd-{survey}-{program}-{healpix}.fits'
    
    ## Get spectra
    spec = read_spectra(coadd_file).select(targets = targetid)
    ## Coadd the spectra across cameras
    coadd_spec = coadd_cameras(spec)
    
    return (coadd_spec)

###################################################################################################

def find_stellar_continuum(specprod, survey, program, healpix, targetid):
    """
    This function finds the fastspecfit stellar continuum for a given spectra.
    
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
        
    Returns
    -------
    modelwave : numpy array
        Model wavelength array
        
    total_cont : numpy array
        Total continuum array including stellar continuum + smooth continuum models    
    """
    
    ## Fastspecfit healpix directory
    fastspec_dir = f'/global/cfs/cdirs/desi/spectro/fastspecfit/{specprod}/v2.0/healpix'
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

    ## Total continuum model
    total_cont = cont_model + smooth_cont_model
    
    return (modelwave, total_cont)

###################################################################################################

def get_emline_spectra(specprod, survey, program, healpix, targetid,\
                       z = None, rest_frame = False, plot_continuum = False):
    """
    This function finds the coadded spectra and stellar continuum model of a given target and
    returns the continuum-subtracted emission-line spectra
    
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
    lam : numpy array
        Wavelength array of the spectra. Rest-frame values if rest_frame = True.
        
    emline_spec : numpy array
        Continuum subtracted spectra array. Rest-frame values if rest_frame = True.
        
    ivar : numpy array
        Inverse variance array of the spectra. Rest-frame values if rest_frame = True.
        
    res_matrix : obj
        Resolution Matrix Object
    """
    
    ## Coadded Spectra
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
    res_matrix = coadd_spec.R[bands][0]
    
    ## Stellar continuum model
    modelwave, total_cont = find_stellar_continuum(specprod, survey, program, healpix, targetid)
    
    ## Subtract the continuum from the flux
    emline_spec = flam - total_cont

    if (rest_frame == True)&(z is not None):
        lam = lam/(1+z)
        emline_spec = emline_spec*(1+z)
        ivar = ivar/((1+z)**2)

    if (plot_continuum == True):
        plot_utils.plot_spectra_continuum(lam, flam, total_cont)
        
    return (lam, emline_spec, ivar, res_matrix)

###################################################################################################

def get_fit_window(lam_rest, flam_rest, ivar_rest, em_line):
    """
    Function to return the fitting windows for the different emission-lines.
    Only for Hb, [OIII], [NII]+Ha and [SII].
    
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
        lam_ii = (lam_rest >= 6350)&(lam_rest <= 6700)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    elif (em_line == 'sii'):
        lam_ii = (lam_rest >= 6650)&(lam_rest <= 6900)
        lam_win = lam_rest[lam_ii]
        flam_win = flam_rest[lam_ii]
        ivar_win = ivar_rest[lam_ii]
    else:
        raise NameError('Emission-line not available!')
        
    return (lam_win, flam_win, ivar_win)

####################################################################################################