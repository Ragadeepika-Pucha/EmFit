"""
This script consists of utility functions for emission-line fitting related stuff.

Author : Ragadeepika Pucha
Version : 2023, March 14
"""

###################################################################################################

import numpy as np

from astropy.table import Table
import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

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
    fastspec_dir = f'/global/cfs/cdirs/desi/spectro/fastspecfit/{specprod}/v1.0/healpix'
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
    lam : numpy array
        Wavelength array of the spectra
        
    flam : numpy array
        MW transmission corrected flux array of the spectra
        
    ivar : numpy array
        Inverse variance array of the spectra
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
    coadd_spec = coadd_cameras(spec)
    bands = coadd_spec.bands[0]
    
    ## EBV
    ebv = coadd_spec.fibermap['EBV'].data
    
    ## MW Transmittion
    mw_trans_spec = dust_transmission(coadd_spec.wave[bands], ebv)
    
    lam = coadd_spec.wave[bands]
    flam = coadd_spec.flux[bands].flatten()/mw_trans_spec
    ivar = coadd_spec.ivar[bands].flatten()
    
    return (lam, flam, ivar)

###################################################################################################

def get_emline_spectra(specprod, survey, program, healpix, targetid, z = None, rest_frame = False):
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
        
    Returns
    -------
    lam : numpy array
        Wavelength array of the spectra. Rest-frame values if rest_frame = True.
        
    emline_spec : numpy array
        Continuum subtracted spectra array. Rest-frame values if rest_frame = True.
        
    ivar : numpy array
        Inverse variance array of the spectra. Rest-frame values if rest_frame = True.
    """
    
    ## Coadded Spectra
    lam, flam, ivar = find_coadded_spectra(specprod, survey, program, healpix, targetid)
    
    ## Stellar continuum model
    modelwave, total_cont = find_stellar_continuum(specprod, survey, program, healpix, targetid)
    
    ## Subtract the continuum from the flux
    emline_spec = flam - total_cont

    if (rest_frame == True)&(z is not None):
        lam = lam/(1+z)
        emline_spec = emline_spec*(1+z)
        ivar = ivar/((1+z)**2)

    return (lam, emline_spec, ivar)

###################################################################################################

def calculate_red_chi2(data, model, ivar, n_free_params):
    """
    This function computed the reduced chi2 for a given fit to the data
    
    Parameters
    ----------
    data : numpy array
        Data array
        
    model : numpy array
        Model array
        
    ivar : numpy array
        Inverse variance array
        
    n_free_params : int
        Number of free parameters associated with the fit
        
    Returns
    -------
    red_chi2 : float
        Reduced chi2 value for the given fit to the fata
    
    """
    
    ## chi2
    chi2 = sum(((data - model)**2)*ivar)
    ## Reduced chi2
    red_chi2 = chi2/(len(data)-n_free_params)
    
    return (red_chi2)
    
####################################################################################################

def lamspace_to_velspace(del_lam, lam_ref):
    """
    This function converts delta_wavelength from wavelength space to velocity space.
    
    Parameters 
    ----------
    del_lam : float
        FWHM or sigma in wavelength units
    lam_ref : float
        Reference wavelength for the conversion
        
    Return
    ------
    vel : float
        FWHM of simga in velocity units
    """
    ## Speed of light in km/s
    c = 2.99792e+5
    
    vel = (del_lam/lam_ref)*c
    
    return (vel)
    
####################################################################################################