"""
This script consists of utility functions for emission-line fitting related stuff.

Author : Ragadeepika Pucha
Version : 2023, March 21
"""

###################################################################################################

import numpy as np

from astropy.table import Table
import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

import matplotlib.pyplot as plt

###################################################################################################

## Making the matplotlib plots look nicer
settings = {
    'font.size':22,
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

def plot_spectra_continuum(lam, flam, total_cont, axs = None):
    """
    This function overplots the stellar continuum on the spectra.
    
    Parameters
    ----------
    lam : numpy array
        Wavelength array of the spectra. 
        
    flam : numpy array
        Flux array of the spectra
        
    total_cont : numpy array
        Total stellar continuum of the spectra
        
    axs : axis object
        Axes where the plot needs to be. Default is None.
        
    Returns
    -------
        None
    """
    
    if (axs == None):
        plt.figure(figsize = (24, 8))
        axs = plt.gca()

    axs.plot(lam, flam, color = 'grey', alpha = 0.8, label = 'Spectra')
    axs.plot(lam, total_cont, color = 'r', lw = 2.0, label = 'Total continuum')
    axs.set(xlabel = '$\lambda$', ylabel = '$F_{\lambda}$')
    axs.legend(fontsize = 16, loc = 'best')
    
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

    if (plot_continuum == True):
        plot_spectra_continuum(lam, flam, total_cont)
        
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
        
    Returns
    -------
    vel : float
        FWHM or simga in velocity units
    """
    ## Speed of light in km/s
    c = 2.99792e+5
    
    vel = (del_lam/lam_ref)*c
    
    return (vel)
    
####################################################################################################

def velspace_to_lamspace(vel, lam_ref):
    """
    This function converts velocity from velocity space to wavelength space.
    
    Parameters
    ----------
    vel : flaot
        FWHM or sigma in velocity space
    lam_ref : float
        Reference wavelength for the conversion
        
    Returns
    -------
    del_lam : float
        FWHM or sigma in wanvelength units
    """
    ## Speed of light in km/s
    c = 2.99792e+5
    
    del_lam = (vel/c)*lam_ref
    
    return (del_lam)

####################################################################################################
    
def compute_aon_emline(lam_rest, flam_rest, ivar_rest, model, emline):
    
    if (emline == 'hb'):
        noise_lam = ((lam_rest >= 4700) & (lam_rest <= 4800))|((lam_rest >= 4920)&(lam_rest <= 4935))
    elif (emline == 'sii'):
        noise_lam = ((lam_rest >= 6650)&(lam_rest <= 6690))|((lam_rest >= 6760)&(lam_rest <= 6800))
    elif (emline == 'oiii'):
        noise_lam = ((lam_rest >= 4900)&(lam_rest <= 4935))|((lam_rest >= 5050)&(lam_rest <= 5100))
    elif (emline == 'nii_ha'):
        noise_lam = ((lam_rest >= 6330)&(lam_rest <= 6450))|((lam_rest >= 6650)&(lam_rest <= 6690))
        
    lam_region = lam_rest[noise_lam]
    flam_region = flam_rest[noise_lam]
    model_region = model(lam_region)
    
    res = flam_region - model_region
    noise = np.std(res)
    
    n_models = model.n_submodels
    
    if (n_models > 1):
        names_models = model.submodel_names
        aon_vals = dict()
        for name in names_models:
            aon = model[name].amplitude/noise
            aon_vals[name] = aon
    else:
        name = model.name
        
        aon_vals = dict()
        aon = model.amplitude/noise
        aon_vals[name] = aon
    
   
    return (aon_vals)
        
####################################################################################################
    