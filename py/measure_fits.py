"""
The functions in this script are related to different measurements related to the fits.
The script consists of following functions:
    1) calculate_red_chi2(data, model, ivar, n_free_params)
    2) lamspace_to_velspace(del_lam, lam_ref)
    3) velspace_to_lamspace(vel, lam_ref)
    4) compute_noise_emline(lam_rest, flam_rest, model, em_line)
    5) compute_emline_flux(amplitude, stddev, amplitude_err, stddev_err)
    
Author : Ragadeepika Pucha
Version : 2023, May 8
"""

###################################################################################################

import numpy as np

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

def lamspace_to_velspace(del_lam, lam_ref, del_lam_err = None, lam_ref_err = None):
    """
    This function converts delta_wavelength from wavelength space to velocity space.
    Error is computed if the delta_wavelength and reference wavelength errors are given.
    
    Parameters 
    ----------
    del_lam : float
        FWHM or sigma in wavelength units
        
    lam_ref : float
        Reference wavelength for the conversion
        
    del_lam_err : float
        Error in FWHM of sigma in wavelength units
        Default is None
        
    lam_ref_err : float
        Error in reference wavelength
        Default is None
        
        
    Returns
    -------
    vel : float
        FWHM or simga in velocity units
    """
    ## Speed of light in km/s
    c = 2.99792e+5
    vel = (del_lam/lam_ref)*c
    
    if ((del_lam_err is not None)&(lam_ref_err is not None)):
        vel_err = vel*np.sqrt(((del_lam_err/del_lam)**2)+((lam_ref_err/lam_ref)**2))
        return (vel, vel_err)
    else:
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

def compute_noise_emline(lam_rest, flam_rest, em_line):
    """
    Function to compute noise near a given emission-line.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame wavelength array of the spectrum
        
    flam_rest : numpy array
        Rest-frame flux array of the spectrum
        
    em_line : str
        Emission-line region where the noise needs to be computed
        
    Returns
    -------
    noise : float
        Noise in the spectra near the specific emission-line.
    
    """
    
    if (em_line == 'hb'):
        lam_ii = ((lam_rest >= 4700) & (lam_rest <= 4800))|((lam_rest >= 4920)&(lam_rest <= 4935))
    elif (em_line == 'oiii'):
        lam_ii = ((lam_rest >= 4900)&(lam_rest <= 4935))|((lam_rest >= 5050)&(lam_rest <= 5100))
    elif (em_line == 'nii_ha'):
        lam_ii = ((lam_rest >= 6330)&(lam_rest <= 6450))|((lam_rest >= 6650)&(lam_rest <= 6690))
    elif (em_line == 'sii'):
        lam_ii = ((lam_rest >= 6650)&(lam_rest <= 6690))|((lam_rest >= 6760)&(lam_rest <= 6800))
    else:
        raise NameError('Emission-line not available!')
    
    lam_region = lam_rest[lam_ii]
    flam_region = flam_rest[lam_ii]
#     model_region = model(lam_region)

#     res = flam_region - model_region
#     noise = np.std(res)

    noise = np.sqrt(sum(flam_region**2)/len(flam_region))
    
    return (noise)

####################################################################################################

def compute_emline_flux(amplitude, stddev, amplitude_err = None, stddev_err = None):
    """
    Function to compute emission-line flux, given it is modeled as a Gaussian.
    
    Parameters
    ----------
    amplitude : float
        Amplitude of the Gaussian
        
    stddev : float
        Standard Deviation of the Gaussian
        
    amplitude_err : float
        Amplitude error of the Gaussian. Default is None.
        
    stddev_err : float
        Standard Deviation error of the Gaussian. Default is None.
        
    Returns
    -------
    flux : float
        Flux of the emission-line which is modeled as a Gaussian
    """
    
    
    flux = np.sqrt(2*np.pi)*amplitude*stddev
    
    if ((amplitude_err is not None)&(stddev_err is not None)):
        flux_err = flux*np.sqrt(((amplitude_err/amplitude)**2) + ((stddev_err/stddev)**2))
        return (flux, flux_err)
    else:
        return (flux)

####################################################################################################