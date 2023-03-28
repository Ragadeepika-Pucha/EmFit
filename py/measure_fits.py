"""
The functions in this script are related to different measurements related to the fits.
The script consists of following functions:
    1) calculate_red_chi2(data, model, ivar, n_free_params)
    2) lamspace_to_velspace(del_lam, lam_ref)
    3) velspace_to_lamspace(vel, lam_ref)
    4) compute_noise_emline(lam_rest, flam_rest, model, em_line)
    
Author : Ragadeepika Pucha
Version : 2023, March 27
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

def compute_noise_emline(lam_rest, flam_rest, model, em_line):
    """
    Function to compute noise near a given emission-line.
    
    Parameters
    ----------
    lam_rest : numpy array
        Rest-frame wavelength array of the spectrum
        
    flam_rest : numpy array
        Rest-frame flux array of the spectrum
        
    model : Astropy model
        Astropy model that is fit to the emission-line
        
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
    model_region = model(lam_ii)
    
    res = flam_region - model_region
    noise = np.std(res)
    
    return (noise)

####################################################################################################