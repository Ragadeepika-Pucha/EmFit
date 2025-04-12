"""
The functions in this script are related to different measurements related to the fits.
The script consists of following functions:
    1) calculate_chi2(data, model, ivar, n_dof, reduced_chi2)
    2) lamspace_to_velspace(del_lam, lam_ref)
    3) velspace_to_lamspace(vel, lam_ref)
    4) sigma_to_fwhm(sigma, sigma_err = None)
    5) compute_noise_emline(lam_rest, flam_rest, model, em_line)
    6) compute_emline_flux(amplitude, stddev, amplitude_err, stddev_err)
    7) measure_sii_difference(lam_sii, flam_sii)
    8) correct_for_rsigma(mean, std, rsig)
    
Author : Ragadeepika Pucha
Version : 2025, April 11
"""

###################################################################################################

import numpy as np

###################################################################################################

def calculate_chi2(data, model, ivar, n_dof = None, reduced_chi2 = False):
    """
    This function computes the chi2 (or reduced chi2) for a given fit to the data
    It returns reduced chi2 if reduced_chi2 is True and n_dof is not None.
    
    Parameters
    ----------
    data : numpy array
        Data array
        
    model : numpy array
        Model array
        
    ivar : numpy array
        Inverse variance array
        
    n_dof : int
        Number of degrees of freedom associated with the fit
        
    reduced_chi2 : bool
        Whether or not to compute reduced chi2
        
    Returns
    -------
    chi2 : float
        chi2 value for the given fit to the data
    
    """
    
    ## chi2
    chi2 = sum(((data - model)**2)*ivar)
    
    if ((reduced_chi2 == True)&(n_dof is not None)):
        ## Reduced chi2
        red_chi2 = chi2/(len(data)-n_dof)
        return (red_chi2)
    
    else:
        return (chi2)

###################################################################################################

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

def sigma_to_fwhm(sigma, sigma_err = None):
    """
    Calculate FWHM of an emission line from sigma values.
    FWHM = 2*sqrt(2*log(2))*sigma
    
    Parameters
    ----------
    sigma : array
        Array of sigma values
        
    sigma_err : array
        Array of sigma error values
        
    Returns
    -------
    fwhm : array
        Array of FWHM values
        
    fwhm_err : array
        Array of FWHM error values
    """
    
    fwhm = 2*np.sqrt(2*np.log(2))*sigma
    
    if (sigma_err is not None):
        fwhm_err = 2*np.sqrt(2*np.log(2))*sigma_err
        return (fwhm, fwhm_err)
    else:
        return (fwhm)
    
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
    elif (em_line == 'nev'):
        lam_ii = ((lam_rest >= 3300)&(lam_rest <= 3330))|((lam_rest >= 3370)&(lam_rest <= 3410))
    else:
        raise NameError('Emission-line not available!')
    
    lam_region = lam_rest[lam_ii]
    flam_region = flam_rest[lam_ii]

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

def measure_sii_difference(lam_sii, flam_sii):
    """
    To measure the difference between the left and right sides of the [SII]6716,6731 doublet.
    
    Parameters
    ----------
    lam_sii : numpy array
        Wavelength array of the [SII] region
        
    flam_sii : numpy array
        Flux array of the [SII] region
        
    Returns
    -------
    diff : float
        Difference between median left flux and median right flux
        
    frac : float
        Fraction of median left flux and median right flux
    """

    lam_left = (lam_sii <= 6670) #|((lam_sii >= 6790)&(lam_sii <= 6700))
    lam_right = (lam_sii >= 6700)
    
    flam_left = flam_sii[lam_left]
    flam_right = flam_sii[lam_right]
    
    diff = np.median(flam_left) - np.median(flam_right)
    frac = np.median(flam_left)/np.median(flam_right)
    
    return (diff, frac)

####################################################################################################

def correct_for_rsigma(mean, std, rsig):
    """
    Function to correct sigma for instrumental resolution
    
    Parameters
    ----------
    mean : float
        Mean of the Gaussian component
        
    std : float
        Standard deviation of the Gaussian component
        
    rsig : float
        Median resolution element in the fit region
        
    Returns
    -------
    sig_corr : float
        Corrected sigma of the Gaussian component
        
    flag : int
        Flag for whether the component is resolved or not.
        Flag = 0 : Resolved
        Flag = 1 : Unresolved
    """
    
    
    if (std > rsig):
        std_corr = np.sqrt((std**2) - (rsig**2))
        sig_corr = lamspace_to_velspace(std_corr, mean)
        flag = 0
    else:
        sig_corr = lamspace_to_velspace(std, mean)
        flag = 1
        
    return (sig_corr, flag)
        
####################################################################################################
