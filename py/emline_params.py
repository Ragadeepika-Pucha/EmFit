"""
This script consists of functions for computing the parameters of the emission-line fits.
It consists of the following functions:

Author : Ragadeepika Pucha
Version : 2023, March 28
"""

###################################################################################################

import numpy as np

import measure_fits as mfit
###################################################################################################

def get_sii_parameters(fitter_sii, gfit_sii):
    
    ## Number of sub-models in the [SII] fit
    n_sii = gfit_sii.n_submodels
    ## If n_sii = 2 -- No outflow components
    ## If n_sii = 4 -- Outflow components for both [SII]6716, 6731
    
    if (n_sii == 2):
        ## Extract information for the fits
        ## Amplitude, Mean, Standard deviation of both [SII]6716,6731
        amp_sii6716, mean_sii6716, std_sii6716,\
        amp_sii6731, mean_sii6731, std_sii6731 = gfit_sii.parameters
        
        ## Mean of [SII]6731 is tied to [SII]6716
        ## Standard deviation of [SII]6731 is tied to [SII]6716
        ## Errors for the fit are therefore only for 
        ## Amp, Mean, Std of [SII]6716 and Amp of [SII]6731
        amperr_sii6716, meanerr_sii6716, stderr_sii6716, \
        amperr_sii6731 = np.sqrt(np.diag(fitter_sii.fit_info['param_cov']))
        
        ## Mean error of [SII]6731 = Mean error of [SII]6716
        meanerr_sii6731 = meanerr_sii6716
        ## Standard deviation error of [SII]6731 depends on other parameters
        ## std_6731 = std_6716*(mean_6731/mean_6716)
        ## Error formula for multiplication and division
        stderr_sii6731 = std_sii6731*np.sqrt(((stderr_sii6716/std_sii6716)**2) + \
                                            ((meanerr_sii6716/mean_sii6716)**2) + \
                                            ((meanerr_sii6731/mean_sii6731)**2))
        
        ## Sigma values in km/s
        sig_sii6716, sigerr_sii6716 = mfit.lamspace_to_velspace(std_sii6716, mean_sii6716, \
                                                                stderr_sii6716, meanerr_sii6716)
        sig_sii6731, sigerr_sii6731 = mfit.lamspace_to_velspace(std_sii6731, mean_sii6731, \
                                                                stderr_sii6731, meanerr_sii6731)
        
        
        ## Flux values 
        flux_sii6716, fluxerr_sii6716 = mfit.compute_emline_flux(amp_sii6716, std_sii6716,\
                                                                 amperr_sii6716, stderr_sii6716)
        flux_sii6731, fluxerr_sii6731 = mfit.compute_emline_flux(amp_sii6731, std_sii6731,\
                                                                 amperr_sii6731, stderr_sii6731)
    
        
        ## The outflow components are set to zero
        amp_sii6716_out, mean_sii6716_out, std_sii6716_out, \
        amp_sii6731_out, mean_sii6731_out, std_sii6731_out = np.zeros(6)
        amperr_sii6716_out, meanerr_sii6716_out, stderr_sii6716_out, \
        amperr_sii6731_out, meanerr_sii6731_out, stderr_sii6731_out = np.zeros(6)

        sig_sii6716_out, sigerr_sii6716_out, \
        sig_sii6731_out, sigerr_sii6731_out = np.zeros(4)
        flux_sii6716_out, fluxerr_sii6716_out, \
        flux_sii6731_out, fluxerr_sii6731_out = np.zeros(4)
        
    elif (n_sii == 4):
        ## Outflow components
        ## Extract information from the fits
        ## Amplitude, mean, standard deviation of all the four components
        amp_sii6716, mean_sii6716, std_sii6716,\
        amp_sii6731, mean_sii6731, std_sii6731,\
        amp_sii6716_out, mean_sii6716_out, std_sii6716_out,\
        amp_sii6731_out, mean_sii6731_out, std_sii6731_out = gfit_sii.parameters
        
        ## Mean of [SII]6731 is tied to [SII]6716
        ## Standard deviation of [SII]6731 is tied to [SII]6716
        ## Mean of [SII]6731_out is tied to [SII]6716_out
        ## Standard deviation of [SII]6731_out is tied to [SII]6716_out
        ## Amplitude of [SII]6731_out is tied to all the other three amplitude
        ## Errors for the fit are therefore in the following order
        ## amperr_sii6716, meanerr_sii6716, stderr_sii6716,
        ## amperr_sii6731, amperr_sii6716_out, meanerr_sii6716_out, stderr_sii6716_out

        amperr_sii6716, meanerr_sii6716, stderr_sii6716,\
        amperr_sii6731, amperr_sii6716_out, meanerr_sii6716_out,\
        stderr_sii6716_out = np.sqrt(np.diag(fitter_sii.fit_info['param_cov']))
        
        ## Mean error of [SII]6731 = Mean error of [SII]6716
        meanerr_sii6731 = meanerr_sii6716
        ## Standard deviation error of [SII]6731 depends on other parameters
        ## std_6731 = std_6716*(mean_6731/mean_6716)
        ## Error formula for multiplication and division
        stderr_sii6731 = std_sii6731*np.sqrt(((stderr_sii6716/std_sii6716)**2) + \
                                            ((meanerr_sii6716/mean_sii6716)**2) + \
                                            ((meanerr_sii6731/mean_sii6731)**2))
        
        ## Mean error of [SII]6731_out = Mean error of [SII]6716_out
        meanerr_sii6731_out = meanerr_sii6716_out
        ## std_6731_out = std_6716_out*(mean_6731_out/mean_6716_out)
        ## Error formula for multiplication and division
        stderr_sii6731_out = std_sii6731_out*np.sqrt(((stderr_sii6716_out/std_sii6716_out)**2) + \
                                                     ((meanerr_sii6716_out/mean_sii6716_out)**2) + \
                                                     ((meanerr_sii6731/mean_sii6731)**2))
        
        ## amp_6731_out = (amp_6731)*(amp_6716_out/amp_6716)
        ## Error formula for multiplication and division
        amperr_sii6731_out = amp_sii6731_out*np.sqrt(((amperr_sii6716/amp_sii6716)**2)+\
                                                     ((amperr_sii6731/amp_sii6731)**2)+\
                                                     ((amperr_sii6716_out/amp_sii6716_out)**2))

        
        ## Sigma values in km/s
        sig_sii6716, sigerr_sii6716 = mfit.lamspace_to_velspace(std_sii6716, mean_sii6716, \
                                                                stderr_sii6716, meanerr_sii6716)
        sig_sii6731, sigerr_sii6731 = mfit.lamspace_to_velspace(std_sii6731, mean_sii6731, \
                                                                stderr_sii6731, meanerr_sii6731)
    
        sig_sii6716_out, sigerr_sii6716_out = mfit.lamspace_to_velspace(std_sii6716_out,\
                                                                        mean_sii6716_out, \
                                                                        stderr_sii6716_out,\
                                                                        meanerr_sii6716_out)
        sig_sii6731_out, sigerr_sii6731_out = mfit.lamspace_to_velspace(std_sii6731_out,\
                                                                        mean_sii6731_out, \
                                                                        stderr_sii6731_out,\
                                                                        meanerr_sii6731_out)
    
        ## Flux values
        flux_sii6716, fluxerr_sii6716 = mfit.compute_emline_flux(amp_sii6716, std_sii6716, \
                                                                 amperr_sii6716, stderr_sii6716)
        flux_sii6731, fluxerr_sii6731 = mfit.compute_emline_flux(amp_sii6731, std_sii6731, \
                                                                 amperr_sii6731, stderr_sii6731)
    
        flux_sii6716_out, fluxerr_sii6716_out = mfit.compute_emline_flux(amp_sii6716_out, \
                                                                         std_sii6716_out, \
                                                                         amperr_sii6716_out, \
                                                                         stderr_sii6716_out)
        flux_sii6731_out, fluxerr_sii6731_out = mfit.compute_emline_flux(amp_sii6731_out, \
                                                                         std_sii6731_out, \
                                                                         amperr_sii6731_out, \
                                                                         stderr_sii6731_out)
        
    ## List of all the [SII] parameters
    ## [SII]6716, [SII]6716_out, [SII]6731, [SII]6731_out
    sii_params = [amp_sii6716, amperr_sii6716, mean_sii6716, meanerr_sii6716, \
                  std_sii6716, stderr_sii6716, sig_sii6716, sigerr_sii6716, \
                  flux_sii6716, fluxerr_sii6716, \
                  amp_sii6716_out, amperr_sii6716_out, mean_sii6716_out, meanerr_sii6716_out, \
                  std_sii6716_out, stderr_sii6716_out, sig_sii6716_out, sigerr_sii6716_out, \
                  flux_sii6716_out, fluxerr_sii6716_out, \
                  amp_sii6731, amperr_sii6731, mean_sii6731, meanerr_sii6731, \
                  std_sii6731, stderr_sii6731, sig_sii6731, sigerr_sii6731, \
                  flux_sii6731, fluxerr_sii6731, \
                  amp_sii6731_out, amperr_sii6731_out, mean_sii6731_out, meanerr_sii6731_out, \
                  std_sii6731_out, stderr_sii6731_out, sig_sii6731_out, sigerr_sii6731_out, \
                  flux_sii6731_out, fluxerr_sii6731_out]
    
    return (sii_params)

###################################################################################################

def get_oiii_params(fitter_oiii, gfit_oiii):
    
    ## Number of sub-models in the [OIII] fit
    n_oiii = gfit_oiii.n_submodels
    ## If n = 2, no outflow components
    ## If n = 4, outflow components in both [OIII]4959, 5007
    
    if (n_oiii == 2):
        ## Extract information from the fits
        amp_oiii4959, mean_oiii4959, std_oiii4959, \
        amp_oiii5007, mean_oiii5007, std_oiii5007 = gfit_oiii.parameters
        
        ## Amp_OIII5007 is tied to Amp_OIII4959
        ## Mean_OIII5007 is tied to Mean_OIII5007
        ## Std_OIII5007 is tied to Std_OIII4959
        ## Error from the fits
        amperr_oiii4959, \
        meanerr_oiii4959, \
        stderr_oiii4959 = np.sqrt(np.diag(fitter_oiii.fit_info['param_cov']))
        
        ## Amp_OIII5007 = 2.98*Amp_OIII4959
        amperr_oiii5007 = 2.98*amperr_oiii4959
        meanerr_oiii5007 = meanerr_oiii4959
        
        ## std_oiii5007 = std_oiii4959*(mean_oiii5007/mean_oiii4959)
        ## Error propagation formula for multiplication and division
        stderr_oiii5007 = std_oiii5007*np.sqrt(((stderr_oiii4959/std_oiii4959)**2)+\
                                               ((meanerr_oiii4959/mean_oiii4959)**2)+\
                                               ((meanerr_oiii5007/mean_oiii5007)**2))
        
        ## Sigma values in km/s
        sig_oiii4959, sigerr_oiii4959 = mfit.lamspace_to_velspace(std_oiii4959, mean_oiii4959, \
                                                                  stderr_oiii4959, meanerr_oiii4959)
        sig_oiii5007, sigerr_oiii5007 = mfit.lamspace_to_velspace(std_oiii5007, mean_oiii5007, \
                                                                  stderr_oiii5007, meanerr_oiii5007)
        
        ## Flux values
        flux_oiii4959, fluxerr_oiii4959 = mfit.compute_emline_flux(amp_oiii4959, std_oiii4959, \
                                                                   amperr_oiii4959, stderr_oiii4959)
        flux_oiii5007, fluxerr_oiii5007 = mfit.compute_emline_flux(amp_oiii5007, std_oiii5007, \
                                                                   amperr_oiii5007, stderr_oiii5007)
        
        
        ## Set all outflow values to zero
        amp_oiii4959_out, mean_oiii4959_out, std_oiii4959_out, \
        amp_oiii5007_out, mean_oiii5007_out, std_oiii5007_out = np.zeros(6)

        amperr_oiii4959_out, meanerr_oiii4959_out, stderr_oiii4959_out, \
        amperr_oiii5007_out, meanerr_oiii5007_out, stderr_oiii5007_out = np.zeros(6)

        sig_oiii4959_out, sigerr_oiii4959_out, \
        flux_oiii4959_out, fluxerr_oiii4959_out = np.zeros(4)
        
        sig_oiii5007_out, sigerr_oiii5007_out, \
        flux_oiii5007_out, fluxerr_oiii5007_out = np.zeros(4)
        
    elif (n_oiii == 4):
        
        ## Include outflow components
        amp_oiii4959, mean_oiii4959, std_oiii4959, \
        amp_oiii5007, mean_oiii5007, std_oiii5007, \
        amp_oiii4959_out, mean_oiii4959_out, std_oiii4959_out, \
        amp_oiii5007_out, mean_oiii5007_out, std_oiii5007_out = gfit_oiii.parameters
        
        ## Amp[OIII]5007(_out) is tied to Amp[OIII]4959(_out)
        ## Mean[OIII]5007(_out) is tied to Mean[OIII]4959(_out)
        ## Std[OIII]5007(_out) is tied to Std[OIII]4959(_out)
        amperr_oiii4959, meanerr_oiii4959, \
        stderr_oiii4959, amperr_oiii4959_out, \
        meanerr_oiii4959_out, \
        stderr_oiii4959_out = np.sqrt(np.diag(fitter_oiii.fit_info['param_cov']))
        
        ## Amp_oiii5007(_out) = 2.98*Amp_oiii4959(_out)
        amperr_oiii5007 = 2.98*amperr_oiii4959
        amperr_oiii5007_out = 2.98*amperr_oiii4959_out
        
        meanerr_oiii5007 = meanerr_oiii4959
        meanerr_oiii5007_out = meanerr_oiii4959_out
        
        ## Std_oiii5007(_out) = (std_oiii4959(_out))*(mean_oiii5007(_out)/mean_oiii4959(_out))
        ## Error propagration formula for multiplication and division
        stderr_oiii5007 = std_oiii5007*np.sqrt(((stderr_oiii4959/std_oiii4959)**2)+\
                                               ((meanerr_oiii4959/mean_oiii4959)**2)+\
                                               ((meanerr_oiii5007/mean_oiii5007)**2))
        
        stderr_oiii5007_out = std_oiii5007_out*np.sqrt(((stderr_oiii4959_out/std_oiii4959_out)**2) + \
                                                       ((meanerr_oiii4959_out/mean_oiii4959_out)**2) + \
                                                       ((meanerr_oiii5007_out/mean_oiii5007_out)**2))
        
        ## Sigma values in km/s
        sig_oiii4959, sigerr_oiii4959 = mfit.lamspace_to_velspace(std_oiii4959, mean_oiii4959, \
                                                                  stderr_oiii4959, meanerr_oiii4959)
        sig_oiii5007, sigerr_oiii5007 = mfit.lamspace_to_velspace(std_oiii5007, mean_oiii5007, \
                                                                  stderr_oiii5007, meanerr_oiii5007)
        
        sig_oiii4959_out, sigerr_oiii4959_out = mfit.lamspace_to_velspace(std_oiii4959_out, \
                                                                          mean_oiii4959_out, \
                                                                          stderr_oiii4959_out, \
                                                                          meanerr_oiii4959_out)
        sig_oiii5007_out, sigerr_oiii5007_out = mfit.lamspace_to_velspace(std_oiii5007_out, \
                                                                          mean_oiii5007_out, \
                                                                          stderr_oiii5007_out, \
                                                                          meanerr_oiii5007_out)
        
        ## Flux values
        flux_oiii4959, fluxerr_oiii4959 = mfit.compute_emline_flux(amp_oiii4959, std_oiii4959, \
                                                                   amperr_oiii4959, stderr_oiii4959)
        flux_oiii5007, fluxerr_oiii5007 = mfit.compute_emline_flux(amp_oiii5007, std_oiii5007, \
                                                                   amperr_oiii5007, stderr_oiii5007)
        
        flux_oiii4959_out, fluxerr_oiii4959_out = mfit.compute_emline_flux(amp_oiii4959_out, \
                                                                           std_oiii4959_out, \
                                                                           amperr_oiii4959_out, \
                                                                           stderr_oiii4959_out)
        flux_oiii5007_out, fluxerr_oiii5007_out = mfit.compute_emline_flux(amp_oiii5007_out, \
                                                                           std_oiii5007_out, \
                                                                           amperr_oiii5007_out, \
                                                                           stderr_oiii5007_out)
        
        
    oiii_params = [amp_oiii4959, amperr_oiii4959, mean_oiii4959, meanerr_oiii4959, \
                   std_oiii4959, stderr_oiii4959, sig_oiii4959, sigerr_oiii4959, \
                   flux_oiii4959, fluxerr_oiii4959, \
                   amp_oiii4959_out, amperr_oiii4959_out, mean_oiii4959_out, meanerr_oiii4959_out, \
                   std_oiii4959_out, stderr_oiii4959_out, sig_oiii4959_out, sigerr_oiii4959_out, \
                   flux_oiii4959_out, fluxerr_oiii4959_out, \
                   amp_oiii5007, amperr_oiii5007, mean_oiii5007, meanerr_oiii5007, \
                   std_oiii5007, stderr_oiii5007, sig_oiii5007, sigerr_oiii5007, \
                   flux_oiii5007, fluxerr_oiii5007, \
                   amp_oiii5007_out, amperr_oiii5007_out, mean_oiii5007_out, meanerr_oiii5007_out, \
                   std_oiii5007_out, stderr_oiii5007_out, sig_oiii5007_out, sigerr_oiii5007_out, \
                   flux_oiii5007_out, fluxerr_oiii5007_out]
    
    return (oiii_params)

###################################################################################################

def get_hb_params(fitter_hb, gfit_hb):
    
    ## Number of submodels
    n_hb = gfit_hb.n_submodels
    ## If n = 1, no broad-component
    ## If n = 2, broad component
    
    if (n_hb == 1):
        ## Extract information from the fits
        
        ## All the variables are independent 
        amp_hb_n, mean_hb_n, std_hb_n = gfit_hb.parameters
        
        ## Errors of the fits
        amperr_hb_n, meanerr_hb_n, stderr_hb_n = np.sqrt(np.diag(fitter_hb.fit_info['param_cov']))
        
        ## Sigma values in km/s
        sig_hb_n, sigerr_hb_n = mfit.lamspace_to_velspace(std_hb_n, mean_hb_n, \
                                                          stderr_hb_n, meanerr_hb_n)
        
        ## Flux values
        flux_hb_n, fluxerr_hb_n = mfit.compute_emline_flux(amp_hb_n, std_hb_n, \
                                                           amperr_hb_n, stderr_hb_n)
        
        ## Set broad flux values to zero
        amp_hb_b, mean_hb_b, std_hb_b = np.zeros(3)
        amperr_hb_b, meanerr_hb_b, stderr_hb_b = np.zeros(3)
        
        sig_hb_b, sigerr_hb_b, flux_hb_b, fluxerr_hb_b = np.zeros(4)
        
    elif (n_hb == 2):
        ## Broad-component exists
        
        ## All the variables are independent
        amp_hb_n, mean_hb_n, std_hb_n, \
        amp_hb_b, mean_hb_b, std_hb_b = gfit_hb.parameters
        
        ## Errors from the fit
        amperr_hb_n, meanerr_hb_n, stderr_hb_n, \
        amperr_hb_b, meanerr_hb_b, stderr_hb_b = np.sqrt(np.diag(fitter_hb.fit_info['param_cov']))
        
        ## Sigma values in km/s
        sig_hb_n, sigerr_hb_n = mfit.lamspace_to_velspace(std_hb_n, mean_hb_n, \
                                                          stderr_hb_n, meanerr_hb_n)
        
        sig_hb_b, sigerr_hb_b = mfit.lamspace_to_velspace(std_hb_b, mean_hb_b, \
                                                          stderr_hb_b, meanerr_hb_b)
        
        ## Flux values
        flux_hb_n, fluxerr_hb_n = mfit.compute_emline_flux(amp_hb_n, std_hb_n, \
                                                           amperr_hb_n, stderr_hb_n)
        
        flux_hb_b, fluxerr_hb_b = mfit.compute_emline_flux(amp_hb_b, std_hb_b, \
                                                           amperr_hb_b, stderr_hb_b)
        
        
    hb_params = [amp_hb_n, amperr_hb_n, \
                 mean_hb_n, meanerr_hb_n, \
                 std_hb_n, stderr_hb_n, \
                 sig_hb_n, sigerr_hb_n, \
                 flux_hb_n, fluxerr_hb_n, \
                 amp_hb_b, amperr_hb_b, \
                 mean_hb_b, meanerr_hb_b, \
                 std_hb_b, stderr_hb_b, \
                 sig_hb_b, sigerr_hb_b, \
                 flux_hb_b, fluxerr_hb_b]
    
    return (hb_params)
                 
###################################################################################################
                   
        


    
    
    
        
        
        