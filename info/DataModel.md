# Data Model

| Column Name                |   Type      | Units |                            Description                                       | 
|:--------------------------:|:-----------:|:-----:|:---------------------------------------------------------------------- |
| TARGETID                   |   int64     |   -   | Unique identifier of the DESI target                                  |
| SPECPROD                   |   bytes9    |   -   | Spectral Production Pipeline fuji|guadalupe                            |
| SURVEY                     |   bytes4    |   -   | DESI Survey of the spectra main or special or sv1 or sv2 or sv3 or cmx |
| PROGRAM                    |   bytes6    |   -   | Observing Program dark or bright or backup or other |
| HEALPIX                    |   int32     |   -   | Healpix number of the target |
| Z                          |   float64   |   -   | Redshift of the target |
| PER_BROAD                  |   float64   |   -   | Percentage of broad Ha detection in iterations |
| HB_N_AMPLITUDE             |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Narrow Hb component |
| HB_N_AMPLITUDE_ERR         |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow Hb component |
| HB_N_MEAN                  |   float64   |   Angstrom  |    Mean of the Narrow Hb component |
| HB_N_MEAN_ERR              |   float64   |   Angstrom  |    Uncertainty in Mean of the Narrow Hb component |
| HB_N_STD                   |   float64   |   Angstrom  |    Standard Deviation of the Narrow Hb component |
| HB_N_STD_ERR               |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Narrow Hb component |
| HB_N_FLUX                  |   float64   | 1e-17 erg/(cm2 s) |     Flux from the Narrow Hb component |
| HB_N_FLUX_ERR              |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Narrow Hb component |
| HB_N_FLUX_LERR             |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow Hb component |
| HB_N_FLUX_UERR             |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow Hb component |
| HB_N_SIGMA                 |   float64   |  km/s |     Width (in km/s) of the Narrow Hb component  |
| HB_N_SIGMA_ERR             |   float64   |  km/s |      Uncertainly in the Width of the Narrow Hb component |
| HB_N_SIGMA_LERR            |   float64   |  km/s |      16th percentile width of the Narrow Hb component |
| HB_N_SIGMA_UERR            |   float64   |  km/s |      84th percentile width of the Narrow Hb component |
| HB_N_SIGMA_FLAG            |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow Hb |
| HB_OUT_AMPLITUDE           |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow Hb component |
| HB_OUT_AMPLITUDE_ERR       |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow Hb component |
| HB_OUT_MEAN                |   float64   |  Angstrom  |     Mean of the Outflow Hb component |
| HB_OUT_MEAN_ERR            |   float64   |  Angstrom  |     Uncertainty in Mean of the Outflow Hb component |
| HB_OUT_STD                 |   float64   |  Angstrom  |     Standard Deviation of the Outflow Hb component |
| HB_OUT_STD_ERR             |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Outflow Hb component |
| HB_OUT_FLUX                |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Outflow Hb component |
| HB_OUT_FLUX_ERR            |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Outflow Hb component |
| HB_OUT_FLUX_LERR           |   float64   | 1e-17 erg/(cm2 s) |     16th percentile Flux from the Outflow Hb component |
| HB_OUT_FLUX_UERR           |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Outflow Hb component |
| HB_OUT_SIGMA               |   float64   |  km/s |      Width (in km/s) of the Outflow Hb component |
| HB_OUT_SIGMA_ERR           |   float64   |  km/s |      Uncertainly in the Width of the Outflow Hb component |
| HB_OUT_SIGMA_LERR          |   float64   |  km/s |      16th percentile width of the Outflow Hb component |
| HB_OUT_SIGMA_UERR          |   float64   |  km/s |      84th percentile width of the Outflow Hb component |
| HB_OUT_SIGMA_FLAG          |    int64    |   -   |     Flag regarding instrumental resolution correction for Outflow Hb |
| HB_B_AMPLITUDE             |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Broad Hb component |
| HB_B_AMPLITUDE_ERR         |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Broad Hb component | 
| HB_B_MEAN                  |   float64   |  Angstrom  |     Mean of the Broad Hb component      |             
| HB_B_MEAN_ERR              |   float64   |  Angstrom  |     Uncertainty in Mean of the Broad Hb component |
| HB_B_STD                   |   float64   |  Angstrom  |     Standard Deviation of the Broad Hb component |
| HB_B_STD_ERR               |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Broad Hb component |
| HB_B_FLUX                  |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Broad Hb component |
| HB_B_FLUX_ERR              |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Broad Hb component |
| HB_B_FLUX_LERR             |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Broad Hb component |
| HB_B_FLUX_UERR             |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Broad Hb component |
| HB_B_SIGMA                 |   float64   |  km/s |      Width (in km/s) of the Broad Hb component |
| HB_B_SIGMA_ERR             |   float64   |  km/s |      Uncertainly in the Width of the Broad Hb component |
| HB_B_SIGMA_LERR            |   float64   |  km/s |      16th percentile width of the Broad Hb component |
| HB_B_SIGMA_UERR            |   float64   |  km/s |      84th percentile width of the Broad Hb component |
| HB_B_SIGMA_FLAG            |    int64    |   -   |     Flag regarding instrumental resolution correction for Broad Hb |
| HB_CONTINUUM               |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Continuum around the Hb components |
| HB_CONTINUUM_ERR           |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Continuum around the Hb components |
| HB_NOISE                   |   float64   | 1e-17 erg/(Angstrom cm2 s) |     rms of the Continuum around the Hb components |
| HB_NDOF                    |    int64    |   -   |     Number of Degrees of Freedom associated with the Hb Fit |
| HB_RCHI2                   |   float64   |   -   |     Reduced chi2 associated with the Hb+[OIII]4959,5007 Fit  |
| OIII4959_AMPLITUDE         |   float64   | 1e-17 erg/(Angstrom cm2 s) |       Amplitude of the Narrow [OIII]4959 component          |
| OIII4959_AMPLITUDE_ERR     |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow [OIII]4959 component |
| OIII4959_MEAN              |   float64   |  Angstrom  |     Mean of the Narrow [OIII]4959 component |
| OIII4959_MEAN_ERR          |   float64   |  Angstrom  |     Uncertainty in Mean of the Narrow [OIII]4959 component |
| OIII4959_STD               |   float64   |  Angstrom  |     Standard Deviation of the Narrow [OIII]4959 component |
| OIII4959_STD_ERR           |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Narrow [OIII]4959 component | |
| OIII4959_FLUX              |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Narrow [OIII]4959 component |
| OIII4959_FLUX_ERR          |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Narrow [OIII]4959 component |
| OIII4959_FLUX_LERR         |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow [OIII]4959 component |
| OIII4959_FLUX_UERR         |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow [OIII]4959 component |
| OIII4959_SIGMA             |   float64   |  km/s |      Width (in km/s) of the Narrow [OIII]4959 component |
| OIII4959_SIGMA_ERR         |   float64   |  km/s |      Uncertainly in the Width of the Narrow [OIII]4959 component |
| OIII4959_SIGMA_LERR        |   float64   |   km/s |     16th percentile width of the Narrow [OIII]4959 component |
| OIII4959_SIGMA_UERR        |   float64   |  km/s |      84th percentile width of the Narrow [OIII]4959 component |
| OIII4959_SIGMA_FLAG        |    int64    |   -   |      Flag regarding instrumental resolution correction for Narrow [OIII]4959 |
| OIII4959_OUT_AMPLITUDE     |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [OIII]4959 component  |
| OIII4959_OUT_AMPLITUDE_ERR |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [OIII]4959 component |
| OIII4959_OUT_MEAN          |   float64   |   Angstrom  |    Mean of the Outflow [OIII]4959 component |
| OIII4959_OUT_MEAN_ERR      |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow [OIII]4959 component |
| OIII4959_OUT_STD           |   float64   |  Angstrom  |     Standard Deviation of the Outflow [OIII]4959 component |
| OIII4959_OUT_STD_ERR       |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Outflow [OIII]4959 component |
| OIII4959_OUT_FLUX          |   float64   |  1e-17 erg/(cm2 s) |     Flux from the Outflow [OIII]4959 component |
| OIII4959_OUT_FLUX_ERR      |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Outflow [OIII]4959 component    | 
| OIII4959_OUT_FLUX_LERR     |   float64   |  1e-17 erg/(cm2 s) |     16th percentile Flux from the Outflow [OIII]4959 component |
| OIII4959_OUT_FLUX_UERR     |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Outflow [OIII]4959 component |
| OIII4959_OUT_SIGMA         |   float64   |   km/s |     Width (in km/s) of the Outflow [OIII]4959 component      |  
| OIII4959_OUT_SIGMA_ERR     |   float64   |   km/s |     Uncertainly in the Width of the Outflow [OIII]4959 component |
| OIII4959_OUT_SIGMA_LERR    |   float64   |   km/s |     16th percentile width of the Outflow [OIII]4959 component |
| OIII4959_OUT_SIGMA_UERR    |   float64   |   km/s |     84th percentile width of the Outflow [OIII]4959 component |
| OIII4959_OUT_SIGMA_FLAG    |    int64    |  -   |     Flag regarding instrumental resolution correction for Outflow [OIII]4959 |
| OIII5007_AMPLITUDE         |   float64   |  1e-17 erg/(Angstrom cm2 s) |     Amplitude of the Narrow [OIII]5007 component |
| OIII5007_AMPLITUDE_ERR     |   float64   |  1e-17 erg/(Angstrom cm2 s) |     Uncertainty in Amplitude of the Narrow [OIII]5007 component |
| OIII5007_MEAN              |   float64   |  Angstrom  |     Mean of the Narrow [OIII]5007 component |
| OIII5007_MEAN_ERR          |   float64   |  Angstrom  |     Uncertainty in Mean of the Narrow [OIII]5007 component |
| OIII5007_STD               |   float64   |  Angstrom  |     Standard Deviation of the Narrow [OIII]5007 component |
| OIII5007_STD_ERR           |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Narrow [OIII]5007 component |
| OIII5007_FLUX              |   float64   |  1e-17 erg/(cm2 s) |     Flux from the Narrow [OIII]5007 component |
| OIII5007_FLUX_ERR          |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Narrow [OIII]5007 component |
| OIII5007_FLUX_LERR         |   float64   |  1e-17 erg/(cm2 s) |     16th percentile Flux from the Narrow [OIII]5007 component |
| OIII5007_FLUX_UERR         |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Narrow [OIII]5007 component |
| OIII5007_SIGMA             |   float64   |   km/s |    Width (in km/s) of the Narrow [OIII]5007 component |
| OIII5007_SIGMA_ERR         |   float64   |   km/s |    Uncertainly in the Width of the Narrow [OIII]5007 component |
| OIII5007_SIGMA_LERR        |   float64   |   km/s |     16th percentile width of the Narrow [OIII]5007 component |
| OIII5007_SIGMA_UERR        |   float64   |   km/s |     84th percentile width of the Narrow [OIII]5007 component |
| OIII5007_SIGMA_FLAG        |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow [OIII]5007 |
| OIII5007_OUT_AMPLITUDE     |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [OIII]5007 component |
| OIII5007_OUT_AMPLITUDE_ERR |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [OIII]4959 component |
| OIII5007_OUT_MEAN          |   float64   |   Angstrom  |    Mean of the Outflow [OIII]5007 component |
| OIII5007_OUT_MEAN_ERR      |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow [OIII]5007 component |
| OIII5007_OUT_STD           |   float64   |   Angstrom  |    Standard Deviation of the Outflow [OIII]5007 component |
| OIII5007_OUT_STD_ERR       |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Outflow [OIII]5007 component |
| OIII5007_OUT_FLUX          |   float64   |  1e-17 erg/(cm2 s) |     Flux from the Outflow [OIII]5007 component |
| OIII5007_OUT_FLUX_ERR      |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Outflow [OIII]5007 component |
| OIII5007_OUT_FLUX_LERR     |   float64   |  1e-17 erg/(cm2 s) |     16th percentile Flux from the Outflow [OIII]5007 component |
| OIII5007_OUT_FLUX_UERR     |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Outflow [OIII]5007 component |
| OIII5007_OUT_SIGMA         |   float64   |   km/s |     Width (in km/s) of the Outflow [OIII]5007 component |
| OIII5007_OUT_SIGMA_ERR     |   float64   |   km/s |     Uncertainly in the Width of the Outflow [OIII]5007 component |
| OIII5007_OUT_SIGMA_LERR    |   float64   |   km/s |     16th percentile width of the Outflow [OIII]5007 component |
| OIII5007_OUT_SIGMA_UERR    |   float64   |   km/s |     84th percentile width of the Outflow [OIII]5007 component |
| OIII5007_OUT_SIGMA_FLAG    |    int64    |   -   |     Flag regarding instrumental resolution correction for Outflow [OIII]5007 |
| OIII_CONTINUUM             |   float64   |  1e-17 erg/(Angstrom cm2 s) |     Continuum around the [OIII] components |
| OIII_CONTINUUM_ERR         |   float64   |  1e-17 erg/(Angstrom cm2 s) |    Uncertainty in Continuum around the [OIII] components |
| OIII_NOISE                 |   float64   |  1e-17 erg/(Angstrom cm2 s) |     rms of the Continuum around the [OIII] components |
| OIII_NDOF                  |    int64    |   -   |     Number of Degrees of Freedom associated with the [OIII]4959,5007 Fit |
| OIII_RCHI2                 |   float64   |   -   |     Reduced chi2 associated with the [OIII]4959,5007 Fit  |
| HB_OIII_NDOF               |    int64    |   -   |    Number of Degrees of Freedom associated with the Hb+[OIII]4959,5007 Fit |
| HB_OIII_RCHI2              |   float64   |   -   |     Reduced chi2 associated with the Hb+[OIII]4959,5007 Fit   |
| NII6548_AMPLITUDE          |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Narrow [NII]6548 component |
| NII6548_AMPLITUDE_ERR      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow [NII]6548 component |
| NII6548_MEAN               |   float64   |   Angstrom  |    Mean of the Narrow [NII]6548 component |
| NII6548_MEAN_ERR           |   float64   |   Angstrom  |    Uncertainty in Mean of the Narrow [NII]6548 component |
| NII6548_STD                |   float64   |   Angstrom  |    Standard Deviation of the Narrow [NII]6548 component |
| NII6548_STD_ERR            |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Narrow [NII]6548 component |
| NII6548_FLUX               |   float64   |  1e-17 erg/(cm2 s) |     Flux from the Narrow [NII]6548 component |
| NII6548_FLUX_ERR           |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Narrow [NII]6548 component |
| NII6548_FLUX_LERR          |   float64   |  1e-17 erg/(cm2 s) |     16th percentile Flux from the Narrow [NII]6548 component |
| NII6548_FLUX_UERR          |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Narrow [NII]6548 component |
| NII6548_SIGMA              |   float64   |   km/s |     Width (in km/s) of the Narrow [NII]6548 component |
| NII6548_SIGMA_ERR          |   float64   |   km/s |     Uncertainly in the Width of the Narrow [NII]6548 component |
| NII6548_SIGMA_LERR         |   float64   |   km/s |     16th percentile width of the Narrow [NII]6548 component |
| NII6548_SIGMA_UERR         |   float64   |   km/s |     84th percentile width of the Narrow [NII]6548 component |
| NII6548_SIGMA_FLAG         |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow [NII]6548 |
| NII6548_OUT_AMPLITUDE      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [NII]6548 component |
| NII6548_OUT_AMPLITUDE_ERR  |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [NII]6548 component |
| NII6548_OUT_MEAN           |   float64   |   Angstrom  |    Mean of the Outflow [NII]6548 component |
| NII6548_OUT_MEAN_ERR       |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow [NII]6548 component |
| NII6548_OUT_STD            |   float64   |   Angstrom  |    Standard Deviation of the Outflow [NII]6548 component |
| NII6548_OUT_STD_ERR        |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Outflow [NII]6548 component |
| NII6548_OUT_FLUX           |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Outflow [NII]6548 component |
| NII6548_OUT_FLUX_ERR       |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Outflow [NII]6548 component |
| NII6548_OUT_FLUX_LERR      |   float64   | 1e-17 erg/(cm2 s) |     16th percentile Flux from the Outflow [NII]6548 component |
| NII6548_OUT_FLUX_UERR      |   float64   | 1e-17 erg/(cm2 s) |     84th percentile Flux from the Outflow [NII]6548 component |
| NII6548_OUT_SIGMA          |   float64   |   km/s |    Width (in km/s) of the Outflow [NII]6548 component |
| NII6548_OUT_SIGMA_ERR      |   float64   |   km/s |     Uncertainly in the Width of the Outflow [NII]6548 component |      
| NII6548_OUT_SIGMA_LERR     |   float64   |   km/s |     16th percentile width of the Outflow [NII]6548 component |
| NII6548_OUT_SIGMA_UERR     |   float64   |   km/s |     84th percentile width of the Outflow [NII]6548 component |
| NII6548_OUT_SIGMA_FLAG     |    int64    |   -   |     Flag regarding instrumental resolution correction for Outflow [NII]6548 |
| NII6583_AMPLITUDE          |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Narrow [NII]6583 component |
| NII6583_AMPLITUDE_ERR      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow [NII]6583 component |
| NII6583_MEAN               |   float64   |   Angstrom  |    Mean of the Narrow [NII]6583 component |
| NII6583_MEAN_ERR           |   float64   |   Angstrom  |    Uncertainty in Mean of the Narrow [NII]6583 component |
| NII6583_STD                |   float64   |   Angstrom  |    Standard Deviation of the Narrow [NII]6583 component |
| NII6583_STD_ERR            |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Narrow [NII]6583 component |
| NII6583_FLUX               |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Narrow [NII]6583 component |
| NII6583_FLUX_ERR           |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Narrow [NII]6583 component |
| NII6583_FLUX_LERR          |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow [NII]6583 component |
| NII6583_FLUX_UERR          |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow [NII]6583 component |
| NII6583_SIGMA              |   float64   |  km/s |     Width (in km/s) of the Narrow [NII]6583 component |
| NII6583_SIGMA_ERR          |   float64   |  km/s |     Uncertainly in the Width of the Narrow [NII]6583 component |
| NII6583_SIGMA_LERR         |   float64   |  km/s |     16th percentile width of the Narrow [NII]6583 component |
| NII6583_SIGMA_UERR         |   float64   |  km/s |     84th percentile width of the Narrow [NII]6583 component |
| NII6583_SIGMA_FLAG         |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow [NII]6583 |
| NII6583_OUT_AMPLITUDE      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [NII]6583 component |
| NII6583_OUT_AMPLITUDE_ERR  |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [NII]6583 component |
| NII6583_OUT_MEAN           |   float64   |  Angstrom  |     Mean of the Outflow [NII]6583 component |
| NII6583_OUT_MEAN_ERR       |   float64   |  Angstrom  |     Uncertainty in Mean of the Outflow [NII]6583 component |
| NII6583_OUT_STD            |   float64   |  Angstrom  |     Standard Deviation of the Outflow [NII]6583 component |
| NII6583_OUT_STD_ERR        |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Outflow [NII]6583 component |
| NII6583_OUT_FLUX           |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Outflow [NII]6583 component |
| NII6583_OUT_FLUX_ERR       |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Outflow [NII]6583 component |
| NII6583_OUT_FLUX_LERR      |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Outflow [NII]6583 component |
| NII6583_OUT_FLUX_UERR      |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Outflow [NII]6583 component |
| NII6583_OUT_SIGMA          |   float64   |  km/s |      Width (in km/s) of the Outflow [NII]6583 component |
| NII6583_OUT_SIGMA_ERR      |   float64   |  km/s |     Uncertainly in the Width of the Outflow [NII]6583 component |
| NII6583_OUT_SIGMA_LERR     |   float64   |  km/s |      16th percentile width of the Outflow [NII]6583 component |
| NII6583_OUT_SIGMA_UERR     |   float64   |  km/s |      84th percentile width of the Outflow [NII]6583 component |
| NII6583_OUT_SIGMA_FLAG     |    int64    |   -   |     Flag regarding instrumental resolution correction for Outflow [NII]6583 |
| HA_N_AMPLITUDE             |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Narrow Ha component |
| HA_N_AMPLITUDE_ERR         |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow Ha component |
| HA_N_MEAN                  |   float64   |  Angstrom  |     Mean of the Narrow Ha component |
| HA_N_MEAN_ERR              |   float64   |  Angstrom  |    Uncertainty in Mean of the Narrow Ha component |
| HA_N_STD                   |   float64   |  Angstrom  |     Standard Deviation of the Narrow Ha component |
| HA_N_STD_ERR               |   float64   |  Angstrom  |     Uncertainty in Standard Deviation of the Narrow Ha component |
| HA_N_FLUX                  |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Narrow Ha component |
| HA_N_FLUX_ERR              |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Narrow Ha component |
| HA_N_FLUX_LERR             |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow Ha component |
| HA_N_FLUX_UERR             |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow Ha component |
| HA_N_SIGMA                 |   float64   |  km/s |      Width (in km/s) of the Narrow Ha component |
| HA_N_SIGMA_ERR             |   float64   |  km/s |      Uncertainly in the Width of the Narrow Ha component |
| HA_N_SIGMA_LERR            |   float64   |  km/s |      16th percentile width of the Narrow Ha component |
| HA_N_SIGMA_UERR            |   float64   |  km/s |      84th percentile width of the Narrow Ha component |
| HA_N_SIGMA_FLAG            |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow Ha |
| HA_OUT_AMPLITUDE           |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow Ha component |
| HA_OUT_AMPLITUDE_ERR       |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow Ha component | 
| HA_OUT_MEAN                |   float64   |   Angstrom  |    Mean of the Outflow Ha component |
| HA_OUT_MEAN_ERR            |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow Ha component |
| HA_OUT_STD                 |   float64   |   Angstrom  |    Standard Deviation of the Outflow Ha component |
| HA_OUT_STD_ERR             |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Outflow Ha component |
| HA_OUT_FLUX                |   float64   |  1e-17 erg/(cm2 s) |     Flux from the Outflow Ha component |
| HA_OUT_FLUX_ERR            |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Outflow Ha component |
| HA_OUT_FLUX_LERR           |   float64   |  1e-17 erg/(cm2 s) |     16th percentile Flux from the Outflow Ha component |
| HA_OUT_FLUX_UERR           |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Outflow Ha component |
| HA_OUT_SIGMA               |   float64   |  km/s |      Width (in km/s) of the Outflow Ha component |
| HA_OUT_SIGMA_ERR           |   float64   |  km/s |      Uncertainly in the Width of the Outflow Ha component |
| HA_OUT_SIGMA_LERR          |   float64   |  km/s |      16th percentile width of the Outflow Ha component |
| HA_OUT_SIGMA_UERR          |   float64   |  km/s |      84th percentile width of the Outflow Ha component |
| HA_OUT_SIGMA_FLAG          |    int64    |   -   |     Flag regarding instrumental resolution correction for Outflow Ha |
| HA_B_AMPLITUDE             |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Broad Ha component |
| HA_B_AMPLITUDE_ERR         |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Broad Ha component |
| HA_B_MEAN                  |   float64   |   Angstrom  |    Mean of the Broad Ha component |
| HA_B_MEAN_ERR              |   float64   |   Angstrom  |    Uncertainty in Mean of the Broad Ha component |
| HA_B_STD                   |   float64   |   Angstrom  |    Standard Deviation of the Broad Ha component |
| HA_B_STD_ERR               |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Broad Ha component |
| HA_B_FLUX                  |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Broad Ha component |
| HA_B_FLUX_ERR              |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Broad Ha component |
| HA_B_FLUX_LERR             |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Broad Ha component |
| HA_B_FLUX_UERR             |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Broad Ha component |
| HA_B_SIGMA                 |   float64   |  km/s |      Width (in km/s) of the Broad Ha component |
| HA_B_SIGMA_ERR             |   float64   |  km/s |      Uncertainly in the Width of the Broad Ha component |
| HA_B_SIGMA_LERR            |   float64   |  km/s |      16th percentile width of the Broad Ha component |
| HA_B_SIGMA_UERR            |   float64   |  km/s |      84th percentile width of the Broad Ha component |
| HA_B_SIGMA_FLAG            |    int64    |   -   |     Flag regarding instrumental resolution correction for Broad Ha |
| NII_HA_CONTINUUM           |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Continuum around the [NII]+Ha components |
| NII_HA_CONTINUUM_ERR       |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Continuum around the [NII]+Ha components |
| NII_HA_NOISE               |   float64   | 1e-17 erg/(Angstrom cm2 s) |      rms of the Continuum around the [NII]+Ha components |
| NII_HA_NDOF                |    int64    |   -   |      Number of Degrees of Freedom associated with the [NII]6548,6583+Ha Fit |
| NII_HA_RCHI2               |   float64   |   -   |     Reduced chi2 associated with the [NII]6548,6583+Ha Fit  |
| SII6716_AMPLITUDE          |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Narrow [SII]6716 component |
| SII6716_AMPLITUDE_ERR      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow [SII]6716 component |
| SII6716_MEAN               |   float64   |   Angstrom  |    Mean of the Narrow [SII]6716 component |
| SII6716_MEAN_ERR           |   float64   |   Angstrom  |    Uncertainty in Mean of the Narrow [SII]6716 component |
| SII6716_STD                |   float64   |   Angstrom  |    Standard Deviation of the Narrow [SII]6716 component |
| SII6716_STD_ERR            |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Narrow [SII]6716 component |
| SII6716_FLUX               |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Narrow [SII]6716 component |
| SII6716_FLUX_ERR           |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Narrow [SII]6716 component |
| SII6716_FLUX_LERR          |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow [SII]6716 component |
| SII6716_FLUX_UERR          |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow [SII]6716 component |
| SII6716_SIGMA              |   float64   |  km/s  |     Width (in km/s) of the Narrow [SII]6716 component |
| SII6716_SIGMA_ERR          |   float64   |  km/s  |     Uncertainly in the Width of the Narrow [SII]6716 component |
| SII6716_SIGMA_LERR         |   float64   |  km/s  |     16th percentile width of the Narrow [SII]6716 component |
| SII6716_SIGMA_UERR         |   float64   |  km/s  |     84th percentile width of the Narrow [SII]6716 component |
| SII6716_SIGMA_FLAG         |    int64    |   -   |     Flag regarding instrumental resolution correction for Narrow [SII]6716 |
| SII6716_OUT_AMPLITUDE      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [SII]6716 component |
| SII6716_OUT_AMPLITUDE_ERR  |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [SII]6716 component |
| SII6716_OUT_MEAN           |   float64   |   Angstrom  |    Mean of the Outflow [SII]6716 component |
| SII6716_OUT_MEAN_ERR       |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow [SII]6716 component |
| SII6716_OUT_STD            |   float64   |   Angstrom  |    Standard Deviation of the Outflow [SII]6716 component |
| SII6716_OUT_STD_ERR        |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Outflow [SII]6716 component |
| SII6716_OUT_FLUX           |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Outflow [SII]6716 component |
| SII6716_OUT_FLUX_ERR       |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Outflow [SII]6716 component |
| SII6716_OUT_FLUX_LERR      |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Outflow [SII]6716 component |
| SII6716_OUT_FLUX_UERR      |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Outflow [SII]6716 component |
| SII6716_OUT_SIGMA          |   float64   |  km/s  |     Width (in km/s) of the Outflow [SII]6716 component |
| SII6716_OUT_SIGMA_ERR      |   float64   |  km/s  |     Uncertainly in the Width of the Outflow [SII]6716 component |
| SII6716_OUT_SIGMA_LERR     |   float64   |  km/s  |     16th percentile width of the Outflow [SII]6716 component |
| SII6716_OUT_SIGMA_UERR     |   float64   |  km/s  |     84th percentile width of the Outflow [SII]6716 component |
| SII6716_OUT_SIGMA_FLAG     |    int64    |   -    |      Flag regarding instrumental resolution correction for Outflow [SII]6716 |
| SII6731_AMPLITUDE          |   float64   | 1e-17 erg/(Angstrom cm2 s) |     Amplitude of the Narrow [SII]6731 component |
| SII6731_AMPLITUDE_ERR      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Narrow [SII]6731 component |
| SII6731_MEAN               |   float64   |   Angstrom  |    Mean of the Narrow [SII]6731 component |
| SII6731_MEAN_ERR           |   float64   |   Angstrom  |    Uncertainty in Mean of the Narrow [SII]6731 component |
| SII6731_STD                |   float64   |   Angstrom  |    Standard Deviation of the Narrow [SII]6731 component |
| SII6731_STD_ERR            |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Narrow [SII]6731 component |
| SII6731_FLUX               |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Narrow [SII]6731 component |
| SII6731_FLUX_ERR           |   float64   |  1e-17 erg/(cm2 s) |     Uncertainty in Flux from the Narrow [SII]6731 component |
| SII6731_FLUX_LERR          |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Narrow [SII]6731 component |
| SII6731_FLUX_UERR          |   float64   | 1e-17 erg/(cm2 s) |      84th percentile Flux from the Narrow [SII]6731 component |
| SII6731_SIGMA              |   float64   |   km/s |     Width (in km/s) of the Narrow [SII]6731 component |
| SII6731_SIGMA_ERR          |   float64   |   km/s |     Uncertainly in the Width of the Narrow [SII]6731 component |
| SII6731_SIGMA_LERR         |   float64   |   km/s |     16th percentile width of the Narrow [SII]6731 component |
| SII6731_SIGMA_UERR         |   float64   |   km/s |     84th percentile width of the Narrow [SII]6731 component |
| SII6731_SIGMA_FLAG         |    int64    |   -    |   Flag regarding instrumental resolution correction for Narrow [SII]6731 |
| SII6731_OUT_AMPLITUDE      |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Amplitude of the Outflow [SII]6731 component |
| SII6731_OUT_AMPLITUDE_ERR  |   float64   | 1e-17 erg/(Angstrom cm2 s) |      Uncertainty in Amplitude of the Outflow [SII]6731 component |
| SII6731_OUT_MEAN           |   float64   |   Angstrom  |    Mean of the Outflow [SII]6731 component |
| SII6731_OUT_MEAN_ERR       |   float64   |   Angstrom  |    Uncertainty in Mean of the Outflow [SII]6731 component |
| SII6731_OUT_STD            |   float64   |   Angstrom  |    Standard Deviation of the Outflow [SII]6731 component |
| SII6731_OUT_STD_ERR        |   float64   |   Angstrom  |    Uncertainty in Standard Deviation of the Outflow [SII]6731 component |
| SII6731_OUT_FLUX           |   float64   | 1e-17 erg/(cm2 s) |      Flux from the Outflow [SII]6731 component |
| SII6731_OUT_FLUX_ERR       |   float64   | 1e-17 erg/(cm2 s) |      Uncertainty in Flux from the Outflow [SII]6731 component |
| SII6731_OUT_FLUX_LERR      |   float64   | 1e-17 erg/(cm2 s) |      16th percentile Flux from the Outflow [SII]6731 component |
| SII6731_OUT_FLUX_UERR      |   float64   |  1e-17 erg/(cm2 s) |     84th percentile Flux from the Outflow [SII]6731 component |
| SII6731_OUT_SIGMA          |   float64   |   km/s |      Width (in km/s) of the Outflow [SII]6731 component |
| SII6731_OUT_SIGMA_ERR      |   float64   |   km/s |     Uncertainly in the Width of the Outflow [SII]6731 component |
| SII6731_OUT_SIGMA_LERR     |   float64   |   km/s |     16th percentile width of the Outflow [SII]6731 component |
| SII6731_OUT_SIGMA_UERR     |   float64   |   km/s |     84th percentile width of the Outflow [SII]6731 component |
| SII6731_OUT_SIGMA_FLAG     |    int64    |   -    |     Flag regarding instrumental resolution correction for Outflow [SII]6731 |
| SII_CONTINUUM              |   float64   | 1e-17 erg/(Angstrom cm2 s) |     Continuum around the [SII] components |
| SII_CONTINUUM_ERR          |   float64   | 1e-17 erg/(Angstrom cm2 s) |     Uncertainty in Continuum around the [SII] components |
| SII_NOISE                  |   float64   | 1e-17 erg/(Angstrom cm2 s) |      rms of the Continuum around the [SII] components |
| SII_NDOF                   |    int64    |   -   |     Number of Degrees of Freedom associated with the [SII]6716,6731 Fit |
| SII_RCHI2                  |   float64   |   -   |      Reduced chi2 associated with the [SII]6717,6731 Fit  |
| NII_HA_SII_NDOF            |    int64    |   -   |     Number of Degrees of Freedom associated with the [NII]6548,6583 + Ha + [SII]6717,6731 Fit |
| NII_HA_SII_RCHI2           |   float64   |   -   |     Reduced chi2 associated with the [NII]6548,6583+Ha+[SII]6717,6731 Fit  |
