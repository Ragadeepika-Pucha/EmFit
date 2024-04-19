# Important to Note

1) The user can select extreme broad line candidates by applying the following cut: \
```
NII_HA_SII_RCHI2 != 0  OR  HB_OIII_RCHI2 != 0
```

2) `*_SIGMA_FLAG` for each emission-line component denotes whether the component is resolved or not:
    * Flag = 1: Resolved component --> `SIGMA` value is corrected for instrumental resolution
    * Flag = 0: Unresolved component --> `SIGMA` value is computed from the Gaussian standard deviation and not corrected for instrumental resolution
    * Flag = -1: No component --> `SIGMA` value = 0
  
3) For double-peaked lines, it is recommended to add the flux values of the two components for the total flux measurement (for majority of the science cases). These can be picked by applying constraints on the amplitude ratio of the two components.
   
