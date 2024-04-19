# Known Issues

**1) Assumption of a Single broad component in the Balmer lines** \
We assume a single Gaussian for the Balmer broad components. This works well for most of the sources, but in some cases of extreme broad-line galaxies, they might require two or more broad components for a good fit.

**2) Wrong Fits with Negative Continuum** \
In a few very noisy spectrum cases, the fit leads to a bad fit consisting of a very negative continuum + a very broad emission (>100,000 km/s) line. There are ~300 such sources in Fuji and Guadalupe. 
They can be removed by placing constraints on the `*_CONTINUUM` column values. 

**3) Incorrect Smooth Continuum From FastSpecFit** \
The smooth continuum is a few candidates take away the broad component, leading to a narrower width measurement. We visually found ~5 candidates so far.


