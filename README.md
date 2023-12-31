# Introduction

Compute eigenfunctions and eigenfrequencies of the linearized perturbation equations with a prescribed background state.

# Scripts:

1. `plot_komega_from_solar.py`: construct and plot a k-omega diagram using the background state from Solar Model S.

1. `plot_komega_from_sim.py`: construct and plot a k-omega diagram using the background state from a simulation of convection.

1. `plot_mode_from_sim.py`: find and plot a single eigenfunction using the background from a simulation of convection. Initial guess for the eigenfunction and eigenfrequency should be changed by modifying the script.

1. `problem.py`: find and plot a single eigenfunction using Solar Model S as the background state. Initial guess for the eigenfunction and eigenfrequency should be changed by modifying the script.


# References:

[BirKosDuv04]
Birch, Kosovichev, Duvall 2004 - Sensitivity of Acoustic Wave Travel Times to Sound-Speed Perturbations in the Solar Interior

[SchCamGiz11]
Schunker, Cameron, Gizon, Moradi 2011 - Constructing and Characterising Solar Structure Models for Computational Helioseismology

# Software versions

Tested with

* Python 3.11.6

* Scipy 1.11.4

* Numpy 1.26.2

* Matplotlib 3.8.1
