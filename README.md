This code is designed for calculating the matter power spectrum incorporating supermassive primordial  black holes, dark matter halo mass distribution, and strong gravitational lensing time delays. We can refer to arXiv:......


SGLGW_PBH/
└── modules                          # The collection of all basic computing modules
    ├── constants.py                 # physical and cosmological constants
    ├── Cosmo.py                     # Cosmology simulator
    ├── Function.py                  # The power spectrum of matter and the mass distribution function of the dark matter halo
    ├── Lensis.py                    # Velocity dispersion, time delay and depth of light calculation under the SIS model
    ├── NIM.py                       # Time delay distribution numerical calculator
    ├── interpolators.py             # Interpolation module
                
└── PsHmfPlots.py                    # Power spectrum and halo mass distribution graphs
└── fpbhData.py                      # Generate datasets of fpbh parameters under different models
└── TransData.py                     # Transform Equation (28) into the time delay distribution data in Equation (27)
└── FiducialPlots.py                 # Generate the simulated time delay data and graphs under the Fiducial model (\Lambda CDM model)
└── McmcData.py                      # Generate the mcmc chain data for the posterior distribution of fpbh
└── McmcPlots.py                     # Mcmc chain diagram of the posterior distribution of fpbh
└── fpbhBound.py                     # Summary graph of the upper limit of fpbh at mass range

    
└── data                             # All input and output data sets
    ├── BBH                          # Redshift distribution of GW sources
    ├── fpbh_bound                   # Other constraints on fpbh
    ├── lensing_analysis_data        # time delay distribution data
    ├── mcmc_data                    # mcmc data for different models
    ├── simulation_data              # simulated time delay data under the Fiducial model


└── Plots                            # The collection of all output graphs
 

Continuously updated

Version 1.0.0
