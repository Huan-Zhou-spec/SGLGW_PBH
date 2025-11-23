#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 21:40:54 2025

@author: ubuntu
"""

"""
利用强透镜限制原初黑洞
"""


from .Functions import halo_mass_function, IsoPS, PkPBH
from .NIM import DeltaFunctionIntegral
from .Lensis import LensingAnalysis, sis_velocity_dispersion
from .Cosmo import Cosmology
from .interpolators import read_hdf5_data, create_interpolators, compute_interpolated_data

__version__ = '1.0.0'