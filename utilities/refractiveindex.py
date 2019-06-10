#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
NAME:
   refractiveindex

PURPOSE:
   Returns the real part of the refractive index of water
   as a function of the wavelength of light and the
   temperature

CATEGORY:
   Physical constants

CALLING SEQUENCE:
    n = refractiveindex(lambda,T)

INPUTS:
    lambda: vacuum wavelength of light [micrometers]
    T: temperature [C]

OUTPUTS:
    n: refractive index

PROCEDURE:
1. The International Association for the Properties of Water and Steam,
   Release on the Refractive Index of Ordinary Water Substance
   as a Function of Wavelength, Temperature and Pressure (1997)
   http://www.iapws.org/relguide/rindex.pdf

2. CRC Handbook of Chemistry and Physics: Thermophysical properties
   of water and steam.

MODIFICATION HISTORY:
06/10/2007 Created by David G. Grier, New York University
03/01/2012 DGG Spelling of Celsius (sheesh)
03/11/2013 DGG Correction for temperature-dependent changes in density of water.  COMPILE_OPT.
  Double precision throughout.
06/31/2019 DGG Translated from IDL to python

Copyright (c) 2007-2019 David G. Grier
'''

import numpy as np


def refractiveindex(wavelength, temperature=24):
    '''Compute the refractive index of water'''

    Tref = 273.15      # [K] reference temperature: freezing point of water
    rhoref = 1000.     # [kg/m^3] reference density
    lambdaref = 0.589  # [um] reference wavelength

    density = (((999.83952 + 16.945176*temperature) -
                (7.9870401e-3*temperature**2 -
                 46.170461e-6*temperature**3) +
                (105.56302e-9*temperature**4 -
                 280.54235e-12*temperature**5)) *
               1.000028/(1. + 16.879850e-3*temperature))

    nT = temperature/Tref + 1.
    nrho = density/rhoref
    nlambda = wavelength/lambdaref

    A = [0.244257733,
         9.74634476e-3,
         -3.73234996e-3,
         2.68678472e-4,
         1.58920570e-3,
         2.45934259e-3,
         0.900704920,
         -1.66626219e-2]

    nlambdauv = 0.2292020
    nlambdaair = 5.432937

    B = (A[0] +
         A[1] * nrho +
         A[2] * nT +
         A[3] * nlambda**2 * nT +
         A[4] / nlambda**2 +
         A[5] / (nlambda**2 - nlambdauv**2) +
         A[6] / (nlambda**2 - nlambdaair**2) +
         A[7] * nrho**2)
    B *= nrho

    return np.sqrt((1. + 2.*B)/(1. - B))


if __name__ == '__main__':
    print(refractiveindex(0.532, 24))
