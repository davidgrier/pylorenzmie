import numpy as np

'''Physical properties of water'''


def density(temperature: float = 24.) -> float:
    '''Returns the density of water

    Accounts for dependence of density on temperature:
    CRC Handbook of Chemistry and Physics:
    Thermophysical properties of water and steam.

    Arguments
    ---------
    temperature: float
        Temperature [celsius]
        Default: 24 C

    Returns
    -------
    density: float
        Density of water [kg/m^3]
    '''
    return (((999.83952 + 16.945176 * temperature) -
             (7.9870401e-3 * temperature**2 -
              46.170461e-6 * temperature**3) +
             (105.56302e-9 * temperature**4 -
              280.54235e-12 * temperature**5)) *
            1.000028/(1. + 16.879850e-3 * temperature))


def refractiveindex(wavelength: float = 0.589,
                    temperature: float = 24) -> float:
    '''Returns the refractive index of water

    Accounts for dispersion and temperature dependence
    The International Association for the Properties of Water and Steam,
    Release on the Refractive Index of Ordinary Water Substance
    as a Function of Wavelength, Temperature and Pressure (1997)
    http://www.iapws.org/relguide/rindex.pdf

    Arguments
    ---------
    wavelength: float
        Wavelength of light [um]
        Default: 0.589 (sodium D line)

    temperature: float
        Temperature of water [celsius]
        Default: 24 C

    Returns
    -------
    refractive index: float
        Refractive index of water at the specified temperature
        and wavelength
    '''

    Tref = 273.15      # [K] reference temperature: freezing point of water
    rhoref = 1000.     # [kg/m^3] reference density
    lambdaref = 0.589  # [um] reference wavelength

    nT = temperature/Tref + 1.
    nrho = density(temperature)/rhoref
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
    print('Properties of water')
    print(f'{density(24.) = :.2e} kg/m^3')
    print(f'{refractiveindex(0.447, 24.) = :.4f}')
