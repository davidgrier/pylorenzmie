import numpy as np
from numpy.polynomial import Polynomial

'''Physical properties of water'''


def density(temperature: float = 24.) -> float:
    '''Returns the density of water

    Accounts for dependence of density on temperature

    Source: CRC Handbook of Chemistry and Physics:
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
    p = Polynomial([999.83952, 16.945176, -7.9870401e-3,
                    46.170461e-6, 105.56302e-9, -280.54235e-12])
    return p(temperature) * 1.000028/(1. + 16.879850e-3 * temperature)


def refractiveindex(wavelength: float = 0.589,
                    temperature: float = 24) -> float:
    '''Returns the refractive index of water

    Accounts for dispersion and temperature dependence

    Source: The International Association for the Properties
    of Water and Steam (IAPWS),
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

    Tref = 273.15      # [K] freezing point of water
    rhoref = 1000.     # [kg/m^3] reference density
    lambdaref = 0.589  # [um] reference wavelength

    nT = temperature/Tref + 1.
    nrho = density(temperature)/rhoref
    nlambda = np.square(wavelength/lambdaref)
    nlambdauv = np.square(0.2292020)
    nlambdaair = np.square(5.432937)

    B = nrho * (0.244257733 +
                9.74634476e-3 * nrho +
                -3.73234996e-3 * nT +
                2.68678472e-4 * nlambda * nT +
                1.58920570e-3 / nlambda +
                2.45934259e-3 / (nlambda - nlambdauv) +
                0.900704920 / (nlambda - nlambdaair) +
                -1.66626219e-2 * np.square(nrho))

    return np.sqrt((1. + 2.*B)/(1. - B))


if __name__ == '__main__':
    print('Properties of water')
    print(f'{density(24.) = :.2e} kg/m^3')
    print(f'{refractiveindex(0.447, 24.) = :.4f}')
