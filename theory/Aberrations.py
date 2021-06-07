import numpy as np


class Aberrations(object):
    '''
    Abstraction of geometric aberrations

    ...
    
    Properties
    ----------
    coordinates : numpy.ndarray
        [2, npts] array of x and y coordinates
    pupil : float
        radius of the imaging system's pupil [pixels]
    coefficients : numpy.ndarray
        9-element array containing coefficients of Zernike polynomials
        c[0] : piston
        c[1] : x tilt
        c[2] : y tilt
        c[3] : defocus
        c[4] : 0 degree astigmatism
        c[5] : 45 degree astigmatism
        c[6] : x coma
        c[7] : y coma
        c[8] : spherical aberration

    Methods
    -------
    phase() : numpy.ndarray
        [npts] array of phase aberration values at each coordinate
    field() : numpy.ndarray
        [npts] array of complex aberration field values
    '''

    def __init__(self,
                 coordinates=None,
                 pupil=None,
                 coefficients=None):
        self.coordinates = coordinates
        self.pupil = pupil
        self.coefficients = coefficients or np.zeros(9)

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates

    @property
    def pupil(self):
        return self._pupil

    @pupil.setter
    def pupil(self, pupil):
        self._pupil = pupil or 1.

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients

    @property
    def piston(self):
        return self.coefficients[0]

    @piston.setter
    def piston(self, value):
        self.coefficients[0] = value

    @property
    def xtilt(self):
        return self.coefficients[1]

    @xtilt.setter
    def xtilt(self, value):
        self.coefficients[1] = value

    @property
    def ytilt(self):
        return self.coefficients[2]

    @ytilt.setter
    def ytilt(self, value):
        self.coefficients[2] = value

    @property
    def defocus(self):
        return self.coefficients[3]

    @defocus.setter
    def defocus(self, value):
        self.coefficients[3] = value

    @property
    def xastigmatism(self):
        return self.coefficients[4]

    @xastigmatism.setter
    def xastigmatism(self, value):
        self.coefficients[4] = value

    @property
    def yastigmatism(self):
        return self.coefficients[5]

    @yastigmatism.setter
    def yastigmatism(self, value):
        self.coefficients[5] = value

    @property
    def xcoma(self):
        return self.coefficients[6]

    @xcoma.setter
    def xcoma(self, value):
        self.coefficients[6] = value

    @property
    def ycoma(self):
        return self.coefficients[7]

    @ycoma.setter
    def ycoma(self, value):
        self.coefficients[7] = value

    @property
    def spherical(self):
        return self.coefficients[8]

    @spherical.setter
    def spherical(self, value):
        self.coefficients[8] = value

    def phase(self):
        a = self.coefficients
        x = self.coordinates[0,:] / self.pupil
        y = self.coordinates[1,:] / self.pupil
        rhosq = x * x + y * y
        phase =  a[1] * x
        phase += a[2] * y
        phase += a[3] * (2. * rhosq - 1.)
        phase += a[4] * (x - y) * (x + y)
        phase += a[5] * 2. * x * y
        phase += a[6] * (3.*rhosq - 2.) * x
        phase += a[7] * (3.*rhosq - 2.) * y
        phase += a[8] * (6.*rhosq * (rhosq - 1.) + 1.)
        return phase

    def field(self):
        return np.exp(1.j * self.phase())
        
        
