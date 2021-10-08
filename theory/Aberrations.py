from pylorenzmie.theory.Field import Field
import numpy as np
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class ZernikeCoefficients:
    pupil: int = 0
    piston: float = 0.
    xtilt: float = 0.
    ytilt: float = 0.
    defocus: float = 0.
    xastigmatism: float = 0.
    yastigmatism: float = 0.
    xcoma: float = 0.
    ycoma: float = 0.
    spherical: float = 0.
    changed: bool = True

    def __setattr__(self, key, value):
        if key != 'changed':
            setattr(self, 'changed', True)
        super().__setattr__(key, value)

    @property
    def properties(self):
        p = self.__dict__.copy()
        p.pop('changed')
        return p

    @properties.setter
    def properties(self, properties):
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Aberrations(Field):
    '''
    Abstraction of geometric aberrations

    ...

    Properties
    ----------
    coordinates : numpy.ndarray
        [2, npts] array of x and y coordinates
    coefficients : ZernikeCoefficients

    Methods
    -------
    phase() : numpy.ndarray
        [npts] array of phase aberration values at each coordinate
    field() : numpy.ndarray
        [npts] array of complex aberration field values
    '''

    def __init__(self,
                 pupil=0,
                 coefficients=None,
                 coordinates=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._phase = 0.
        self.coefficients = coefficients or ZernikeCoefficients()
        self.pupil = pupil
        self.coordinates = coordinates

    @Field.coordinates.setter
    def coordinates(self, coordinates):
        logger.debug('Setting coordinates')
        Field.coordinates.fset(self, coordinates)
        self._coordinates_changed = True

    @property
    def pupil(self):
        return self.coefficients.pupil

    @pupil.setter
    def pupil(self, pupil):
        if pupil > 0:
            self.coefficients.pupil = pupil
            self._coordinates_changed = True

    @property
    def properties(self):
        return self.coefficients.properties

    @properties.setter
    def properties(self, properties):
        self.coefficients.properties = properties

    def _compute_polynomials(self):
        try:
            x = self.coordinates[0, :] / self.coefficients.pupil
            y = self.coordinates[1, :] / self.coefficients.pupil
        except Exception as ex:
            logger.debug(f'Could not compute: {ex}')
            self.polynomials = np.zeros(9)
            return
        rhosq = x*x + y*y
        self.polynomials = \
            dict(pupil=0.,
                 piston=1.,
                 xtilt=x,
                 ytilt=y,
                 defocus=(2.*rhosq - 1.),
                 xastigmatism=(x - y)*(x + y),
                 yastigmatism=0.5*x*y,
                 xcoma=(3.*rhosq - 2.)*x,
                 ycoma=(3.*rhosq - 2.)*y,
                 spherical=6.*rhosq*(rhosq - 1.) + 1.)

    def _compute_phase(self):
        phase = 0.
        for term in self.coefficients.properties.keys():
            a_n = getattr(self.coefficients, term)
            if a_n != 0:
                phase += a_n * self.polynomials[term]
        self._phase = phase

    def phase(self):
        if self._coordinates_changed:
            self._compute_polynomials()
        if self._coordinates_changed or self.coefficients.changed:
            self._compute_phase()
        self._coordinates_changed = False
        self.coefficients.changed = False
        return self._phase

    def field(self):
        if self._coordinates_changed or self.coefficients.changed:
            self._field = np.exp(1.j * self.phase())
        return self._field
