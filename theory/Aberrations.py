from .Field import Field
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
        self.pupil = pupil
        self.coefficients = coefficients or ZernikeCoefficients()
        self.coordinates = coordinates

    @Field.coordinates.setter
    def coordinates(self, coordinates):
        logger.debug('Setting coordinates...')
        Field.coordinates.fset(self, coordinates)
        self._coordinates_changed = True

    @property
    def pupil(self):
        return self.coefficients.pupil

    @pupil.setter
    def pupil(self, pupil):
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
        except (AttributeError, TypeError) as ex:
            logger.debug('Could not compute: {}'.format(ex))
            return
        rhosq = x*x + y*y
        self.polynomials = [1.,
                            x, y,
                            2.*rhosq - 1.,
                            (x - y)*(x + y), 2.*x*y,
                            (3.*rhosq - 2.) * x, (3.*rhosq - 2.) * y,
                            6.*rhosq * (rhosq - 1.) + 1.]
        self._coordinates_changed = False

    def _compute_phase(self):
        phase = 0.
        coefficients = self.coefficients.properties.values()
        for a_n, phase_n in zip(coefficients, self.polynomials):
            if a_n != 0:
                phase += a_n * phase_n
        self._phase = phase
        self.coefficients.changed = False

    def phase(self):
        if self._coordinates_changed:
            self._compute_polynomials()
            self._compute_phase()
        if self.coefficients.changed:
            self._compute_phase()
        return self._phase

    def field(self):
        return np.exp(1.j * self.phase())
