'''A sphere with one shell as a scatterer for Lorenz-Mie microscopy.

A :class:`LayeredSphere` is a :class:`~pylorenzmie.theory.Sphere.Sphere`, the
core, with a concentric shell of thickness ``t_s`` and refractive index
``n_s + i k_s``.  It computes the Mie scattering coefficients of the coated
sphere with the recursion of Yang for a layered sphere, which is stable for
thin shells and for large spheres.  The calculators in ``pylorenzmie.theory``
use the coefficients as they use those of a homogeneous sphere, so that a
model, an estimator or an optimizer takes a layered sphere wherever it takes
a sphere.

What a hologram can tell about a shell depends on the shell.  A thick shell
(50 nm and more) or one whose index differs much from that of the core
changes the hologram enough to measure its thickness and its index.  For a
thin shell with an index near that of the core, such as a layer of protein on
a polystyrene sphere, the hologram depends on the thickness and the index only
through the product ``t_s * (n_s**2 - n_m**2)`` to first order, so that fits
cannot separate the two: fix ``n_s`` (in ``Optimizer.fixed``), or limit it
(``Optimizer.bounds``).  A homogeneous sphere that fits such a hologram has
nearly the radius of the core plus half the thickness of the shell.

Lengths are in micrometers and positions in pixels, as everywhere in the
package.

See Also
--------
~pylorenzmie.theory.Sphere.Sphere : the homogeneous sphere.
'''

from dataclasses import dataclass

import numpy as np

from pylorenzmie.lib.lmtypes import Coefficients, Properties
from pylorenzmie.theory import Particle
from pylorenzmie.theory._mie import layered_ab, mie_ab
from pylorenzmie.theory.Sphere import ModelDomainError, Sphere


@dataclass
class LayeredSphere(Sphere):
    '''Sphere with one concentric shell.

    The attributes of a :class:`~pylorenzmie.theory.Sphere.Sphere`
    describe the core: ``a_p`` is the radius of the core, ``n_p`` and ``k_p``
    its refractive index.  The radius of the whole sphere is
    :attr:`outer_radius`.

    Attributes
    ----------
    t_s : float
        Thickness of the shell, in μm.  Default: 0.05.  It must not be
        negative.
    n_s : float
        Refractive index of the shell.  Default: 1.45.
    k_s : float
        Absorption coefficient of the shell.  Default: 0.

    Notes
    -----
    A coefficient is computed for a core of positive radius, a shell that is
    not negative in thickness, and a whole sphere that is no larger than
    ``MAX_RADIUS``.  Otherwise it raises a
    :class:`~pylorenzmie.theory.Sphere.ModelDomainError`, which
    :class:`~pylorenzmie.analysis.Optimizer.Optimizer` takes as a step of the
    fit that it must reject.

    References
    ----------
    1. W. Yang, "Improved recursive algorithm for light scattering
       by a multilayered sphere," *Applied Optics* **42**, 1710–1720
       (2003).
    2. C. F. Bohren and D. R. Huffman, *Absorption and Scattering
       of Light by Small Particles* (Wiley, 1983), Sec. 8.1.
    '''

    t_s: float = 0.05
    n_s: float = 1.45
    k_s: float = 0.

    @property
    def outer_radius(self) -> float:
        '''Radius of the whole sphere, in μm (``a_p + t_s``).

        Returns
        -------
        float
        '''
        return self.a_p + self.t_s

    @Particle.properties.getter
    def properties(self) -> Properties:
        return {**super().properties,
                't_s': self.t_s,
                'n_s': self.n_s,
                'k_s': self.k_s}

    def ab(self,
           n_m: float | complex = 1.42,
           wavelength: float = 0.532) -> Coefficients:
        '''Mie scattering coefficients for the coated sphere.

        Parameters
        ----------
        n_m : float or complex
            Refractive index of the medium. Default: 1.42.
        wavelength : float
            Vacuum wavelength of the illuminating light, in μm.
            Default: 0.532.

        Returns
        -------
        ab : numpy.ndarray, shape (n_terms, 2), dtype complex
            Mie scattering coefficients.

        Raises
        ------
        ModelDomainError
            If the core does not have a positive radius, the shell has a
            negative thickness, the sphere is larger than ``MAX_RADIUS``, or
            the sum needs more than ``MAX_ORDERS`` partial waves.  It is a
            :class:`ValueError`.
        '''
        if not self.a_p > 0.:
            raise ModelDomainError('the core of a layered sphere needs a '
                                   f'positive radius, not a_p = {self.a_p:g} '
                                   'um')
        if not self.t_s >= 0.:
            raise ModelDomainError('the shell of a layered sphere cannot be '
                                   f'thinner than none, not t_s = '
                                   f'{self.t_s:g} um')
        Sphere.check_radius(self.outer_radius)

        k = 2. * np.pi * np.real(n_m) / wavelength   # wave number in medium
        x = k * self.a_p                             # core
        y = k * self.outer_radius                    # whole sphere
        m1 = (self.n_p + 1.j * self.k_p) / n_m
        m2 = (self.n_s + 1.j * self.k_s) / n_m

        # the order of the sum is set by the largest of the size parameters
        # in the layers
        nmax = Sphere.wiscombe_yang(y, max(abs(m1) * x / y, abs(m2)))
        Sphere.check_orders(nmax, self.outer_radius, max(self.n_p, self.n_s))

        if self.t_s == 0.:
            return mie_ab(x, m1, nmax)
        return layered_ab(x, y, m1, m2, nmax)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        '''Run a demonstration of the class, for use as a script.'''
        from time import perf_counter

        s = cls(a_p=0.5, n_p=1.59, t_s=0.02, n_s=1.45)
        print(s)
        ab = s.ab(1.34, 0.447)
        print(f'{ab.shape = }')
        start = perf_counter()
        s.ab(1.34, 0.447)
        print(f'time = {perf_counter() - start:.2e} s')


if __name__ == '__main__':  # pragma: no cover
    LayeredSphere.example()
