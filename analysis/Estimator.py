'''Conventional estimates of the starting values of a fit.

An :class:`Estimator` obtains the center, the height and the radius of a
particle from the azimuthal average of its hologram.  The center comes first,
because the rings of a hologram are smeared by an error in it, and a smeared
profile reads as a smaller sphere: the estimator moves the center to where the
hologram is most nearly rotationally symmetric.  The spacing of the fringes
then gives the height in the paraxial approximation, and a match of the
profile to the Lorenz-Mie profile of a sphere with the given refractive index
gives the radius and refines the height.  The estimator is fast and needs no
random search over the pixels, but it cannot determine the refractive index,
which it takes as given, and it needs a clean fringe pattern around the center
of the crop.

It is one of four estimators, which return the same table of starting values
for :class:`~pylorenzmie.analysis.Optimizer.Optimizer`:
:class:`~pylorenzmie.analysis.DEEstimator.DEEstimator` (global search on the
pixels, the default of :class:`~pylorenzmie.analysis.Feature.Feature`),
:class:`~pylorenzmie.analysis.RadialEstimator.RadialEstimator` (global search
on the radial profile) and
:class:`~pylorenzmie.analysis.MLPEstimator.MLPEstimator` (a trained neural
network).
'''

from dataclasses import dataclass
from pylorenzmie.lib import Azimuthal
from pylorenzmie.lib.lmtypes import Properties, Result
from pylorenzmie.analysis.BaseEstimator import BaseEstimator
from pylorenzmie.analysis.Hologram import Hologram
from pylorenzmie.theory import Instrument, LorenzMie, Sphere
import pandas as pd
import numpy as np
from scipy.signal import argrelmin, argrelmax


_NPIXELS = 20000      # pixels used to find the center
_RING = 0.5           # width (pixels) of the rings used to find the center
_RADII = (0.1, 5.)    # range (um) searched for the radius


@dataclass
class Estimator(BaseEstimator):
    '''Estimate initial parameters of a holographic feature.

    Uses the azimuthal average of a cropped hologram to estimate the
    position and radius of the scattering particle, providing
    initial values for :class:`~pylorenzmie.analysis.Optimizer.Optimizer`.

    Inherits from :class:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator`.

    Parameters
    ----------
    instrument : Instrument
        Optical parameters of the microscope.
    n_p : float, optional
        Particle refractive index. Default: 1.5. Cannot be estimated
        from the azimuthal profile; serves as an initial value for
        the optimizer.
    k_p : float, optional
        Imaginary part of the refractive index. Default: 0.
    max_shift : float, optional
        Largest distance (pixels) by which the center of the particle may
        differ from the center of the crop.  Default: 10.  Zero keeps the
        center of the crop.

    Notes
    -----
    Call :meth:`estimate` to populate all particle parameters.
    Results are available via :attr:`properties` afterwards.

    The center is found by minimizing the variance of the pixels within rings
    of 0.5 pixel about it, on a coarse grid within ``max_shift`` of the center
    of the crop and then on finer grids.  It needs neither the radius nor the
    height, and it recovers the center to about 0.1 pixel to 1 pixel, the
    larger value for crops of 100 pixels, from a start that is 10 pixels off,
    which keeps the radius within a few percent of that of a fit.  Without it
    the radius of a small sphere is too small by 25% to 40% at that error.
    The search assumes that the feature is the only rotationally symmetric
    pattern in the crop.

    The radius is the one for which the Lorenz-Mie profile at ``n_p`` best
    matches the azimuthal average, searched on a grid of radii and of heights
    within 40% of the paraxial estimate.  The height of the best match
    replaces the paraxial estimate.

    ``predict`` is a backward-compatibility alias for
    :meth:`estimate`.

    This class provides a lightweight conventional-algorithm baseline.
    For higher-quality initial estimates — including ``n_p`` — use the
    CATCH deep neural network model [1]_ [2]_.

    References
    ----------
    .. [1] L. E. Altman and D. G. Grier,
       "CATCH: Characterizing and Tracking Colloids Holographically
       Using Deep Neural Networks,"
       J. Phys. Chem. B **124**, 1602 (2020).
    .. [2] L. E. Altman and D. G. Grier,
       "Machine learning enables precise holographic characterization
       of colloidal materials in real time,"
       Soft Matter **19**, 3002 (2023).
    '''

    instrument: Instrument
    n_p: float = 1.5
    k_p: float = 0.
    max_shift: float = 10.

    def __post_init__(self) -> None:
        self.predict = self.estimate
        self.x_p: float | None = None
        self.y_p: float | None = None
        self.z_p: float | None = None
        self.a_p: float | None = None

    @BaseEstimator.properties.getter
    def properties(self) -> Properties:
        '''Estimated particle properties.'''
        return dict(x_p=self.x_p, y_p=self.y_p,
                    z_p=self.z_p, a_p=self.a_p,
                    n_p=self.n_p, k_p=self.k_p)

    def _initialize(self, hologram: Hologram) -> None:
        self._k = self.instrument.wavenumber()
        self._data = hologram.data
        self._corner = hologram.corner
        self.x_p, self.y_p = self._find_center(hologram)
        center = (self.x_p - self._corner[0], self.y_p - self._corner[1])
        self._profile = Azimuthal.avg(self._data, center)

    def _find_center(self, hologram: Hologram) -> tuple[float, float]:
        '''Center of the most rotationally symmetric pattern.

        Minimizes the variance of the pixels within rings of 0.5 pixel
        about a trial center, on grids of decreasing size and spacing.
        '''
        x = float(hologram.coordinates[0].mean())
        y = float(hologram.coordinates[1].mean())
        if self.max_shift <= 0.:
            return x, y
        data = hologram.flat_data
        xy = hologram.flat_coordinates
        keep = np.isfinite(data)
        step = max(1, int(np.ceil(keep.sum() / _NPIXELS)))
        pixels = np.flatnonzero(keep)[::step]
        d, px, py = data[pixels], xy[0, pixels], xy[1, pixels]

        def unexplained(cx: float, cy: float) -> float:
            bins = np.floor(np.hypot(px - cx, py - cy) / _RING).astype(int)
            n = np.bincount(bins)
            occupied = n > 0
            total = np.bincount(bins, d)[occupied]
            squares = np.bincount(bins, d * d)[occupied]
            return float(np.sum(squares - total**2 / n[occupied]))

        for reach, spacing in ((self.max_shift, 2.), (2., 1.), (1., 0.25)):
            offsets = np.arange(-reach, reach + spacing / 2., spacing)
            trials = [(x + dx, y + dy) for dx in offsets for dy in offsets]
            x, y = min(trials, key=lambda c: unexplained(*c))
        return x, y

    def _estimate_z(self) -> None:
        '''Estimate axial position from fringe spacing.

        Uses the paraxial approximation: consecutive extrema at radii
        r_n satisfy Δ(r²) = 2π z_p / k.
        '''
        b = self._profile
        nmax = argrelmax(b, order=5)
        nmin = argrelmin(b, order=5)
        rn = np.sort(np.concatenate(nmax + nmin))
        zn = self._k / (2. * np.pi) * (rn[1:]**2 - rn[:-1]**2)
        self.z_p = float(np.median(zn))

    def _profile_about_center(self) -> tuple:
        '''Mean radius, mean value and pixel count of each unit ring.'''
        ny, nx = self._data.shape
        x = np.arange(nx) + self._corner[0] - self.x_p
        y = np.arange(ny) + self._corner[1] - self.y_p
        r = np.hypot.outer(y, x).ravel()
        d = self._data.ravel()
        ok = np.isfinite(d)
        r, d = r[ok], d[ok]
        bins = r.astype(int)
        n = np.bincount(bins)
        occupied = n > 0
        return (np.bincount(bins, r)[occupied] / n[occupied],
                np.bincount(bins, d)[occupied] / n[occupied],
                n[occupied])

    def _estimate_a(self) -> None:
        '''Estimate radius, and refine height, by matching the profile.

        The azimuthal average is compared with the profile of a sphere
        of index ``n_p`` on a grid of radii and heights, and then on a
        finer grid about the best match.  The weight of a ring is the
        square root of the number of its pixels.

        A match needs the right center: the profile of a feature with
        the wrong one is smeared, which :meth:`_find_center` avoids.
        '''
        if not np.isfinite(self.z_p) or self.z_p <= 0.:
            self.logger.warning(
                'No fringes found; a_p could not be estimated')
            return
        r, profile, n = self._profile_about_center()
        model = LorenzMie(coordinates=np.vstack([r, np.zeros_like(r)]),
                          instrument=self.instrument)
        if hasattr(model, 'parallel'):
            model.parallel = False
        sphere = Sphere(n_p=self.n_p, k_p=self.k_p)
        k = self.instrument.wavenumber()
        weight = np.sqrt(n)
        along = np.vstack([k * r, np.zeros_like(r), np.zeros_like(r)])

        def search(radii, heights) -> tuple[float, float]:
            '''Best (radius, height) on a grid.

            The Mie coefficients depend on the radius only, so they are
            computed once for all of the heights.
            '''
            best = (np.inf, None, None)
            for a in radii:
                sphere.a_p = a
                ab = sphere.ab(self.instrument.n_m,
                               self.instrument.wavelength)
                for z in heights:
                    kdr = along.copy()
                    kdr[2] = -k * z
                    field = model.lorenzmie(ab, kdr)
                    field[0] += np.exp(1.j * k * z)
                    holo = np.sum(field.real**2 + field.imag**2, axis=0)
                    cost = float(np.sum((weight * (holo - profile))**2))
                    if cost < best[0]:
                        best = (cost, a, z)
            return best[1:]

        z_p = self.z_p
        a_p, z_p = search(np.geomspace(*_RADII, 60),
                          z_p * np.linspace(.6, 1.4, 17))
        a_p, z_p = search(a_p * np.geomspace(.93, 1.07, 15),
                          z_p * np.linspace(.96, 1.04, 9))
        self.a_p, self.z_p = float(a_p), float(z_p)

    def estimate(self, hologram: Hologram) -> Result:
        '''Estimate particle properties from a normalized hologram.

        Parameters
        ----------
        hologram : Hologram
            Normalized hologram crop.

        Returns
        -------
        result : pandas.Series
            Estimated particle properties.
        '''
        self._initialize(hologram)
        self._estimate_z()
        self._estimate_a()
        return pd.Series(self.properties)

    @classmethod
    def example(cls) -> None:  # pragma: no cover
        from time import perf_counter
        from pylorenzmie.utilities import example_hologram

        instrument = Instrument()
        instrument.wavelength = 0.447
        instrument.magnification = 0.048
        instrument.n_m = 1.34

        estimator = cls(instrument=instrument)
        hologram = example_hologram()

        print(f'{cls.__name__} example')
        start = perf_counter()
        result = estimator.estimate(hologram)
        print(f'Time: {perf_counter() - start:.3f} s')
        print(result)


if __name__ == '__main__':  # pragma: no cover
    Estimator.example()
