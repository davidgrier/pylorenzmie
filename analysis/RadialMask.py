from pylorenzmie.analysis.Mask import Mask
import numpy as np


class RadialMask(Mask):
    '''Pixel selection mask with radially-weighted sampling probability.

    Overrides :meth:`Mask._select` to sample pixels with a probability
    that varies linearly with distance from the image center.

    When ``fraction < 0.5`` the sampling probability decreases toward
    the edges (center-weighted). When ``fraction >= 0.5`` it increases
    toward the edges (edge-weighted), which emphasizes the outer fringes
    of a holographic ring pattern.

    Inherits all parameters from :class:`Mask`.
    '''

    def _select(self) -> None:
        f = self.fraction
        a, b = (0., 2. * f) if (f < 0.5) else (2. * f - 1., 2. * (1. - f))

        h, w = self.shape
        x = 2. * np.arange(w) / (w - 1.) - 1.
        y = 2. * np.arange(h) / (h - 1.) - 1.

        x = a + b * x
        y = a + b * y
        p = np.hypot.outer(y, x)
        self._mask = p >= np.random.random_sample(self.shape)


if __name__ == '__main__':  # pragma: no cover
    RadialMask.example()
