from pylorenzmie.analysis.Mask import Mask
import numpy as np


class RadialMask(Mask):
    '''Pixel selection mask with radially-weighted sampling probability.

    Overrides :meth:`Mask._select` to sample pixels with a probability
    that varies with radial distance from the image center.

    When ``fraction < 0.5`` the image center has probability zero of
    being selected; probability increases toward the edges, emphasizing
    the outer ring fringes of a holographic pattern. When
    ``fraction >= 0.5`` all pixels have a nonzero selection probability;
    the distribution becomes more uniform as ``fraction`` → 1.

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
