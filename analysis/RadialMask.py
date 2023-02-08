from pylorenzmie.analysis.Mask import (Mask, example)
import numpy as np


class RadialMask(Mask):

    def _select(self) -> None:
        if self.shape is None:
            self._mask = None
            return
        f = self.fraction
        a, b = (0., 2.*f) if (f < 0.5) else (2.*f-1., 2.*(1.-f))
        
        w, h = self.shape
        x = 2.*np.arange(w)/(w - 1.) - 1.
        y = 2.*np.arange(h)/(h - 1.) - 1.

        x = a + b*x
        y = a + b*y
        p = np.hypot.outer(x, y)
        # p = np.add.outer(x*x, y*y)
        sample = np.random.random_sample(self.shape)
        self._mask = p >= sample
        


if __name__ == '__main__':  # pragma: no cover
    example(RadialMask)
