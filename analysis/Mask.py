from dataclasses import dataclass
import numpy as np
from typing import (Optional, Tuple, Any)


@dataclass
class Mask(object):
    '''
    Mask for selecting pixels to analyze

    ...

    Properties
    ----------
    shape: tuple(int, int)
        (w, h) shape of the mask
    fraction : float
        percentage of pixels to sample
    exclude : numpy.ndarray
        boolean mask of pixels to exclude from mask

    Methods
    -------
    update():
        Create new random mask
    '''

    shape: Optional[Tuple[int, int]] = None
    fraction: float = 0.1
    exclude: Optional[np.ndarray] = None

    def __setattr__(self, prop: str, value: Any) -> None:
        super().__setattr__(prop, value)
        if prop in ['shape', 'fraction', 'exclude']:
            self.update()

    def __call__(self) -> np.ndarray:
        return self._mask

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def _select(self) -> None:
        '''Selects pixels for analysis

        Randomly chooses a proportion of pixels to include
        set by self.fraction.

        Subclasses can override this to implement other distributions
        '''
        choice = np.random.choice
        self._mask = choice([True, False],
                            size=self.shape,
                            p=[self.fraction, 1.-self.fraction])

    def update(self) -> None:
        self._select()
        if self.exclude is not None:
            self._mask[self.exclude] = False


def example():
    shape = (5, 5)
    mask = Mask()
    mask.shape = shape
    mask.fraction = 0.5
    print(mask)
    data = np.arange(25).reshape(shape)
    print(data[mask()])


if __name__ == '__main__': # pragma: no cover
    example()
