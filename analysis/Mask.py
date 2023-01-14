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

    shape: Tuple[int, int]
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

    def _uniform(self) -> None:
        p = self.fraction
        choice = np.random.choice
        self._mask = choice([True, False], self.shape, [p, 1.-p])

    def update(self) -> None:
        self._uniform()
        if self.exclude is not None:
            self._mask[self.exclude] = False


def example():
    shape = (5, 5)
    mask = Mask(shape)
    mask.fraction = 0.5
    print(mask)
    data = np.arange(25).reshape(shape)
    print(data[mask()])


if __name__ == '__main__': # pragma: no cover
    example()
