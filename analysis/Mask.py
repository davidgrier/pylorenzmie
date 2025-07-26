from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


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

    shape: tuple[int, int] | None = None
    fraction: float = 0.1
    exclude: NDArray[int] | None = None

    def __setattr__(self, prop: str, value: object) -> None:
        super().__setattr__(prop, value)
        if prop in ['shape', 'fraction', 'exclude']:
            self.update()

    def __call__(self) -> NDArray[bool]:
        return self._mask

    @property
    def mask(self) -> NDArray[bool]:
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

    @classmethod
    def example(cls) -> None:
        import matplotlib.pyplot as plt

        shape = (201, 201)
        mask = cls()
        mask.shape = shape
        mask.fraction = 0.2
        print(f'fraction = {np.sum(mask())/mask().size:.2f}')
        plt.imshow(mask(), cmap='gray')
        plt.show()


if __name__ == '__main__':  # pragma: no cover
    Mask.example()
