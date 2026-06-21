import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter


logger = logging.getLogger(__name__)


class Normalizer:
    '''Normalize a hologram image by its estimated background illumination.

    Parameters
    ----------
    method : str
        Normalization method:

        ``'median'`` (default)
            Divide by the scalar median of the image. Assumes spatially
            uniform illumination.

        ``'filter'``
            Estimate the background with a large-kernel median filter.
            Suppresses the rapidly-oscillating fringe pattern while
            preserving slow illumination variation. Requires no reference.

        ``'reference'``
            Divide by a prerecorded reference image or a scalar value.
            Gives the highest quality result when a background image is
            available.

    size : int
        Kernel size (pixels) for the ``'filter'`` method. Should exceed
        the largest fringe spacing expected in the holograms. Default: 51.
    reference : ndarray or float, optional
        Background for the ``'reference'`` method: either an image array
        with the same shape as the input or a scalar intensity value.
    '''

    methods = ('filter', 'median', 'reference')

    def __init__(self,
                 method: str = 'median',
                 size: int = 51,
                 reference: NDArray[float] | float | None = None) -> None:
        self.method = method
        self.size = size
        self.reference = reference

    def __call__(self, image: NDArray[float]) -> NDArray[float]:
        '''Return the image divided by its estimated background.

        Parameters
        ----------
        image : ndarray
            Raw hologram pixel values as a floating-point array.

        Returns
        -------
        normalized : ndarray
            Hologram divided by background; background pixels are ≈ 1.
        '''
        if self.method == 'reference' and self.reference is not None:
            background = np.asarray(self.reference)
            if background.ndim > 0 and background.shape != image.shape:
                logger.warning(
                    f'Reference shape {background.shape} != '
                    f'image shape {image.shape}; falling back to median')
                background = np.median(image)
        elif self.method == 'filter':
            background = median_filter(image, size=self.size)
        else:
            background = np.median(image)
        safe = np.where(background > 0, background, 1.)
        return image / safe
