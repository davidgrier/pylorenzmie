'''Normalization of holograms by their background.

A hologram must be divided by the intensity of the background illumination
before it is analyzed, so that the background is 1.  A :class:`Normalizer`
does so by one of three methods: the median of the image, for uniform
illumination; a large median filter, which estimates a slowly varying
background from the image itself; or a recorded reference image, which is the
most accurate, because it also removes dust and fringes of the optics.  For a
video, :class:`~pylorenzmie.utilities.vmedian.VMedian` supplies a running
median background.  Subtract the dark count of the camera from the image and
the reference before normalizing.
'''

import os
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter


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
        with the same shape as the input or a positive scalar intensity.

    Raises
    ------
    ValueError
        If *method* is not one of :attr:`methods`, or if *reference* is a
        scalar that is not positive and finite.

    Notes
    -----
    A pixel where the background is not positive has no meaningful
    normalized value, and is NaN in the result.  Masks exclude such pixels.
    The normalizer does not guess: calling it with the ``'reference'``
    method and no reference, or with a reference image of another shape
    than the image, raises :exc:`ValueError`.
    '''

    methods = ('filter', 'median', 'reference')

    #: Name of the file that the reference was read from, if it was.
    source: str | None = None

    def __init__(self,
                 method: str = 'median',
                 size: int = 51,
                 reference: NDArray[np.floating] | float | None = None) -> None:
        self.method = method
        self.size = size
        self.reference = reference

    @classmethod
    def read(cls, background: str | os.PathLike | float, **kwargs) -> Self:
        '''A normalizer that divides by a background that is read.

        Parameters
        ----------
        background : str, path-like or float
            A number is the intensity of a uniform background.  A string
            that reads as a number is the same.  Any other string, and a
            path, is the name of an image file, which is read as 8-bit gray
            levels, as lmtool reads holograms.
        **kwargs
            Passed to the constructor (``size``).

        Returns
        -------
        Normalizer
            With the ``'reference'`` method.  :attr:`source` is the name of
            the file, if there is one.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not an image that can be read, or the intensity
            is not positive and finite.
        '''
        if not isinstance(background, os.PathLike):
            try:
                value = float(background)
            except (TypeError, ValueError):
                if not isinstance(background, str):
                    raise
            else:
                return cls('reference', reference=value, **kwargs)
        path = Path(background)
        if not path.is_file():
            raise FileNotFoundError(f'Background not found: {str(path)!r}')
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Could not read the background: {str(path)!r}')
        normalizer = cls('reference', reference=image.astype(float), **kwargs)
        normalizer.source = str(path)
        return normalizer

    @property
    def method(self) -> str:
        '''Normalization method, one of :attr:`methods`.'''
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        if method not in self.methods:
            raise ValueError(f'Unknown method {method!r}: choose from '
                             + ', '.join(self.methods))
        self._method = method

    @property
    def reference(self) -> NDArray[np.floating] | float | None:
        '''Background of the ``'reference'`` method, an image or a number.'''
        return self._reference

    @reference.setter
    def reference(self, reference: NDArray[np.floating] | float | None
                  ) -> None:
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            if reference.ndim == 0:
                if not (np.isfinite(reference) and reference > 0):
                    raise ValueError('A scalar background must be positive '
                                     f'and finite, not {float(reference)}')
                reference = float(reference)
        self._reference = reference
        self.source = None

    def __call__(self, image: NDArray[np.floating]) -> NDArray[np.floating]:
        '''Return the image divided by its estimated background.

        Parameters
        ----------
        image : ndarray
            Raw hologram pixel values as a floating-point array.

        Returns
        -------
        normalized : ndarray
            Hologram divided by background; background pixels are ≈ 1.
            Pixels where the background is not positive are NaN.

        Raises
        ------
        ValueError
            If the method is ``'reference'`` and there is no reference, or
            the reference is an image of a different shape.
        '''
        if self.method == 'reference':
            if self.reference is None:
                raise ValueError("The 'reference' method needs a reference")
            background = np.asarray(self.reference)
            if background.ndim > 0 and background.shape != image.shape:
                raise ValueError(
                    f'The reference has shape {background.shape} but the '
                    f'image has shape {image.shape}')
        elif self.method == 'filter':
            background = median_filter(image, size=self.size)
        else:
            background = np.median(image)
        return image / np.where(background > 0, background, np.nan)
