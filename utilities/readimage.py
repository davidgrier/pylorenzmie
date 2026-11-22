'''Read an image file as an 8-bit gray-scale array.'''

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

__all__ = ['imread']


def imread(path: str | Path) -> NDArray[np.uint8] | None:
    '''Read an image file as an 8-bit gray-scale array.

    Color images are converted to gray, and images of more than eight bits
    per pixel are scaled to eight bits, as OpenCV's ``IMREAD_GRAYSCALE``
    does.

    Parameters
    ----------
    path : str or pathlib.Path
        Name of the image file.

    Returns
    -------
    image : numpy.ndarray or None
        Intensities, shape ``(height, width)``, dtype ``uint8``, or
        ``None`` if the file is missing or is not an image that can be read.
    '''
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode in ('I', 'I;16', 'I;16L', 'I;16B'):
                data = np.asarray(image, dtype=np.int64)
                return np.clip(data >> 8, 0, 255).astype(np.uint8)
            return np.asarray(image.convert('L'))
    except (OSError, UnidentifiedImageError, ValueError):
        return None
