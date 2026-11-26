'''Read an image file as a gray-scale array.'''

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

__all__ = ['imread']


def imread(path: str | Path) -> NDArray[Any] | None:
    '''Read an image file as a gray-scale array, at its own bit depth.

    The values are the ones in the file: a camera that records 10 or 12 bits
    in a 16-bit file gives values up to 1023 or 4095, and a dark count or a
    background that is given as a number is in the same units.

    Parameters
    ----------
    path : str or pathlib.Path
        Name of the image file.

    Returns
    -------
    image : numpy.ndarray or None
        Intensities, shape ``(height, width)``, or ``None`` if the file is
        missing or is not an image that can be read.  The dtype is ``uint8``
        for an image of eight bits, ``uint16`` for a gray image of more bits
        whose values fit in it (``int32`` otherwise), and ``float32`` for a
        floating-point image.

    Notes
    -----
    Color images are converted to eight-bit gray.  Pillow reads color images
    of 16 bits per channel at eight bits, so record those in gray.
    '''
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode.startswith('I'):
                data = np.asarray(image)
                limits = np.iinfo(np.uint16)
                if limits.min <= data.min() and data.max() <= limits.max:
                    return data.astype(np.uint16)
                return data.astype(np.int32)
            if image.mode == 'F':
                return np.asarray(image, dtype=np.float32)
            return np.asarray(image.convert('L'))
    except (OSError, UnidentifiedImageError, ValueError):
        return None
