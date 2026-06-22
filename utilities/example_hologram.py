from pathlib import Path
from pylorenzmie.lib.lmtypes import Image


_TUTORIALS = Path(__file__).parent.parent / 'docs' / 'tutorials'


def example_hologram(name: str = 'crop.png') -> Image:
    '''Return a normalized example hologram.

    Loads an image from the package's tutorials directory and returns
    it as a normalized floating-point array.

    Parameters
    ----------
    name : str, optional
        Filename of the image in ``docs/tutorials/``.
        Default: ``'crop.png'``.

    Returns
    -------
    data : Image
        Hologram intensity divided by the background level (100 counts),
        shape ``(height, width)``, dtype ``float64``.
    '''
    import cv2
    return cv2.imread(str(_TUTORIALS / name),
                      cv2.IMREAD_GRAYSCALE).astype(float) / 100.
