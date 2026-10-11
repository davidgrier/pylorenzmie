'''Example holograms for the documentation and the examples.

:func:`example_hologram` returns one of the images that are installed with the
package, as a normalized
:class:`~pylorenzmie.analysis.Hologram.Hologram`.  The ``example`` methods of
the classes in ``pylorenzmie.analysis`` use it.
'''

from pathlib import Path
from pylorenzmie.analysis.Hologram import Hologram


# The three images that the examples use ship with the package; the other
# tutorial images are in the source tree only.
_DATA = Path(__file__).parent / 'data'
_TUTORIALS = Path(__file__).parent.parent / 'docs' / 'tutorials'


def example_hologram(name: str = 'crop.png') -> Hologram:
    '''Return a normalized example hologram.

    Loads an image that is distributed with the package and returns it
    as a :class:`~pylorenzmie.analysis.Hologram`.

    Parameters
    ----------
    name : str, optional
        Filename of the image: ``'crop.png'`` (a 390 x 390 crop of one
        sphere), ``'image0010.png'`` (two spheres) or ``'image0400.png'``.
        In a source checkout, the other images in ``docs/tutorials/`` can
        also be loaded.  Default: ``'crop.png'``.

    Returns
    -------
    hologram : Hologram
        Hologram intensity divided by the background level (100 counts),
        shape ``(height, width)``, dtype ``float64``.
    '''
    import cv2
    path = _DATA / name
    if not path.exists():
        path = _TUTORIALS / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'No example image named {name!r}')
    return Hologram(image.astype(float) / 100.)
