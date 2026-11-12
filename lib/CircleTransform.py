'''Orientation alignment transform for detecting ring-like features.

The implementation lives in `dgimage <https://dgimage.readthedocs.io>`_, the
library of image transformations that pylorenzmie, QVideo and other projects of
the group share; this module keeps the names that pylorenzmie has always
offered: :class:`CircleTransform` and :func:`circletransform`.  See
:mod:`dgimage.circletransform`, which also has the options of a faster,
less accurate transform that the localizer does not use.
'''

import numpy as np

from dgimage.circletransform import CircleTransform, circletransform

__all__ = ['CircleTransform', 'circletransform']


def example() -> None:  # pragma: no cover
    import matplotlib.pyplot as plt
    from pylorenzmie.utilities import example_hologram

    a = example_hologram('image0400.png').data
    b = circletransform(np.asarray(a))
    fig, (axa, axb) = plt.subplots(nrows=2, sharex=True, sharey=True)
    axa.imshow(a, cmap='gray')
    axb.imshow(b, cmap='gray')
    axa.axis('off')
    axb.axis('off')
    axa.set_xlim(400, 1050)
    axa.set_ylim(500, 950)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':  # pragma: no cover
    example()
