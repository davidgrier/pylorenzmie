'''Estimate the noise in an image.'''
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

# the Laplacian-like kernel of Immerkaer (1996) responds to the noise and
# not to smooth structure; the sum of its squared entries is 36, so white
# noise of standard deviation sigma gives a response of 6 sigma
_KERNEL = np.array([[1., -2., 1.],
                    [-2., 4., -2.],
                    [1., -2., 1.]])
_MAD = 1.4826            # MAD to standard deviation, for Gaussian noise


def estimate_noise(image: NDArray[np.floating]) -> float:
    '''Estimate the standard deviation of the pixel noise in an image.

    The estimate applies a second-difference filter, which removes smooth
    structure such as the fringes of a hologram, and takes the robust
    (median absolute deviation) width of its response, so that a particle
    or sharp feature in a small part of the image does not change it.  It is
    exact for white Gaussian noise.  It does not see noise that is correlated
    over several pixels or a fixed pattern, so a fit can be worse than the
    estimate suggests; that is what a reduced chi-squared above 1 means.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional image, normalized so that the background is 1.

    Returns
    -------
    sigma : float
        Estimate of the standard deviation of the noise, in the units of
        the image.

    References
    ----------
    J. Immerkaer, "Fast noise variance estimation," Computer Vision and
    Image Understanding **64**, 300-302 (1996).
    '''
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError('an image with at least 3 pixels on a side is '
                         f'required, not shape {image.shape}')
    response = convolve(image, _KERNEL, mode='nearest')[1:-1, 1:-1]
    mad = np.median(np.abs(response - np.median(response)))
    return float(_MAD * mad / 6.)
