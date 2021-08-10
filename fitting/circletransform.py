import numpy as np
from scipy.signal import savgol_filter


def circletransform(image,
                    extent=None,
                    order=None,
                    phase=False):
    """
    Transform image to emphasize circular features

    Parameters
    ----------
    image: numpy.ndarray
        grayscale image data
    extent: int [optional]
        Size of the convolution kernel [pixels]
        Default: 13
    order: int [optional]
        Order of filter used to compute derivatives
        Default: 3
    phase: bool [optional]
        True: return intensity and phase angle
        False: return intensity

    Returns
    -------
    transform: numpy.ndarray
        An array with the same shape as image, transformed
        to emphasize circular features.
    phase: numpy.ndarray [optional]
        Phase angle of local orientation of image gradients.

    Notes
    -----
    Algorithm described in
    B. J. Krishnatreya and D. G. Grier
    "Fast feature identification for holographic tracking:
    The orientation alignment transform,"
    Optics Express 22, 12773-12778 (2014)
    """

    extent = extent or 13
    order = order or 3
   
    # Orientational order parameter:
    # psi(r) = |\partial_x a + i \partial_y a|^2
    psi = np.empty_like(image, dtype=np.complex)
    psi.real = savgol_filter(image, extent, order, 1, axis=1)
    psi.imag = savgol_filter(image, extent, order, 1, axis=0)
    psi *= psi

    # Fourier transform of the orientational alignment kernel:
    # K(k) = e^(-2 i \theta) / k^3
    # Shift to accommodate FFT pixel ordering
    ny, nx = image.shape
    kx = np.fft.ifftshift(np.linspace(-0.5, 0.5, nx))
    ky = np.fft.ifftshift(np.linspace(-0.5, 0.5, ny))
    k = np.hypot.outer(ky, kx) + 0.001
    kernel = np.subtract.outer(1.j*ky, kx)
    kernel *= kernel / k**3

    # Convolve psi(r) with K(r) using the
    # Fourier convolution theorem
    psi = np.fft.fft2(psi)
    psi *= kernel
    psi = np.fft.ifft2(psi)

    # Transformed image is the intensity of the convolution
    intensity = psi.real*psi.real + psi.imag*psi.imag
    if phase:
       return intensity, np.angle(psi)
    else:
       return intensity
