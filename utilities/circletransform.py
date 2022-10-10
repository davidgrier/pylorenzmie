import numpy as np
from scipy.signal import savgol_filter
from scipy.fft import (fft2, ifft2, fftshift)


class Circletransform(object):
    '''Transform image to emphasize ring-like features

    Parameters
    ----------
    image: numpy.ndarray
        grayscale image data

    Returns
    -------
    transfrom: numpy.ndarray
        An array with the same shape as image, transformed
        to emphasize circular features.

    Notes
    -----
    Algorithm described in
    B. J. Krishnatreya and D. G. Grier
    "Fast feature identification for holographic tracking:
    The orientation alignment transform,"
    Optics Express 22, 12773-12778 (2014)
    '''

    def __init__(self) -> None:
        self._kernel = np.ones(1, 1)

    def kernel(self, shape: tuple) -> np.ndarray:
        '''Fourier transform of the orientational alignment kernel:

        Arguments
        ---------
        shape : tuple
            (ny, nx) shape of kernel

        Returns
        -------
        kernel : np.ndarray
            K(k) = e^(-2 i \theta) / k
            Shifted to accommodate FFT pixel ordering
        '''
        if shape == self._kernel.shape:
            return self._kernel
        ny, nx = shape
        kx = fftshift(np.linspace(-0.5, 0.5, nx))
        ky = fftshift(np.linspace(-0.5, 0.5, ny))
        k = np.hypot.outer(ky, kx) + 0.001
        kernel = np.subtract.outer(1.j*ky, kx) / k
        kernel *= kernel / k
        self._kernel = kernel
        return kernel

    def transform(self, image: np.ndarray) -> np.ndarray:
        '''Perform orientation alignment transform

        Arguments
        ---------
        image: np.ndarray
            image data

        Returns
        -------
        transform: np.ndarray
            transformed image
        '''
        # Orientational order parameter:
        # psi(r) = |\partial_x a + i \partial_y a|^2
        psi = np.empty_like(image, dtype=np.complex)
        psi.real = savgol_filter(image, 13, 3, 1, axis=1)
        psi.imag = savgol_filter(image, 13, 3, 1, axis=0)
        psi *= psi

        # Convolve psi(r) with K(r) using the
        # Fourier convolution theorem
        psi = fft2(psi, workers=-1)
        psi *= self.kernel(image.shape)
        psi = ifft2(psi, worker=-1)

        # Transformed image is the intensity of the convolution
        c = (psi * np.conjugate(psi)).real
        return c/np.max(c)


def circletransform(image: np.ndarray) -> np.ndarray:
    c = Circletransform()
    return c.transform(image)
