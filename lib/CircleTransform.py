'''Orientation alignment transform for detecting ring-like features.'''

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import savgol_filter


class CircleTransform:
    '''Transform an image to emphasize ring-like holographic features.

    Implements the orientation alignment transform described in
    Krishnatreya & Grier, *Opt. Express* **22**, 12773 (2014).

    The kernel is cached so repeated calls on images of the same shape
    pay no recomputation cost.
    '''

    def __init__(self) -> None:
        self._kernel = np.ones((1, 1))

    def kernel(self, shape: tuple[int, int]) -> NDArray[complex]:
        '''Fourier-space orientation alignment kernel.

        Parameters
        ----------
        shape : tuple of int
            ``(ny, nx)`` shape of the image to be transformed.

        Returns
        -------
        kernel : ndarray, complex
            ``K(k) = e^{-2i\\theta} / k``, shifted for FFT pixel ordering.
        '''
        if shape == self._kernel.shape:
            return self._kernel
        ny, nx = shape
        kx = fftshift(np.linspace(-1., 1, nx, endpoint=False))
        ky = fftshift(np.linspace(-1., 1, ny, endpoint=False))
        k = np.hypot.outer(ky, kx) + 0.001
        kernel = np.subtract.outer(1.j * ky, kx) / k
        kernel *= kernel / k
        self._kernel = kernel
        return kernel

    def transform(self, image: NDArray[float]) -> NDArray[float]:
        '''Apply the orientation alignment transform.

        Parameters
        ----------
        image : ndarray
            Grayscale image data.

        Returns
        -------
        transformed : ndarray
            Array with the same shape as *image*, with ring-like features
            enhanced. Values are normalized to [0, 1].
        '''
        psi = np.empty_like(image, dtype=complex)
        psi.real = savgol_filter(image, 13, 3, 1, axis=1)
        psi.imag = savgol_filter(image, 13, 3, 1, axis=0)
        psi *= psi
        psi = fft2(psi, workers=-1, overwrite_x=True)
        psi *= self.kernel(image.shape)
        psi = ifft2(psi, workers=-1, overwrite_x=True)
        c = np.square(psi.real) + np.square(psi.imag)
        c /= np.max(c)
        return c


_transform = CircleTransform()


def circletransform(image: NDArray[float]) -> NDArray[float]:
    '''Transform an image to emphasize ring-like features.

    Convenience wrapper around :class:`CircleTransform` that reuses a
    module-level instance so the Fourier kernel is cached across calls.

    Parameters
    ----------
    image : ndarray
        Grayscale image data.

    Returns
    -------
    transformed : ndarray
        Transformed image with ring-like features enhanced.
    '''
    return _transform.transform(image)


def example() -> None:  # pragma: no cover
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt

    directory = Path(__file__).parent.parent.resolve()
    filename = directory / 'docs' / 'tutorials' / 'image0400.png'
    a = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    b = circletransform(a)
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
