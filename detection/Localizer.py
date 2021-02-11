import numpy as np
from scipy.signal import savgol_filter
import trackpy as tp


class Localizer(object):
    '''Identify and localize features in holograms

    Properties
    ----------
    nfringes : int
        Number of interference fringes used to determine feature extent
        Default: 20
    maxrange : int
        Maximum extent of feature [pixels]
        Default: 400
    tp_opts : dict
        Dictionary of options for trackpy.locate()
        Default: dict(diameter=31, minmass=30)
    
    Methods
    -------
    predict(image) :
        Returns centers and bounding boxes of features
        detected in image
    '''
    def __init__(self,
                 tp_opts=None,
                 nfringes=None,
                 maxrange=None,
                 **kwargs):
        self._tp_opts = tp_opts or dict(diameter=31, minmass=30)
        self._nfringes = nfringes or 20
        self._maxrange = maxrange or 400
        self._shape = None

    def predict(self, image):
        '''
        Localize features in normalized holographic microscopy images
        
        Parameters
        ----------
        image : array_like
            image data

        Returns
        -------
        centers : numpy.array
            (x, y) coordinates of feature centers
        bboxes : tuple
            ((x0, y0), w, h) bounding box of feature
        '''
        a = self._circletransform(image)
        a /= np.max(a)
        features = tp.locate(a, **self._tp_opts)

        nfeatures = len(features)
        if nfeatures == 0:
            return None, None

        centers = features[['x', 'y']].to_numpy()
        bboxes = []
        for center in centers:
            extent = self._extent(image, center)
            r0 = tuple((center - extent/2).astype(int))
            bboxes.append((r0, extent, extent))
        return centers, bboxes

    def _kernel(self, image):
        '''
        Fourier transform of the orientational alignment kernel:
        K(k) = e^(-2 i \theta) / k^3

        kernel ordering is shifted to accommodate FFT pixel ordering

        Parameters
        ----------
        image : numpy.ndarray
            image shape used to compute kernel

        Returns
        -------
        kernel : numpy.ndarray
            orientation alignment kernel in Fourier space
        '''
        if image.shape != self._shape:
            self._shape = image.shape
            ny, nx = image.shape
            kx = np.fft.ifftshift(np.linspace(-0.5, 0.5, nx))
            ky = np.fft.ifftshift(np.linspace(-0.5, 0.5, ny))
            k = np.hypot.outer(ky, kx) + 0.001
            kernel = np.subtract.outer(1.j*ky, kx)
            kernel *= kernel / k**3
            self.__kernel = kernel
        return self.__kernel

    def _circletransform(self, image):
        """
        Transform image to emphasize circular features

        Parameters
        ----------
        image : numpy.ndarray
            grayscale image data

        Returns
        -------
        transform : numpy.ndarray
            An array with the same shape as image, transformed
            to emphasize circular features.

        Notes
        -----
        Algorithm described in
        B. J. Krishnatreya and D. G. Grier
        "Fast feature identification for holographic tracking:
        The orientation alignment transform,"
        Optics Express 22, 12773-12778 (2014)
        """
        
        # Orientational order parameter:
        # psi(r) = |\partial_x a + i \partial_y a|^2
        psi = np.empty_like(image, dtype=np.complex)
        psi.real = savgol_filter(image, 13, 3, 1, axis=1)
        psi.imag = savgol_filter(image, 13, 3, 1, axis=0)
        psi *= psi

        # Convolve psi(r) with K(r) using the
        # Fourier convolution theorem
        psi = np.fft.fft2(psi)
        psi *= self._kernel(image)
        psi = np.fft.ifft2(psi)

        # Transformed image is the intensity of the convolution
        return psi.real**2 + psi.imag**2

    def _extent(self, norm, center):
        '''
        Radius of feature based on counting diffraction fringes

        Parameters
        ----------
        norm : array_like
            Normalized image data
        center : tuple
            (x_p, y_p) coordinates of feature center

        Returns
        -------
        extent : int
            Extent of feature [pixels]
        '''
        b = self._aziavg(norm, center) - 1.
        ndx = np.where(np.diff(np.sign(b)))[0] + 1
        if len(ndx) <= self._nfringes:
            extent = self._maxrange
        else:
            extent = ndx[self._nfringes]
        return extent

    def _aziavg(self, data, center):
        '''Azimuthal average of data about center

        Parameters
        ----------
        data : array_like
            image data
        center : tuple
            (x_p, y_p) center of azimuthal average

        Returns
        -------
        avg : array_like
            One-dimensional azimuthal average of data about center
        '''
        x_p, y_p = center
        ny, nx = data.shape
        x = np.arange(nx) - x_p
        y = np.arange(ny) - y_p

        d = data.ravel()
        r = np.hypot.outer(y, x).astype(np.int).ravel()
        nr = np.bincount(r)
        avg = np.bincount(r, d) / nr
        return avg
