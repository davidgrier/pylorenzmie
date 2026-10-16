Conventions
===========

These conventions are used everywhere in the package.  Most of the surprises
that new users meet are one of them.

Units and coordinates
---------------------

* **Lengths are in micrometers**, except coordinates in an image, which are in
  **pixels**.  The radius ``a_p`` of a sphere and the wavelength are in
  micrometers; ``x_p``, ``y_p`` and ``z_p`` are in pixels.  The
  ``magnification`` of an :class:`~pylorenzmie.theory.Instrument.Instrument`
  is the size of a pixel in the sample, in micrometers per pixel, so
  ``a_p / magnification`` is the radius in pixels.
* ``x_p`` runs along the columns of an image and ``y_p`` along the rows, and an
  image is indexed ``image[y, x]``.  Pixel coordinates are arrays of shape
  ``(2, npts)`` or ``(3, npts)`` (the third row, the axial coordinate, is
  zero) from :func:`~pylorenzmie.lib.meshgrid.meshgrid`.
* ``z_p`` is the distance of a particle from the focal plane.  A particle above
  the focal plane, between the focal plane and the light source, has a
  positive ``z_p`` and produces a diverging wave; the default is 100 pixels.
* The illumination is a plane wave that propagates along -z and is polarized
  along x.  The scattered field is returned in Cartesian components
  ``(x, y, z)`` as an array of shape ``(3, npts)``, or in spherical components
  with ``cartesian=False``.

Clusters
--------

The position of a :class:`~pylorenzmie.theory.Cluster.Cluster` is the **origin
of the coordinates of its members**.  A sphere at ``r_p = (10, 0, 50)`` in a
cluster at ``(100, 100, 100)`` is at ``(110, 100, 150)`` in the image.  The
default position of a cluster is ``z_p = 100``, not the focal plane: set
``x_p``, ``y_p`` and ``z_p`` of the cluster to zero to give the members
absolute coordinates.

Holograms and fits
------------------

* A **hologram is normalized**: its background is 1, so the mean of an image
  without particles is 1.  Divide a raw image by its background (after
  subtracting the dark count of the camera) with a
  :class:`~pylorenzmie.utilities.normalizer.Normalizer`.
* The ``noise`` of an instrument is the noise of the camera as a fraction of
  the background.  A fit weights each pixel by it, so the reduced chi-squared
  that a fit reports is about 1 only if the noise is right and the model
  explains the data.  A reduced chi-squared of a robust fit is computed from
  the robust cost and is not comparable with one from least squares.
* The table that an optimizer returns holds each fitted parameter ``p``
  followed by its uncertainty ``dp``, then ``success``, ``npix`` (the number
  of pixels used) and ``redchi``.
* Mie coefficients ``ab`` have the shape ``(n_orders, 2)`` and are complex;
  row 0 is not used.
* The wavelength of an instrument is the wavelength **in vacuum**, and
  ``n_m`` is the refractive index of the medium.
  :meth:`~pylorenzmie.theory.Instrument.Instrument.wavenumber` returns the
  wave number in the medium, in radians per pixel by default.

Aberrations
-----------

The coefficients of :func:`~pylorenzmie.theory.Aberrated.Aberrated` are
expressed in **wavelengths of the illumination in vacuum**: a coefficient of 1
corresponds to the amplitude :math:`\alpha_n = \lambda` of the operator
formalism, not to one wavelength of classical wavefront error at the edge of
the pupil.  The two differ by a power of the refractive index of the medium.
Coma, astigmatism and distortion depend on the position of the aberration
center (``x_c``, ``y_c``, in pixels) relative to the particle; defocus and
spherical aberration do not.  Defocus is nearly degenerate with ``z_p``.

:func:`~pylorenzmie.theory.Distorted.Distorted` uses the same coefficients,
units and sign, and adds field curvature.  It applies the aberrations as
flows of the image plane instead of as phase factors.  The flows of different
aberrations do not commute, so its default ``mode='composed'`` defines the
coefficients together with the order of the operators (the default is the
order of ``pylorenzmie.theory.generators.NAMES``; ``order`` changes it).
``mode='summed'`` integrates one flow of the sum of the generators and does
not depend on the order.  The two agree to first order in the coefficients
and differ when the aberration is large or the particle is close to the focal
plane.  Report the mode and the order with fitted coefficients.

Randomness
----------

Searches and pixel masks are random.  Give a
:class:`~pylorenzmie.analysis.Mask.Mask` a ``seed``, and an estimator a
``seed``, for a reproducible analysis.  The masks and the estimators use their
own random generators and do not read or change NumPy's global state.
