Extending pylorenzmie
=====================

The pieces of the pipeline are replaceable.  This page says what a new piece
has to provide.  Read :doc:`architecture` and :doc:`conventions` first.

A new calculator
----------------

A calculator evaluates the partial-wave sum faster, on different hardware or
with a different tool.  Subclass
:class:`~pylorenzmie.theory.LorenzMie.LorenzMie` and override

``lorenzmie(ab, kdr, cartesian=True, bohren=True)``
    the scattered field of one particle.  ``ab`` holds the Mie coefficients
    (``(n_orders, 2)``) and ``kdr`` the displacement of every pixel from the
    particle times the wave number, shape ``(3, npts)``; return the complex
    field, shape ``(3, npts)``, in Cartesian components if ``cartesian`` is true
    and in spherical components otherwise.  ``bohren`` selects the sign
    convention of the axial coordinate.
``_allocate()``
    called when the coordinates change, to allocate the working arrays.

and set the class attribute ``method`` to a string that contains
``'numpy'`` and names the calculator, for example ``'mine numpy'``: an
optimizer is compatible with a model if its own ``method`` is a substring of
the model's.

Optional features:

* ``jac()`` and ``jac_params``: if the calculator can return the derivative of
  the hologram with respect to the particle parameters, an optimizer uses it
  whenever every parameter that it fits is in ``jac_params``.
* ``_aberration_coefficients()``: a calculator that evaluates the field inside
  a compiled kernel can apply the aberrations of
  :func:`~pylorenzmie.theory.Aberrated.Aberrated` itself; the existing Metal
  and JAX calculators are examples.  Override ``scattered_field`` only if you
  must, and then make sure that a shortcut that skips it is not used for the
  aberrated subclass.
* Add the calculator to ``pylorenzmie/theory/__init__.py`` so that
  ``PYLORENZMIE_BACKEND`` can select it.  An optional dependency must be
  imported there lazily, and a missing one must raise :exc:`ImportError`.

Test a new calculator against the NumPy calculator: the holograms of a sphere
and of a cluster, both field conventions, a particle below the focal plane,
and a pixel on the axis of the particle.  A calculator that needs hardware
that the continuous-integration machines lack must skip its tests when the
hardware is absent, and say in its documentation what has been tested.

A new particle
--------------

Subclass :class:`~pylorenzmie.theory.Particle.Particle` (or
:class:`~pylorenzmie.theory.Sphere.Sphere`), add the physical parameters as
dataclass fields, list those that can be fitted in ``properties``, and
implement ``ab(n_m, wavelength)``, which returns the Mie coefficients for a
medium of refractive index ``n_m`` at a vacuum wavelength ``wavelength``
(micrometers).  The calculators need nothing else.

A new estimator
---------------

Subclass :class:`~pylorenzmie.analysis.BaseEstimator.BaseEstimator`, implement
``estimate(hologram)``, which takes a
:class:`~pylorenzmie.analysis.Hologram.Hologram` and returns a
:class:`pandas.Series` of the particle's properties, and the ``properties``
getter.  Use it by assigning it to the ``estimator`` of a
:class:`~pylorenzmie.analysis.Feature.Feature`.  If it is random, take a
``seed``.

A new mask or localizer
-----------------------

A subclass of :class:`~pylorenzmie.analysis.Mask.Mask` overrides ``_select`` and
draws from ``self._generator()``, which honors the ``seed``.  A localizer is any
object whose ``localize(image)`` returns a table with the columns ``x_p``,
``y_p`` and ``bbox`` (``((x0, y0), width, height)``); give it to a
:class:`~pylorenzmie.analysis.Frame.Frame` as ``localizer``.
