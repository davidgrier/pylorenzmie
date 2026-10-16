Architecture
============

pylorenzmie computes holograms of colloidal particles with Lorenz-Mie theory
and fits recorded holograms to them.  The pipeline is

**coordinates** → **compute a hologram** → **detect features** →
**estimate parameters** → **optimize the fit**

and the packages follow it.

.. code-block:: text

   utilities     Normalizer, VMedian      raw image -> normalized hologram
        |
   analysis      Frame -> Localizer       find the particles, crop a box each
                 Feature                  one particle: Hologram + model
                   |- Estimator           starting values
                   '- Optimizer           least-squares fit
        |
   theory        LorenzMie (+ backends)   field and hologram of the particles
                 Particle, Sphere, ...    what scatters
                 Instrument               how it is observed
        |
   lib           LMObject, meshgrid, ...  shared base classes and tools

``lmtool`` is a graphical application on top of all of them.

The base class
--------------

:class:`~pylorenzmie.lib.LMObject.LMObject` is the base of every major class.
It defines the **properties protocol**: the ``properties`` dictionary of an
object holds its adjustable numerical parameters.  It serves for
serialization (``to_json``, ``to_pandas``) and for fitting: an
:class:`~pylorenzmie.analysis.Optimizer.Optimizer` fits the entries of the
model's ``properties`` that are not listed in its ``fixed`` list, and sets them
through the same dictionary.  A parameter that is not in ``properties`` can
never be fitted, which is how, for example, the members of a
:class:`~pylorenzmie.theory.Cluster.Cluster` are kept out of a fit.

Theory
------

A calculator, :class:`~pylorenzmie.theory.LorenzMie.LorenzMie` or one of its
subclasses, holds the pixel ``coordinates``, a ``particle`` (a particle, or a
list or cluster of them) and an ``instrument``.  ``field()`` returns the
scattered field and ``hologram()`` the normalized intensity.  The particle
supplies the Mie coefficients through ``ab()``; the calculator turns them
into a field by summing the partial waves at every pixel, which is the
expensive step and the part that the accelerated calculators replace.

The calculators are interchangeable.  Their class attribute ``method``
('numpy', 'numba numpy', 'torch numpy', ...) names them, and an optimizer
checks that its own ``method`` is part of the model's.
:func:`~pylorenzmie.theory.Aberrated.Aberrated` wraps any calculator with the
aberrations of the microscope as phase factors, and
:func:`~pylorenzmie.theory.Distorted.Distorted` applies them as the exact
distortions of the operator formalism (NumPy and Numba calculators).  Which calculator ``pylorenzmie.theory``
exports as ``LorenzMie`` is decided when the package is imported; see the
theory page of the API reference.

Analysis
--------

A :class:`~pylorenzmie.analysis.Hologram.Hologram` is a normalized image
together with the position of its corner in the full frame, so a crop knows
where it came from.  A :class:`~pylorenzmie.analysis.Feature.Feature` is a
hologram of one particle with a model, an estimator and an optimizer, and
a :class:`~pylorenzmie.analysis.Frame.Frame` is the whole camera image with a
localizer that creates a feature for every particle it finds.  The steps of a
frame (``detect``, ``estimate``, ``optimize``) can be run one at a time, which
is how an estimator or an optimizer is adjusted for a particular sample.

Four estimators produce starting values:
:class:`~pylorenzmie.analysis.Estimator.Estimator` (from the fringes),
:class:`~pylorenzmie.analysis.DEEstimator.DEEstimator` (global search),
:class:`~pylorenzmie.analysis.RadialEstimator.RadialEstimator` (global search
on the radial profile) and
:class:`~pylorenzmie.analysis.MLPEstimator.MLPEstimator` (a trained network).
:class:`~pylorenzmie.analysis.Mask.Mask` selects the pixels that a fit uses.

Layout of the repository
------------------------

======================  ====================================================
``lib/``                ``LMObject``, ``meshgrid``, azimuthal statistics, the
                        circle transform, type aliases
``theory/``             particles, the instrument, the calculators
``analysis/``           holograms, features, frames, estimators, the optimizer
``utilities/``          normalization, running median, example holograms
``lmtool/``             the graphical application
``tests/``              unit tests (``tests/slow`` for the slow ones)
``docs/``               these pages, the notebooks and the tutorial images
======================  ====================================================

The packages are installed as ``pylorenzmie.lib``, ``pylorenzmie.theory`` and
so on.
