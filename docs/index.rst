pylorenzmie
===========

**pylorenzmie** provides Python tools for tracking and characterizing
colloidal particles with in-line holographic video microscopy (HVM).
The hologram of a colloidal particle encodes its size, composition,
and three-dimensional position; this package extracts that information
by fitting recorded holograms to a generative model based on
Lorenz-Mie theory.

.. code-block:: python

   from pylorenzmie.theory import LorenzMie, Sphere, Instrument

   instrument = Instrument(wavelength=0.447, magnification=0.048, n_m=1.340)
   particle = Sphere(a_p=0.75, n_p=1.45)
   particle.r_p = [100, 100, 200]

   model = LorenzMie(instrument=instrument)
   model.coordinates = model.meshgrid((201, 201))
   model.particle = particle
   hologram = model.hologram()

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   lmtool
   api/lib
   api/theory
   api/analysis
   api/utilities
   api/lmtool

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
