Theory (``pylorenzmie.theory``)
================================

Calculators
-----------

:class:`~pylorenzmie.theory.LorenzMie.LorenzMie` is the reference
implementation in NumPy. The accelerated calculators below are subclasses
that replace the inner calculation. ``pylorenzmie.theory.LorenzMie`` is
:class:`~pylorenzmie.theory.numbaLorenzMie.numbaLorenzMie` if Numba works and
the NumPy class otherwise. The GPU calculators
(:class:`~pylorenzmie.theory.torchLorenzMie.TorchLorenzMie`,
:class:`~pylorenzmie.theory.cupyLorenzMie.cupyLorenzMie`) and
:class:`~pylorenzmie.theory.jaxLorenzMie.jaxLorenzMie` are never chosen
automatically. Select one by setting the environment variable
``PYLORENZMIE_BACKEND`` to ``numpy``, ``numba``, ``torch``, ``jax`` or
``cupy`` before importing pylorenzmie (a backend that cannot be loaded
raises ``ImportError``), or import the class from its module. See
:doc:`../quickstart` for timings.

Particles and instrument
------------------------

Instrument
----------

.. automodule:: pylorenzmie.theory.Instrument
   :members:

Particle
--------

.. automodule:: pylorenzmie.theory.Particle
   :members:

Sphere
------

.. automodule:: pylorenzmie.theory.Sphere
   :members:

Cluster
-------

.. automodule:: pylorenzmie.theory.Cluster
   :members:

Dimer
-----

.. automodule:: pylorenzmie.theory.Dimer
   :members:

LorenzMie
---------

.. automodule:: pylorenzmie.theory.LorenzMie
   :members:

Aberrated
---------

.. automodule:: pylorenzmie.theory.Aberrated
   :members:

cupyLorenzMie
-------------

.. automodule:: pylorenzmie.theory.cupyLorenzMie
   :members:

numbaLorenzMie
--------------

.. automodule:: pylorenzmie.theory.numbaLorenzMie
   :members:

torchLorenzMie
--------------

.. automodule:: pylorenzmie.theory.torchLorenzMie
   :members:

torchLorenzMieBatch
-------------------

.. automodule:: pylorenzmie.theory.torchLorenzMieBatch
   :members:

jaxLorenzMie
------------

.. automodule:: pylorenzmie.theory.jaxLorenzMie
   :members:
