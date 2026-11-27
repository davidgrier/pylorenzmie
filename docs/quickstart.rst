Quickstart
==========

Installation
------------

Install from PyPI::

    pip install pylorenzmie

Optional accelerated calculators are installed as extras::

    pip install pylorenzmie[numba]   # fast CPU calculator, recommended
    pip install pylorenzmie[torch]   # GPU: CUDA, ROCm, Intel, or Apple Metal
    pip install pylorenzmie[gpu]     # GPU: CuPy (requires CUDA 12)
    pip install pylorenzmie[jax]     # JAX, with automatic differentiation

Extras can be combined, as in ``pip install pylorenzmie[numba,torch]``.
See `Speed: the Numba backend and threads`_ below for what each one buys.

The interactive GUI tool (:doc:`lmtool`) needs a Qt binding, which is not
installed by default. Install one with ``pip install pylorenzmie[pyqt6]``
(PyQt6 and PyTables, which saves results as HDF5; ``[pyqt5]`` and
``[pyside6]`` use PyQt5 and PySide6 instead), and launch the tool with::

    lmtool

Computing a hologram
--------------------

.. code-block:: python

   from pylorenzmie.theory import LorenzMie, Sphere, Instrument
   import matplotlib.pyplot as plt

   # Describe the microscope
   instrument = Instrument()
   instrument.wavelength = 0.447      # vacuum wavelength [um]
   instrument.magnification = 0.048   # image scale [um/pixel]
   instrument.n_m = 1.340             # medium refractive index

   # Describe the particle
   particle = Sphere()
   particle.a_p = 0.75    # radius [um]
   particle.n_p = 1.45    # refractive index
   particle.r_p = [100, 100, 200]   # position [pixels]

   # Compute hologram on a 201x201 grid
   model = LorenzMie(instrument=instrument)
   model.coordinates = model.meshgrid((201, 201))
   model.particle = particle
   hologram = model.hologram().reshape(201, 201)

   plt.imshow(hologram, cmap='gray')
   plt.show()

Fitting a hologram
------------------

.. code-block:: python

   from pylorenzmie.analysis import Feature, Hologram
   from pylorenzmie.theory import LorenzMie
   from pylorenzmie.utilities import imread

   # Load and normalize a hologram (divide by background level)
   data = imread('crop.png').astype(float)
   data /= data.mean()

   # Set up the model
   model = LorenzMie()
   model.instrument.wavelength = 0.447
   model.instrument.magnification = 0.048
   model.instrument.n_m = 1.340
   model.particle.r_p = [data.shape[1]//2, data.shape[0]//2, 330.]
   model.particle.a_p = 1.0
   model.particle.n_p = 1.45

   # Optimize
   feature = Feature(Hologram(data), model=model)
   result = feature.optimize()
   print(result)    # fitted values, uncertainties and goodness of fit

Choosing how to fit
~~~~~~~~~~~~~~~~~~~

The optimizer of a feature, ``feature.optimizer``, does least squares with the
Levenberg-Marquardt algorithm.  Three settings change that; set them before
calling ``feature.optimize()``.

A robust fit limits the influence of pixels that the model does not explain,
such as dust or the fringes of a neighbor:

.. code-block:: python

   feature.optimizer.robust = True

Bounds keep the parameters within limits, with positions in pixels and the
radius in micrometers.  The starting values must lie within them:

.. code-block:: python

   feature.optimizer.bounds = {'a_p': (0.5, 2.), 'n_p': (1.38, 1.7)}

The method ``'broyden'`` updates the Jacobian instead of computing it at
every step.  It is a little more reliable than the default in double
precision, at about the same cost, and takes neither a robust loss nor
bounds:

.. code-block:: python

   feature.optimizer.settings['method'] = 'broyden'

See :class:`~pylorenzmie.analysis.Optimizer.Optimizer` for all of the
settings.

Analyzing a full frame
----------------------

.. code-block:: python

   from pylorenzmie.analysis import Frame
   from pylorenzmie.utilities import imread

   frame = Frame()
   data = imread('image.png').astype(float)
   data /= data.mean()
   results = frame.analyze(data)
   print(results)   # pandas.DataFrame with one row per particle

Each particle that is detected is fitted on its own, even if its pattern
overlaps another's.  Two particles too close to be told apart can be fitted
as a pair instead: ``Frame(split_redchi=2.0)`` refits a particle that one
sphere explains poorly, and ``Frame(pairs=True)`` fits two detections with
overlapping boxes together.  Check what a pair fit returns, since a lower
reduced chi-squared does not make it physical.

.. _finding-particles-with-catch:

Finding particles with CATCH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CATCH <https://github.com/laltman2/CATCH>`_ uses neural networks to find
holograms and to estimate each particle's height, radius and refractive index.
It is optional, needs PyTorch, and is installed from its repository (the
package ``CATCH`` of PyPI is something else)::

    pip install git+https://github.com/laltman2/CATCH.git

A :class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer` is the localizer
of a frame:

.. code-block:: python

   from pylorenzmie.analysis import CATCHAnalyzer, Frame
   from pylorenzmie.theory import Instrument

   instrument = Instrument()
   instrument.wavelength = 0.447
   instrument.magnification = 0.048
   instrument.n_m = 1.340

   catch = CATCHAnalyzer(instrument)
   frame = Frame(localizer=catch)     # shares the instrument of catch
   results = frame.analyze(data)      # data: a normalized hologram

CATCH runs once for the whole frame.  Each fit starts from its estimates, so
there is no search and there are no ranges of parameters to choose.  The fits
are robust unless ``Frame(robust=False)`` asks for least squares, and the
results have two more columns: the ``confidence`` of each detection and its
``quality``, which is empty or says why not to trust the particle.  A particle
for which CATCH has no estimate keeps its row, with empty values and
``success`` False.  Pass the saturated pixels of the image, if there are any,
as ``frame.analyze(data, saturated=saturated)``.  For many frames,
``CATCHAnalyzer(instrument, compile=True)`` found particles about twice as
fast after the first frame on an Apple GPU, though the fits take most of the
time of a frame.

The networks were trained for a range of pixel sizes, measured in wavelengths
in the medium; ``found.attrs['warning']`` of ``catch.localize(data)`` says if
yours is outside it.  A frame with ``split_redchi`` is fitted by least
squares, because a robust fit's reduced chi-squared cannot be compared with
that threshold.

.. note::

   The global search that starts each fit runs in a single process by
   default, which is fastest with the Numba calculator. For the NumPy
   calculator and large crops it can use a pool of worker processes, with
   ``feature.estimator.settings['workers'] = -1`` for each feature (after
   ``frame.detect()``). On macOS and Windows, Python starts each worker by
   importing your script again, so a script that uses the pool must be
   protected by ``if __name__ == '__main__':``.

Speed: the Numba backend and threads
------------------------------------

If Numba is installed (``pip install pylorenzmie[numba]``),
:class:`~pylorenzmie.theory.numbaLorenzMie` evaluates the Lorenz-Mie
field one point at a time on all CPU threads: about 4 times faster than
the NumPy calculator on one thread, and 10 to 20 times faster on a
ten-core machine.  It is what ``from pylorenzmie.theory import
LorenzMie`` returns when Numba works; otherwise you get the NumPy
calculator, which needs no extra packages.  Numba also compiles the
recurrences of the Mie coefficients, which a global search evaluates for
every candidate.

The GPU calculators are never chosen automatically, because they pay off
for large images, not for the small masked fits that are typical.
:class:`~pylorenzmie.theory.torchLorenzMie.TorchLorenzMie` runs on CUDA GPUs
(Triton) and on Apple GPUs (Metal, through PyTorch's MPS back end) in single
precision; on an Apple M1 Pro it fits a particle in an image of 25,000 or more
pixels two to three times faster than Numba does, and takes about as long
for a masked fit of a few thousand pixels.  CuPy needs a CUDA GPU and JAX is slower than Numba on a CPU.
To use one, set the environment variable ``PYLORENZMIE_BACKEND`` to
``numpy``, ``numba``, ``torch``, ``jax`` or ``cupy`` (an unavailable choice
raises ``ImportError``), or import the class from its module, for example
``from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie``.

The threads are on by default because the points of a hologram are
independent.  Turn them off when your own code is already parallel,
because the two levels of parallelism then compete for the same cores:

.. code-block:: python

   from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie

   numbaLorenzMie.parallel = False      # all models
   model = numbaLorenzMie()
   model.parallel = False               # this model only

   # or share the cores among your worker processes
   import numba
   numba.set_num_threads(2)

The cases to look for are worker pools (``multiprocessing``, ``joblib``,
``concurrent.futures``) with a worker per core, and calls from inside
another Numba parallel function, which Numba does not support.  Set the
``NUMBA_NUM_THREADS`` environment variable before importing Numba to
limit the number of threads for the whole program.  See the
:class:`~pylorenzmie.theory.numbaLorenzMie` documentation for details.
