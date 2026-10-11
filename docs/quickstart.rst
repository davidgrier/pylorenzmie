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
installed by default. Install one with ``pip install pylorenzmie[gui]``
(PyQt6; PyQt5, PySide2 and PySide6 also work), and launch the tool with::

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

   import cv2
   from pylorenzmie.analysis import Feature, Hologram
   from pylorenzmie.theory import LorenzMie

   # Load and normalize a hologram (divide by background level)
   data = cv2.imread('crop.png', cv2.IMREAD_GRAYSCALE).astype(float)
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

Analyzing a full frame
----------------------

.. code-block:: python

   import cv2
   from pylorenzmie.analysis import Frame

   frame = Frame()
   data = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE).astype(float)
   data /= data.mean()
   results = frame.analyze(data)
   print(results)   # pandas.DataFrame with one row per particle

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
calculator, which needs no extra packages.

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
