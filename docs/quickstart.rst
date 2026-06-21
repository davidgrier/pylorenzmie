Quickstart
==========

Installation
------------

Install from PyPI::

    pip install pylorenzmie

For GPU-accelerated hologram computation (requires CUDA 12)::

    pip install pylorenzmie[gpu]

The interactive GUI tool (:doc:`lmtool`) is included in the standard
install and can be launched immediately::

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
   from pylorenzmie.analysis import Feature
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
   feature = Feature(data=data,
                     coordinates=model.meshgrid(data.shape),
                     model=model)
   result = feature.optimize()
   print(result)

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
