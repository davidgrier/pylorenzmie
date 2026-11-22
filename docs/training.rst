Training the MLP estimator
==========================

:class:`~pylorenzmie.analysis.MLPEstimator.MLPEstimator` guesses the
height, radius and refractive index of a sphere from the azimuthal average
of its hologram, with a small neural network, in under a millisecond.  The
network that ships with the package was trained for one instrument: a
447 nm laser, 0.048 |um| per pixel, and a medium of water
(:math:`n_m = 1.340`).  If your instrument is different, train a network
for it.  The
:class:`~pylorenzmie.analysis.Optimizer.Optimizer` refines the estimate
either way, so a network that was trained for another instrument still
starts the fit, only from a worse place.

.. |um| unicode:: U+03BC U+006D

What the network learns
-----------------------

The network maps the first 100 pixels of an azimuthal profile to
:math:`(z_p, a_p, n_p)`.  It learns from synthetic profiles, which
``tools/train_mlp_estimator.py`` computes from the same Lorenz-Mie model
that the fits use:

* The parameters are drawn at random, with :math:`z_p` from 20 to 500
  pixels and :math:`a_p` from 0.2 to 4 |um|, both uniform in their
  logarithms, and :math:`n_p` from 1.3 to 2.5, uniformly.
* A profile is the model evaluated along a radius, which equals the
  azimuthal average for a sphere.
* Each profile is cut off at a random radius of at least 30 pixels and
  padded with zeros, so that the network learns to cope with small crops.
  :class:`~pylorenzmie.analysis.MLPEstimator.MLPEstimator` pads in the same
  way.
* Draws for which the Mie series overflows are dropped.

Requirements
------------

Training needs a source checkout, because the script writes into the
package, and scikit-learn, which does the fitting.  Using the trained
network needs neither.

.. code-block:: bash

   git clone https://github.com/davidgrier/pylorenzmie
   cd pylorenzmie
   pip install -e ".[train,numba]"

Numba is optional: the model uses it if it is installed, and the
200,000 model evaluations are the slow part of training.

Training
--------

1. Set the instrument.  Edit ``INSTRUMENT`` near the top of
   ``tools/train_mlp_estimator.py``:

   .. code-block:: python

      INSTRUMENT = dict(wavelength=0.447,      # um
                        magnification=0.048,   # um/pixel
                        n_m=1.340)             # medium

   The other settings in that block are ``BOUNDS`` (the ranges of the
   parameters), ``N_TRAIN`` (the number of profiles, 200,000),
   ``N_FEATURES`` (the length of a profile, 100), ``HIDDEN`` (the sizes of
   the layers, ``(256, 128, 64)``) and ``SEED``.  Keep ``z_p`` in pixels,
   as the estimators do.  A network trained with another ``N_FEATURES``
   must be used with ``MLPEstimator(n_features=...)`` to match.

2. Run the script from the top of the checkout:

   .. code-block:: bash

      python tools/train_mlp_estimator.py

   It generates the profiles, reports its progress, trains the network,
   and prints the typical error on 10% of the profiles that it held back.
   Then it reports how far the NumPy network that it saves is from the
   scikit-learn network that it trained (they should agree to rounding),
   and writes ``analysis/mlp_estimator.npz``.

   The profiles are saved in ``tools/mlp_training_data.npz``, which git
   ignores, so that training again (with another ``HIDDEN``, say) does not
   compute them again.  Add ``--regen`` to compute them anew, which you
   must do after changing ``INSTRUMENT``, ``BOUNDS`` or ``N_FEATURES``.

3. Check the result with the tests and with a hologram that you know:

   .. code-block:: bash

      python -m pytest tests/test_mlpestimator.py
      python -m pylorenzmie.analysis.MLPEstimator

   The second command estimates the example hologram that ships with the
   package.

By default the script overwrites ``analysis/mlp_estimator.npz``, which is
the network that ships.  To keep it, name another file with
``--output``:

.. code-block:: bash

   python tools/train_mlp_estimator.py --output my_network.npz

or use git to get the shipped network back
(``git checkout analysis/mlp_estimator.npz``).

Using the trained network
-------------------------

:class:`~pylorenzmie.analysis.MLPEstimator.MLPEstimator` reads the bundled
network by default.  To use another, pass its path:

.. code-block:: python

   from pylorenzmie.analysis.MLPEstimator import MLPEstimator
   from pylorenzmie.theory import LorenzMie

   model = LorenzMie()
   model.instrument.wavelength = 0.532
   model.instrument.magnification = 0.1
   estimator = MLPEstimator(model=model, weights='my_network.npz')

The instrument of the model should be the instrument that the network was
trained for.  The estimator reads the instrument of the model but does not
check it against the network.

The file is a NumPy archive of the standardization of the input and of the
output and of the weights and biases of each layer, with no pickled
objects.  It loads with any version of NumPy, and
:class:`~pylorenzmie.analysis.MLPEstimator.MLPNetwork` evaluates it with
no other library.  Networks that earlier versions saved as ``.joblib``
files cannot be read: train again.

Training a localizer
--------------------

pylorenzmie does not train a localizer.  Its
:class:`~pylorenzmie.analysis.Localizer.Localizer` is an algorithm
(the circle transform and trackpy) with nothing to learn, and a
neural-network localizer is not part of the package.  `CATCH <https://github.com/davidgrier/CATCH>`_, which
is built on pylorenzmie, has a trained localizer, and its documentation
describes how to train one.
