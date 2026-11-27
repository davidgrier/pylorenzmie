LMTool
======

**lmtool** is an interactive graphical application for loading holographic
microscopy images, positioning a region of interest around a particle, and
fitting a Lorenz-Mie scattering model to the hologram.

Launching
---------

**lmtool** needs a Qt binding, which pylorenzmie does not install by default:
``pip install pylorenzmie[pyqt6]`` (or ``[gui]``, which is the same) adds PyQt6
and PyTables; ``[pyqt5]`` and ``[pyside6]`` add PyQt5 and PySide6 instead, and
all three are tested.  Without PyTables, results are saved as JSON.  The
environment variable ``QT_API`` (``pyqt6``, ``pyqt5`` or ``pyside6``) chooses
among several bindings that are installed.  Then launch from the command
line::

    lmtool [image.png]

or equivalently::

    python -m pylorenzmie.lmtool [image.png]

Passing an image file loads it immediately on startup.  Without an
argument lmtool opens with a built-in example hologram.

Normalization
^^^^^^^^^^^^^

The Lorenz-Mie model predicts a hologram whose background intensity is 1,
so recorded images must be normalized before fitting.  lmtool normalizes
by dividing each pixel by the median pixel value of the image.  This works
well when most of the frame is background.

When a prerecorded background image is available, use it for higher-quality
normalization::

    lmtool image.png --background background.png

Images are read at their own bit depth, so a background image must have the
bit depth of the hologram.  A scalar background value is also accepted, in
the gray levels of the image (useful with a known camera offset)::

    lmtool image.png --background 100

A camera records a constant dark count (black level) even without light.  It
must be subtracted from the hologram and the background before the division,
or the fringes lose contrast and fits are biased.  Give it on the command
line, in gray levels::

    lmtool image.png --background background.png --dark 22

or set it at any time with **File → Set Dark Count...**.  The status bar shows
it with the background.

A background image can also be loaded after startup via
**File → Open Background...**.  The status bar shows the background in use.  A
background that cannot be read is reported there and in the log, and the
previous one stays.  A background image must have the shape of the hologram:
otherwise the hologram is divided by its median, and the status bar says so.

Estimates and fits leave out the pixels at which the camera saturated, in the
hologram or in the background image.  These are the pixels at the largest
value that the file can hold: 255 for 8 bits.  A camera of 10 or 12 bits that
writes 16-bit files saturates below that, so give its level::

    lmtool image.tif --saturation 4095

Model
^^^^^

By default lmtool models the aberrations of the microscope as phase factors
(:func:`~pylorenzmie.theory.Aberrated.Aberrated`) and computes the
scattering with the fastest calculator that is available.  Two options
change that::

    lmtool image.png --distorted
    lmtool image.png --backend numpy

``-d`` or ``--distorted`` models the aberrations as geometric distortions
(:func:`~pylorenzmie.theory.Distorted.Distorted`), which is exact within the
operator formalism and costs a little more.  ``-B`` or ``--backend`` chooses
the calculator: ``numpy``, ``numba``, ``torch``, ``jax`` or ``cupy``.  The
choice overrides the environment variable ``PYLORENZMIE_BACKEND``.  The
distorted model works with ``numpy`` and ``numba`` only.  An unavailable
choice is reported at startup.  (The short option ``-b`` is the background.)

Interface
---------

The window is divided into a tabbed display area at the top and a control
panel at the bottom.

Image tab
^^^^^^^^^

.. figure:: lmtool/LMTool_ImageWidget.png
   :alt: Image tab showing a hologram with a circular ROI

The **Image** tab shows the full normalized hologram.  A circular
region-of-interest (ROI) marks the area around the particle to be fitted.

When an image is loaded, lmtool looks for particles in the background and
puts the ROI on the first one, sized to enclose its fringes; the window stays
usable while it looks.  **Particles → Next Particle** (Ctrl+N) moves to the next
one, and **Particles → Find Particles** (Ctrl+L) looks again.  A noisy image can
have hundreds of candidates, so lmtool keeps the ten strongest, strongest first,
and the status bar says how many it found; many candidates usually means that
the image is noisy or badly normalized.  If there are none, or only one, the
status bar says that as well, and the ROI stays where it is.  The ROI can also
be set by hand:

* **Drag the ROI centre** to reposition it over the particle.
* **Drag the ROI edge** to resize it so the rings are enclosed.

The ROI centre is used as the initial estimate of the particle's lateral
position (:math:`x_p, y_p`).

If `CATCH <https://github.com/laltman2/CATCH>`_ is installed in the same
environment, **Particles → Use CATCH** is enabled and checked.  **Find
Particles** then uses the detector of CATCH, and **Optimization → Estimate**
uses its network, instead of the circle transform and the search of
differential evolution; uncheck it to use those.  The networks load the first
time that they are used, which takes a few seconds, and they use the
wavelength, the magnification and the index of the medium that the model has.
They were trained for pixels 1 to 2.5 times the size of those of a reference
microscope (0.048 um at 0.447 um in water), measured in wavelengths in the
medium, and lmtool warns when the pixels of the model are more than 5%
outside that range.  Without CATCH the entry is disabled and the built-in
methods are used.

The region of a particle that CATCH finds reaches the 10th bright ring of its
hologram, or stops where the rings fade into the noise, though not short of
the 5th (see
:class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer`).  Saturated pixels
are left out.  Scripts get the same detector and estimates
from :class:`~pylorenzmie.analysis.CATCHAnalyzer.CATCHAnalyzer` (see
:ref:`finding-particles-with-catch`).

CATCH needs PyTorch.  Install it from its repository:

.. code-block:: sh

   pip install git+https://github.com/laltman2/CATCH.git

Do not use ``pip install CATCH``, which installs an unrelated package from
PyPI.

Particles tab
^^^^^^^^^^^^^

The **Particles** tab lists the particles that lmtool knows about: the ones
that it found, and the ones that you added.  Each row shows where the particle
is, whether it was found or added, what has been done with it (``new``,
``estimated`` or ``fitted``), the fitted height, radius and index, and the
reduced chi-squared of the fit.

* Click a row to move the ROI to the particle and to restore its parameters.
* Uncheck a row to leave a particle out: Next Particle skips it and Save Result
  does not write it.  Use this for false detections.
* **Particles → Add Particle** (Ctrl+Shift+A) adds the ROI as a particle,
  for one that the search missed; **Remove Particle** (Ctrl+Backspace) removes
  the selected one.
* An estimate or a fit is recorded for the particle that the ROI is on, and
  a fit with the ROI on no particle adds one.

lmtool fits one particle at a time, and the table collects the results.  It does
not process a whole image or a stream of images by itself.

Profile tab
^^^^^^^^^^^

.. figure:: lmtool/LMTool_ProfileWidget.png
   :alt: Profile tab showing radial intensity and model prediction

The **Profile** tab shows the azimuthally averaged radial intensity profile
of the hologram (black curve) together with the current model prediction
(red curve).  The orange band is ±1σ of the azimuthal variation.

Use this view to judge initial parameter estimates before running an
optimization.  A good starting point has the model rings roughly aligned
with the data rings in both position and amplitude.

Fit tab
^^^^^^^

.. figure:: lmtool/LMTool_FitWidget.png
   :alt: Fit tab showing data, model, and residuals side by side

The **Fit** tab shows three panels sharing the same spatial axes:

* **Left** — raw cropped hologram from the ROI.
* **Centre** — current model hologram, on the color scale of the data (the
  model is not clipped).
* **Right** — normalized residuals ``(data − model) / noise``.

The residuals are divided by the **noise** control, which lmtool sets to an
estimate of the pixel noise when it loads an image, and again from the region of
interest with **Optimization → Estimate Noise**.  The estimate comes from the
pixel-to-pixel variation, leaving out pixels that are not finite (where the
background is no brighter than the dark count), and does not see structure
that is correlated over several pixels, such as a fixed pattern in the
background.  A reduced
chi-squared in the status bar well above 1 after the fit therefore means that
something other than noise is left in the residuals.

The model and residuals update live as parameters change and again after
each optimization.  Well-converged fits show structureless residuals with
magnitude ≲ 3.

Control panel
^^^^^^^^^^^^^

The bottom section of the window holds two groups of controls.

**Parameter controls** (left) — sliders for every particle and instrument
parameter.  Each slider has a lock button; locked parameters are held fixed
during optimization and are excluded from uncertainty estimates.  A locked
slider cannot be moved, except those of the wavelength, the magnification and
the index of the medium, which describe the microscope: set them with their
locks closed, and release one only if a fit is to adjust it.

**Optimizer settings** (right) — controls the behaviour of the least-squares
solver: pixel fraction, method, loss function, scale of the loss
(``f_scale``), convergence tolerances and the step of the finite differences.

Changing the optics or the background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Particles are found and fitted with the wavelength, the magnification and the
index of the medium that the controls show, in an image normalized by the
background and the dark count.  When you change one of these, lmtool starts
over, as with a new image: it empties the Particles tab, including the
particles that you added, looks for particles again, and selects the first.
After a change of the optics it waits until the value has been left alone for
half a second.  If a checked particle has a fit that has not been saved,
lmtool first asks whether to discard it; if you keep it, the previous value of
the optics returns, or the previous background stays.  A fit that adjusts an
unlocked parameter of the optics does not start over.  Opening an image also
starts over, after the same question about unsaved fits.

Robust fits
^^^^^^^^^^^

**Optimization → Set Robust Mode** sets up a fit that limits the influence of
pixels that the model does not explain, such as dust or the fringes of a
neighbor.  It selects the trust-region reflective method with a Cauchy loss,
and ``f_scale`` ``auto``, which matches the scale of the loss to the data:
each fit is made by least squares first, and the loss then discounts the
pixels that differ from the model by much more than those residuals typically
do (see :attr:`~pylorenzmie.analysis.Optimizer.Optimizer.settings`).  lmtool
starts with these settings.

Any of them may be changed afterwards, and the menu entry restores them.  For
a least-squares fit, choose the method Levenberg-Marquardt, or the loss
``linear``.  A numerical ``f_scale`` discounts pixels on the scale of the
**noise** control, which for many real holograms, with a scale of 1, discounts
too many of them.  The reduced chi-squared in the status bar of a robust fit
is computed from the robust cost, and is not comparable with that of least
squares.

Workflow
--------

1. **Load a hologram** — **File → Open ...** or pass a filename on the
   command line.
2. **Position the ROI** — lmtool puts it on the first particle it finds;
   choose another with **Particles → Next Particle**, or drag the circle in
   the Image tab to centre it on the particle and resize it to enclose the
   fringe pattern.
3. **Adjust parameters** — move the sliders until the Profile tab model curve
   (red) matches the data in fringe spacing and amplitude.  The axial
   position :math:`z_p` has the strongest effect on fringe spacing;
   :math:`a_p` and :math:`n_p` affect the contrast.
4. **Optimize** — press **Ctrl+P** or choose **Optimization → Optimize**.
   The Fit tab updates to show the fitted model and residuals.  An estimate
   or a fit runs in the background, on a copy of the parameters, and **Cancel**
   (Esc) stops it and leaves the parameters as they were.
5. **Repeat** — choose the next particle and repeat steps 2 to 4; the Particles
   tab keeps what you did.
6. **Save results** — **Optimization → Save Result** writes the fits of the
   particles that are checked, one row each, to ``~/data/lmtool/``; **Save
   Result As...** lets you choose the path and the format: HDF5 (or JSON if
   PyTables is not installed), JSON or CSV.  The instrument, the fixed
   parameters, the optimizer settings and the name of the image go with them.
