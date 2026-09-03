'''Run a normalized hologram through the full analysis pipeline.

Detects features, merges overlapping bounding boxes into two-particle
groups (:func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`),
then estimates and optimizes every
:class:`~pylorenzmie.analysis.Feature`.  An isolated box is fit as a
single :class:`~pylorenzmie.theory.Sphere`; a merged pair is fit as a
:class:`~pylorenzmie.theory.Pair` with
:class:`~pylorenzmie.analysis.PairEstimator`.

Usage
-----
    python devel/pair_pipeline_example.py [image]

``image`` is a bare filename in ``docs/simulated pair holograms/`` or
``docs/tutorials/``, or a path to any 8-bit hologram normalized to 100
counts.  Default: ``close_pair.png`` (the mid-separation simulated
pair from ``devel/make_test_holograms.py``).
'''

import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pylorenzmie.analysis import Frame, Localizer
from pylorenzmie.theory import Pair
from pylorenzmie.utilities import example_hologram


SIMULATED = (Path(__file__).resolve().parent.parent
             / 'docs' / 'simulated pair holograms')


def _load(image: str) -> np.ndarray:
    '''Normalized hologram from a path, a ``docs/simulated pair
    holograms/`` name, or a ``docs/tutorials/`` name.'''
    for candidate in (Path(image), SIMULATED / image):
        if candidate.is_file():
            return (cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    .astype(float) / 100.)
    return example_hologram(image).data


# Tutorial microscope (docs/tutorials/*.png).
INSTRUMENT = dict(wavelength=0.447, magnification=0.048,
                  numerical_aperture=1.45, n_m=1.34)


def _fmt(properties: dict) -> str:
    '''One-line ``key=value`` view of a particle's properties.'''
    return '  '.join(f'{k}={float(v):.3f}' for k, v in properties.items())


def run_pipeline(name: str = 'close_pair.png',
                 show: bool = True,
                 nfringes: int | None = None,
                 diameter: int | None = None,
                 **instrument: float) -> Frame:
    '''Detect, estimate, and optimize every feature in an example image.

    Parameters
    ----------
    name : str, optional
        A bare filename in ``docs/simulated pair holograms/`` or
        ``docs/tutorials/``, or a path to any hologram normalized to
        100 counts.  Default: ``'close_pair.png'``.
    show : bool, optional
        Display the detection boxes and per-feature data/fit/residual
        panels.  Default: ``True``.
    nfringes : int, optional
        Fringes enclosed in each detection box.  Smaller boxes are less
        likely to merge dense fields into 3+ clusters that
        :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`
        discards.  Default: the :class:`Localizer` default (20).
    diameter : int, optional
        ``trackpy.locate`` feature diameter, in pixels.  Default: the
        :class:`Localizer` default (31).
    **instrument
        Instrument overrides (``wavelength`` [um], ``magnification``
        [um/pixel], ``n_m``, ...).  Default: the tutorial microscope.

    Returns
    -------
    frame : Frame
        The analyzed frame, with ``features`` and ``results`` populated.
    '''
    settings = {**INSTRUMENT, **instrument}

    localizer = Localizer()
    if nfringes is not None:
        localizer.nfringes = nfringes
    if diameter is not None:
        localizer.diameter = diameter

    frame = Frame(localizer=localizer)
    for key, value in settings.items():
        setattr(frame.instrument, key, value)
    frame.data = _load(name)

    print(f'image      : {name}  {frame.shape}')
    print(f'instrument : {settings}')
    print(f'localizer  : nfringes={localizer.nfringes} '
          f'diameter={localizer.diameter}')

    n = frame.detect()
    print(f'\ndetected {n} feature(s):')
    for i, (feature, bbox) in enumerate(zip(frame.features, frame.bboxes)):
        kind = 'pair  ' if isinstance(feature.particle, Pair) else 'single'
        (x0, y0), w, h = bbox
        print(f'  [{i}] {kind}  {type(feature.estimator).__name__:<13} '
              f'shape={feature.shape}  '
              f'bbox=(({int(x0)}, {int(y0)}), {int(w)}, {int(h)})')
    if n == 0:
        return frame

    start = perf_counter()
    frame.estimate()
    print(f'\nestimated in {perf_counter() - start:.2f} s:')
    for i, feature in enumerate(frame.features):
        print(f'  [{i}]  ' + _fmt(feature.particle.properties))

    start = perf_counter()
    results = frame.optimize()
    print(f'\noptimized in {perf_counter() - start:.2f} s:')
    for i, feature in enumerate(frame.features):
        r = feature.optimizer.result
        print(f'  [{i}] success={bool(r.success)}  '
              f'npix={r.npix:.0f}  redchi={r.redchi:.2f}')
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print()
        print(results.T)

    if show:
        _show(frame, name)
    return frame


def _show(frame: Frame, name: str) -> None:
    '''Full frame with detection boxes; data/fit/residual per feature.'''
    box_style = dict(fill=False, linewidth=2, edgecolor='red')
    fig, ax = plt.subplots(num=f'{name}: detections')
    ax.imshow(frame.data, cmap='gray')
    for (x0, y0), w, h in frame.bboxes:
        ax.add_patch(Rectangle((x0, y0), w, h, **box_style))
    ax.set_title(f'{name}: {len(frame.features)} feature(s)')

    for i, feature in enumerate(frame.features):
        # Pin the model to the fitted optimum before predicting so the
        # residual reflects the reported fit, not the last DE/LM probe.
        result = feature.optimizer.result
        feature.model.properties = {v: result[v]
                                    for v in feature.optimizer.variables}
        data = feature.data
        fit = feature.predicted()

        vmin, vmax = 0.9 * data.min(), 1.1 * data.max()
        img_style = dict(vmin=vmin, vmax=vmax, cmap='gray')
        fig, axes = plt.subplots(ncols=3, figsize=(10, 4),
                                 constrained_layout=True,
                                 num=f'{name}: feature {i}')
        panels = [(data, 'data'),
                  (fit, f'fit  (redchi = {result.redchi:.2f})'),
                  (fit - data + 1., 'residual + 1')]
        for ax, (img, label) in zip(axes, panels):
            ax.imshow(img, **img_style)
            ax.set_title(label)
            ax.axis('off')

    plt.show()


if __name__ == '__main__':  # pragma: no cover
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else 'close_pair.png')
