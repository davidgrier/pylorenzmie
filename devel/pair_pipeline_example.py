'''Run a normalized hologram through the full analysis pipeline.

Detects features, merges overlapping bounding boxes into two-particle
groups (:func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`),
then estimates and optimizes every
:class:`~pylorenzmie.analysis.Feature`.  An isolated box is fit as a
single :class:`~pylorenzmie.theory.Sphere`; a merged pair is fit as a
:class:`~pylorenzmie.theory.Pair` with
:class:`~pylorenzmie.analysis.PairEstimator`.  A single fit whose
redchi exceeds ``Frame.split_redchi`` is refit as a pair, and kept
only if that lowers its redchi.

Usage
-----
    python devel/make_test_holograms.py   # generates the example
                                           # images first; not
                                           # committed to git
    python devel/pair_pipeline_example.py [image]

``image`` is a bare filename in
``docs/tutorials/simulated pair holograms/`` (or its ``random
particles/`` / ``random particles (confined)/`` subfolders), or
``docs/tutorials/``, or a path to any 8-bit hologram normalized to 100
counts.  Default: ``random_particles.png`` (a random, un-curated
scatter of particles -- some isolated, some crowded into discarded 3+
clusters).
'''

import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pylorenzmie.analysis import Feature, Frame, Localizer
from pylorenzmie.theory import Pair
from pylorenzmie.utilities import example_hologram


DOCS = Path(__file__).resolve().parent.parent / 'docs'
SIMULATED = DOCS / 'tutorials' / 'simulated pair holograms'
RANDOM_PARTICLES = SIMULATED / 'random particles'
RANDOM_PARTICLES_CONFINED = SIMULATED / 'random particles (confined)'


def _load(image: str) -> np.ndarray:
    '''Normalized hologram from a path, a ``docs/simulated pair
    holograms/`` (or its ``random particles/`` / ``random particles
    (confined)/`` subfolders) name, or a ``docs/tutorials/`` name.'''
    candidates = (Path(image), SIMULATED / image, RANDOM_PARTICLES / image,
                 RANDOM_PARTICLES_CONFINED / image)
    for candidate in candidates:
        if candidate.is_file():
            return (cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    .astype(float) / 100.)
    return example_hologram(image).data


# Tutorial microscope (docs/tutorials/*.png).
INSTRUMENT = dict(wavelength=0.447, magnification=0.048,
                  numerical_aperture=1.45, n_m=1.34)


def run_pipeline(name: str = 'random_particles.png',
                 show: bool = True,
                 nfringes: int | None = 18,
                 diameter: int | None = 41,
                 percentile: int | None = 20,
                 separation: float | None = 50,
                 seed: int | None = 0,
                 n_starts: int = 1,
                 outdir: Path | None = None,
                 **instrument: float) -> Frame:
    '''Detect, estimate, and optimize every feature in an example image.

    Parameters
    ----------
    name : str, optional
        A bare filename in ``docs/simulated pair holograms/`` (or its
        ``random particles/`` subfolder) or ``docs/tutorials/``, or a
        path to any hologram normalized to 100 counts.  Default:
        ``'random_particles.png'``.
    show : bool, optional
        Display and save the detection boxes and per-feature
        data/fit/residual panels to ``outdir``.  Default: ``True``.
    outdir : Path, optional
        Folder for the saved figures.  Default: ``docs/<name stem>/``.
    nfringes : int, optional
        Fringes enclosed in each detection box.  Smaller boxes are less
        likely to merge dense fields into 3+ clusters that
        :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`
        discards.  Default: 18.  The :class:`Localizer` default (20)
        unnecessarily discards several features in both
        ``random_particles.png`` and ``random_particles_2.png``.  17
        also stays clean for both, but 16 starts splitting a genuine
        pair in ``random_particles_2.png`` into two contaminated
        singles (one redchi jumps to ~5.7); 18 keeps a bit of margin
        above that.
    diameter : int, optional
        ``trackpy.locate`` feature diameter, in pixels.  Default: 41.
        The :class:`Localizer` default (31) misses the blended core of
        ``overlapping_pair.png`` entirely; 41 also fits the wider pairs.
    percentile : int, optional
        ``trackpy.locate`` brightness cutoff.  Default: 20.  The
        :class:`Localizer` default (64) misses several real, fainter
        particles in ``random_particles.png``, including two that
        contaminate a neighboring single-particle fit instead of
        being recognized as their own pair.
    separation : float, optional
        Minimum pixel separation ``trackpy.locate`` requires between
        two detections.  Default: 50.  Trims a few spurious duplicate
        detections on the same ring introduced by the lower
        ``percentile``.
    seed : int, optional
        Random seed applied to every feature's DE-based estimator
        (:class:`~pylorenzmie.analysis.DEEstimator` or
        :class:`~pylorenzmie.analysis.PairEstimator`).  Default: 0.
        Both otherwise default to an unseeded search, so the same
        image can occasionally produce a different (sometimes worse)
        fit from one run to the next.  Pass ``None`` to restore that
        default, unseeded behavior.
    n_starts : int, optional
        Number of DE random restarts tried per feature; the restart
        with the lowest redchi is kept.  Default: 1 (a single attempt
        using ``seed``).  The DE search occasionally converges to a
        wrong local minimum -- one particle landing on an unphysical
        value (e.g. ``n_p`` near the search bound) or an inflated size
        that's really absorbing an unmodeled neighbor -- especially
        for :class:`Pair` fits, which search a higher-dimensional
        space than a single particle.  Raising this retries with a few
        different starting points and keeps the best, at roughly
        ``n_starts`` times the runtime; it fixes some cases (an
        unlucky search) but not others (a genuinely bad model fit,
        which every restart converges to alike).
    **instrument
        Instrument overrides (``wavelength`` [um], ``magnification``
        [um/pixel], ``n_m``, ...).  Default: the tutorial microscope.

    Returns
    -------
    frame : Frame
        The analyzed frame, with ``features`` and ``results`` populated.
    '''
    settings = {**INSTRUMENT, **instrument}
    if outdir is None:
        outdir = DOCS / Path(name).stem

    localizer = Localizer()
    if nfringes is not None:
        localizer.nfringes = nfringes
    if diameter is not None:
        localizer.diameter = diameter
    if percentile is not None:
        localizer.percentile = percentile
    if separation is not None:
        localizer.separation = separation

    frame = Frame(localizer=localizer)
    for key, value in settings.items():
        setattr(frame.instrument, key, value)
    frame.data = _load(name)

    print(f'image      : {name}  {frame.shape}')
    print(f'instrument : {settings}')
    print(f'localizer  : nfringes={localizer.nfringes} '
          f'diameter={localizer.diameter} '
          f'percentile={localizer.percentile} '
          f'separation={localizer.separation}')

    # Mask (used by both Optimizer and DEEstimator/PairEstimator for
    # random pixel subsampling) draws from NumPy's global RNG rather
    # than taking its own seed, so reproducing a run needs this too --
    # `feature.estimator.seed` alone only fixes the DE search itself.
    if seed is not None:
        np.random.seed(seed)

    n = frame.detect()
    _print_detections(frame, n)
    if n == 0:
        return frame

    start = perf_counter()
    winners = [_fit_feature(feature, seed, n_starts)
              for feature in frame.features]
    nsplit = frame.split_suspects()
    results = pd.DataFrame([feature.optimizer.result
                            for feature in frame.features])
    frame._results = results  # keep frame.results consistent with above
    _print_fits(frame, winners, results, perf_counter() - start)
    print(f'split {nsplit} single fit(s) into pairs')

    if show:
        _show(frame, name, outdir)
    return frame


def _fit_feature(feature: Feature, seed: int | None,
                 n_starts: int) -> tuple[int | None, int]:
    '''Estimate and optimize one feature with several DE restarts.

    Tries ``n_starts`` seeds (``seed, seed + 1, ...``, or ``n_starts``
    independent unseeded attempts if ``seed`` is ``None``), keeping
    whichever leaves the feature's model and optimizer result at the
    lowest redchi.

    Returns
    -------
    best_seed : int or None
        The winning restart's seed.
    n_tried : int
        Number of restarts actually tried.
    '''
    seeds = ([None] * n_starts if seed is None
             else [seed + i for i in range(n_starts)])

    best_seed, best_redchi = seeds[0], None
    for s in seeds:
        feature.estimator.seed = s
        feature.estimate()
        result = feature.optimize()
        if best_redchi is None or result.redchi < best_redchi:
            best_seed, best_redchi = s, result.redchi

    if best_seed != seeds[-1]:
        feature.estimator.seed = best_seed
        feature.estimate()
        feature.optimize()
    return best_seed, len(seeds)


def _print_detections(frame: Frame, n: int) -> None:
    '''One line per detected feature: kind, estimator, shape, bbox.'''
    print(f'\ndetected {n} feature(s):')
    for i, (feature, bbox) in enumerate(zip(frame.features, frame.bboxes)):
        kind = 'pair  ' if isinstance(feature.particle, Pair) else 'single'
        (x0, y0), w, h = bbox
        print(f'  [{i}] {kind}  {type(feature.estimator).__name__:<13} '
              f'shape={feature.shape}  '
              f'bbox=(({int(x0)}, {int(y0)}), {int(w)}, {int(h)})')


def _print_fits(frame: Frame, winners: list[tuple[int | None, int]],
                results: pd.DataFrame, elapsed: float) -> None:
    '''One line per feature (winning seed, success, npix, redchi),
    then the full results table.'''
    print(f'\nestimated + optimized in {elapsed:.2f} s '
          f'(best of up to {winners[0][1]} restart(s) per feature):')
    for i, (feature, (seed, n_tried)) in enumerate(zip(frame.features,
                                                        winners)):
        r = feature.optimizer.result
        kind = 'pair  ' if isinstance(feature.particle, Pair) else 'single'
        print(f'  [{i}] {kind} seed={seed}  success={bool(r.success)}  '
              f'npix={r.npix:.0f}  redchi={r.redchi:.2f}')
    with pd.option_context('display.width', 200, 'display.max_columns', 40):
        print()
        print(results.T)


def _show(frame: Frame, name: str, outdir: Path) -> None:
    '''Full frame with detection boxes; data/fit/residual per feature.
    Each figure is also saved into ``outdir``.'''
    outdir.mkdir(parents=True, exist_ok=True)

    _show_detections(frame, name, outdir)
    for i, feature in enumerate(frame.features):
        _show_feature(feature, i, name, outdir)
    plt.show()


def _show_detections(frame: Frame, name: str, outdir: Path) -> None:
    '''Full frame with detection boxes overlaid; saved as
    ``detections.png``.'''
    box_style = dict(fill=False, linewidth=2, edgecolor='red')
    fig, ax = plt.subplots(num=f'{name}: detections')
    ax.imshow(frame.data, cmap='gray')
    for (x0, y0), w, h in frame.bboxes:
        ax.add_patch(Rectangle((x0, y0), w, h, **box_style))
    ax.set_title(f'{name}: {len(frame.features)} feature(s)')
    fig.savefig(outdir / 'detections.png')


def _show_feature(feature: Feature, i: int, name: str,
                  outdir: Path) -> None:
    '''Data/fit/residual panels for one feature; saved as
    ``feature_<i>.png``.'''
    # Pin the model to the fitted optimum before predicting so the
    # residual reflects the reported fit, not the last DE/LM probe.
    result = feature.optimizer.result
    feature.model.properties = {v: result[v]
                                for v in feature.optimizer.variables}
    data = feature.data
    fit = feature.predicted()

    vmin, vmax = 0.9 * data.min(), 1.1 * data.max()
    img_style = dict(vmin=vmin, vmax=vmax, cmap='gray')
    # Normalized residuals, styled after lmtool/FitWidget.py's
    # CET-D1 display: (data - fit) / noise, levels fixed at +-10.
    noise = feature.model.instrument.noise
    res_style = dict(vmin=-10, vmax=10, cmap='coolwarm')
    fig, axes = plt.subplots(ncols=3, figsize=(10, 4),
                             constrained_layout=True,
                             num=f'{name}: feature {i}')
    panels = [(data, 'Data', img_style),
              (fit, 'Fit', img_style),
              ((data - fit) / noise, 'Residuals', res_style)]
    for ax, (img, label, style) in zip(axes, panels):
        ax.imshow(img, **style)
        ax.axis('off')
        ax.set_title(label)
    fig.savefig(outdir / f'feature_{i}.png')


if __name__ == '__main__':  # pragma: no cover
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else 'random_particles.png')
