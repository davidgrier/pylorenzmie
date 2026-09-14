'''Synthesize overlapping two-particle holograms for the analysis pipeline.

None of the ``docs/tutorials/`` images contain a genuinely overlapping
pair, so this renders a few with the forward model
(:class:`~pylorenzmie.theory.LorenzMie` over two
:class:`~pylorenzmie.theory.Sphere` scatterers), adds Gaussian noise at
the instrument level, and writes 8-bit grayscale PNGs (background = 100
counts) into ``docs/simulated pair holograms/``.

The two scatterers are the first two particles from
``devel/speed_example.py`` -- a small high-index sphere and a larger
low-index one at different depths.  Only their lateral separation
changes between cases, to vary how much the fringe systems overlap.

``multi_pair.png`` places four such pairs on a 2x2 grid in one frame:
each pair's cores overlap (so
:func:`~pylorenzmie.analysis.pair_grouping.group_overlapping` merges its
two detection boxes into a two-particle group), but the pairs are spaced
far enough apart that no group grows to three-plus boxes and gets
discarded -- so every pair yields a usable fit.

``random_pairs.png`` scatters a random number of *independent*
particles -- position, radius, refractive index, and depth all drawn
per-particle from a seeded RNG, not built from a fixed template or
constructed as pairs.  Nothing keeps them apart, so proximity is
incidental: most land alone, a few happen to fall close enough to
merge into a usable two-particle group, and a few crowd into a
discarded three-plus cluster -- a messier, un-curated frame than the
tidy ``multi_pair.png`` grid, closer to a real field of view.  It
renders on its own, larger frame (:data:`RANDOM_SHAPE`) so more
particles fit while most still land clear of their neighbors.
``random_pairs_2.png`` is a second, independent draw from the same
generator (:data:`RANDOM_SEED_2`) -- a different specific formation,
not a different kind of frame.

Usage
-----
    python devel/make_test_holograms.py [outdir]

Ground-truth parameters are printed and written next to the images as
``ground_truth.json``.
'''

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pylorenzmie.lib import LMObject
from pylorenzmie.theory import Instrument, LorenzMie, Sphere


DOCS = Path(__file__).resolve().parent.parent / 'docs'
OUTDIR = DOCS / 'simulated pair holograms'

INSTRUMENT = dict(wavelength=0.447, magnification=0.048,
                  numerical_aperture=1.45, n_m=1.340, noise=0.05)

SHAPE = (1024, 1024)
CENTER = (SHAPE[1] / 2., SHAPE[0] / 2.)

# devel/speed_example.py particles 0 and 1: (a_p [um], n_p, z_p [pixels]).
SPHERE_A = dict(a_p=0.5, n_p=1.45, z_p=200.)
SPHERE_B = dict(a_p=1.0, n_p=1.40, z_p=300.)


def _sphere(x: float, y: float, z: float, a_p: float, n_p: float) -> Sphere:
    p = Sphere()
    p.r_p = [x, y, z]
    p.a_p = a_p
    p.n_p = n_p
    return p


def _pair(separation: float,
          center: tuple[float, float] = CENTER) -> list[Sphere]:
    '''SPHERE_A and SPHERE_B split by ``separation`` pixels along the
    diagonal (B up-and-left of A, as in speed_example), centered on
    ``center`` (default: the frame center).'''
    cx, cy = center
    d = separation / 2. / 2. ** 0.5
    return [_sphere(cx + d, cy + d,
                    SPHERE_A['z_p'], SPHERE_A['a_p'], SPHERE_A['n_p']),
            _sphere(cx - d, cy - d,
                    SPHERE_B['z_p'], SPHERE_B['a_p'], SPHERE_B['n_p'])]


# Cell centers for the 2x2 grid of pairs in ``multi_pair.png``, near the
# quarter and three-quarter points of the 1024 px frame.  At this
# spacing the pair boxes clear each other by ~140 px -- each pair stays
# a two-box group and no group grows past two.
GRID = [(300., 300.), (724., 300.), (300., 724.), (724., 724.)]


def _multi(separation: float,
           centers: list[tuple[float, float]]) -> list[Sphere]:
    '''Flattened spheres for one :func:`_pair` at each of ``centers``.'''
    return [sphere for c in centers for sphere in _pair(separation, c)]


# devel/make_test_holograms.py:_random_particles -- generation seed
# (distinct from the per-case noise seed _render assigns via CASES
# order), the ranges particle count/size/index/depth/margin are drawn
# from, and the frame it renders on.
#
# a_p and n_p are each drawn independently, but kept narrower than the
# other cases' 0.5-1.0 um / 1.40-1.45 range: trackpy.locate applies one
# frame-wide brightness threshold, so once ~30 particles share a frame,
# a faint small/low-index one next to a bright large/high-index one
# falls below it and is missed entirely (verified empirically -- the
# full 0.5-1.0/1.38-1.48 range left a third or more of the particles
# undetected). This range keeps every particle's fringe contrast within
# roughly the same order of magnitude, so nearly all stay visible.
RANDOM_SEED = 25
RANDOM_SEED_2 = 29    # random_pairs_2.png: a second, independent draw.
RANDOM_SHAPE = (2048, 2048)
N_PARTICLES_RANGE = (24, 36)  # rng.integers high is exclusive: 24-35.
PARTICLE_A_RANGE = (0.55, 0.75)    # radius, um.
PARTICLE_N_RANGE = (1.41, 1.46)    # refractive index.
PARTICLE_Z_RANGE = (200., 300.)    # depth, pixels.
MARGIN = 150.                # keeps particles comfortably inside the frame.


def _random_particles(seed: int = RANDOM_SEED,
                      shape: tuple[int, int] = RANDOM_SHAPE,
                      n_range: tuple[int, int] = N_PARTICLES_RANGE,
                      a_range: tuple[float, float] = PARTICLE_A_RANGE,
                      n_p_range: tuple[float, float] = PARTICLE_N_RANGE,
                      z_range: tuple[float, float] = PARTICLE_Z_RANGE,
                      margin: float = MARGIN) -> list[Sphere]:
    '''Independent particles scattered at random -- not built from a
    fixed template or constructed as pairs.

    Position, radius, refractive index, and depth are all drawn
    independently per particle, with no collision check between them,
    so any "pairs" that result are incidental proximity rather than
    deliberate construction: most particles land alone, some happen to
    fall close enough to merge into a usable two-particle group, and a
    few crowd together into a group
    :func:`~pylorenzmie.analysis.pair_grouping.group_overlapping`
    discards.

    Parameters
    ----------
    seed : int, optional
        RNG seed for the layout (count, positions, sizes, indices,
        depths).  Default: :data:`RANDOM_SEED`.
    shape : tuple[int, int], optional
        Frame shape particles are scattered over.
        Default: :data:`RANDOM_SHAPE`.
    n_range : tuple[int, int], optional
        ``[low, high)`` bounds on the number of particles.
        Default: :data:`N_PARTICLES_RANGE`.
    a_range : tuple[float, float], optional
        ``[low, high]`` bounds on particle radius, in um.
        Default: :data:`PARTICLE_A_RANGE`.
    n_p_range : tuple[float, float], optional
        ``[low, high]`` bounds on particle refractive index.
        Default: :data:`PARTICLE_N_RANGE`.
    z_range : tuple[float, float], optional
        ``[low, high]`` bounds on particle depth, in pixels.
        Default: :data:`PARTICLE_Z_RANGE`.
    margin : float, optional
        Minimum distance from a particle to the frame edge, in pixels.
        Default: :data:`MARGIN`.

    Returns
    -------
    particles : list[Sphere]
        One sphere per particle.
    '''
    rng = np.random.default_rng(seed)
    n = rng.integers(*n_range)
    x = rng.uniform(margin, shape[1] - margin, n)
    y = rng.uniform(margin, shape[0] - margin, n)
    z = rng.uniform(*z_range, n)
    a_p = rng.uniform(*a_range, n)
    n_p = rng.uniform(*n_p_range, n)
    return [_sphere(xi, yi, zi, ai, ni)
            for xi, yi, zi, ai, ni in zip(x, y, z, a_p, n_p)]


# Lateral core separation, in pixels.  The tightest pair needs the
# Localizer run with diameter=41 (the default 31 splits the blended
# core into spurious sub-detections); the other two work at defaults.
CASES = {
    'overlapping_pair.png': _pair(50.),
    'close_pair.png':       _pair(110.),
    'separated_pair.png':   _pair(230.),
    'multi_pair.png':       _multi(110., GRID),
    'random_pairs.png':     _random_particles(),
    'random_pairs_2.png':   _random_particles(seed=RANDOM_SEED_2),
}

# Cases whose frame is not the standard SHAPE.  Looked up by _render,
# make(), and the ground-truth JSON.
CASE_SHAPES = {'random_pairs.png': RANDOM_SHAPE,
               'random_pairs_2.png': RANDOM_SHAPE}


def _render(particles: list[Sphere], seed: int,
           shape: tuple[int, int] = SHAPE) -> np.ndarray:
    '''Noisy hologram of ``particles`` on ``shape`` (default: SHAPE).'''
    model = LorenzMie(coordinates=LMObject.meshgrid(shape),
                      instrument=Instrument(**INSTRUMENT))
    model.particle = particles
    hologram = model.hologram().reshape(shape)
    noise = np.random.default_rng(seed).normal(0., INSTRUMENT['noise'], shape)
    return hologram + noise


def _describe(particles: list[Sphere]) -> list[dict]:
    return [dict(x_p=p.r_p[0], y_p=p.r_p[1], z_p=p.r_p[2],
                 a_p=p.a_p, n_p=p.n_p) for p in particles]


def make(outdir: Path = OUTDIR) -> None:
    '''Render every case in :data:`CASES` and write PNGs plus a JSON key.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth = {'instrument': INSTRUMENT, 'holograms': {}}
    for seed, (name, particles) in enumerate(CASES.items()):
        shape = CASE_SHAPES.get(name, SHAPE)
        data = _render(particles, seed, shape)
        counts = np.clip(data * 100., 0, 255).astype(np.uint8)
        cv2.imwrite(str(outdir / name), counts)
        truth['holograms'][name] = dict(shape=list(shape),
                                        particles=_describe(particles))
        span = f'[{data.min():.2f}, {data.max():.2f}]'
        print(f'{name:<22} {shape}  {span}')
        for p in truth['holograms'][name]['particles']:
            print('    ' + '  '.join(f'{k}={v:.3f}' for k, v in p.items()))
    (outdir / 'ground_truth.json').write_text(json.dumps(truth, indent=2))
    print(f'\nwrote {len(CASES)} holograms + ground_truth.json to {outdir}')


if __name__ == '__main__':  # pragma: no cover
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR)
