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

``random particles/random_particles.png`` scatters a random number of
*independent* particles -- position, radius, refractive index, and
depth all drawn per-particle from a seeded RNG, not built from a fixed
template or constructed as pairs.  Nothing keeps them apart, so
proximity is incidental: most land alone, a few happen to fall close
enough to merge into a usable two-particle group, and a few crowd
together into a discarded three-plus cluster -- a messier, un-curated
frame than the tidy ``multi_pair.png`` grid, closer to a real field of
view.  It renders on its own, larger frame (:data:`RANDOM_SHAPE`) so
more particles fit while most still land clear of their neighbors.
``random_particles_2.png``, ``random_particles_3.png``, etc. (same
subfolder) are further independent draws from the same generator, one
per seed in :data:`RANDOM_SEEDS` -- different specific formations, not
a different kind of frame.

``random particles (confined)/random_particles_confined.png`` also scatters
independent particles with fully random position/size/index/depth, but
each group (a lone particle, or two close enough to merge into a pair)
is placed by rejection sampling so it clears every other group's
Localizer box -- see :func:`_confined_scatter`.  No group of 3+ ever
forms, so nothing gets discarded: every particle is either a clean
single or part of a clean pair.  One seed per
:data:`CONFINED_SEEDS` entry, same numbering scheme as the
unconfined family.

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
OUTDIR = DOCS / 'tutorials' / 'simulated pair holograms'

INSTRUMENT = dict(wavelength=0.447, magnification=0.048,
                  n_m=1.340, noise=0.05)

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
#
# One seed per random-scatter image: RANDOM_SEEDS[0] is
# ``random_particles.png``, RANDOM_SEEDS[1] is
# ``random_particles_2.png``, and so on -- append a seed to add
# another independent draw.
RANDOM_SEEDS = [25, 29, 2, 30, 21]
RANDOM_SHAPE = (2048, 2048)
N_PARTICLES_RANGE = (24, 36)  # rng.integers high is exclusive: 24-35.
PARTICLE_A_RANGE = (0.55, 0.75)    # radius, um.
PARTICLE_N_RANGE = (1.41, 1.46)    # refractive index.
PARTICLE_Z_RANGE = (200., 300.)    # depth, pixels.
MARGIN = 150.                # keeps particles comfortably inside the frame.


def _random_particles(seed: int = RANDOM_SEEDS[0],
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
        depths).  Default: ``RANDOM_SEEDS[0]``.
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


# devel/make_test_holograms.py:_confined_scatter -- like
# _random_particles (independent random position/size/index/depth per
# particle), but every group (a lone particle, or two close enough to
# merge into a pair) is placed by rejection sampling so it clears every
# other group's Localizer box: no group of 3+ can ever form, so nothing
# gets discarded.
#
# CONFINED_FOOTPRINT_RADIUS is a conservative box-radius estimate for
# PARTICLE_A_RANGE/PARTICLE_N_RANGE/PARTICLE_Z_RANGE (empirically,
# actual boxes run roughly 90-130 px in radius); CONFINED_CLEARANCE is
# extra margin on top of that, keeping different groups' boxes apart.
#
# The clearance check alone isn't sufficient: CONFINED_SEPARATION_RANGE's
# lower bound also has to stay clear of the regime where trackpy itself
# splits one tightly-blended pair into 3+ spurious sub-detections (the
# same failure ``overlapping_pair.png`` hits at 50 px, needing
# diameter=41) -- that would masquerade as a 3+ cluster even though the
# two groups involved never actually collided. 110 px is the separation
# ``close_pair.png``/``multi_pair.png`` already validate as always
# resolving cleanly to 2 detections at the default diameter=31.
CONFINED_SEEDS = [2, 5, 8, 13, 19]
CONFINED_N_GROUPS_RANGE = (15, 26)   # rng.integers high is exclusive.
CONFINED_PAIR_PROBABILITY = 0.35     # fraction of groups that are pairs.
CONFINED_SEPARATION_RANGE = (110., 170.)   # pixels, within a pair.
CONFINED_FOOTPRINT_RADIUS = 140.
CONFINED_CLEARANCE = 40.
CONFINED_MAX_ATTEMPTS = 300    # per group, before giving up on it.

# random_particles_confined_6.png: same singles-or-pairs-only placement
# as the rest of the confined family (group-to-group clearance still
# guarantees no 3+ cluster can ever form -- see _confined_scatter), but
# the per-pair separation floor is dropped far below
# CONFINED_SEPARATION_RANGE's 110 px, so some pairs land almost merged.
# A deliberately hard case for how the pipeline handles badly-blended
# fringe systems, unlike the rest of the confined family which stays in
# the cleanly-resolvable regime. Seed 43 gives separations from 13 px
# (nearly on top of each other) up to a normal 169 px.
CONFINED_6_SEED = 43
CONFINED_6_SEPARATION_RANGE = (10., 170.)

# random_particles_confined_7.png -- second draw from the same
# deliberately-hard CONFINED_6_SEPARATION_RANGE as _6, a different
# formation for the same stress test. Seed 54 gives separations from
# 15 px (nearly merged) up to 158 px, across 9 pairs.
CONFINED_7_SEED = 54


def _confined_scatter(seed: int,
                      shape: tuple[int, int] = RANDOM_SHAPE,
                      n_range: tuple[int, int] = CONFINED_N_GROUPS_RANGE,
                      pair_probability: float = CONFINED_PAIR_PROBABILITY,
                      sep_range: tuple[float, float] = (
                          CONFINED_SEPARATION_RANGE),
                      a_range: tuple[float, float] = PARTICLE_A_RANGE,
                      n_p_range: tuple[float, float] = PARTICLE_N_RANGE,
                      z_range: tuple[float, float] = PARTICLE_Z_RANGE,
                      margin: float = MARGIN,
                      footprint_radius: float = CONFINED_FOOTPRINT_RADIUS,
                      clearance: float = CONFINED_CLEARANCE,
                      max_attempts: int = CONFINED_MAX_ATTEMPTS,
                      ) -> list[Sphere]:
    '''Random singles and pairs only -- no group of 3+.

    Each group (a lone particle, or a pair) gets a footprint radius
    (``footprint_radius``, plus half its separation if it is a pair)
    and is placed at a random candidate center, retried up to
    ``max_attempts`` times until it clears every already-placed
    group's center by both radii plus ``clearance``.  A group that
    never finds a clear spot is skipped, so the final particle count
    can run under what ``n_range`` and ``pair_probability`` suggest.
    Position, radius, refractive index, and depth are otherwise all
    independently random, exactly as in :func:`_random_particles`.

    Parameters
    ----------
    seed : int
        RNG seed for the layout (group count/kind/order, positions,
        separations, orientations, sizes, indices, depths).
    shape : tuple[int, int], optional
        Frame shape groups are scattered over.  Default: :data:`RANDOM_SHAPE`.
    n_range : tuple[int, int], optional
        ``[low, high)`` bounds on the number of groups (not particles).
        Default: :data:`CONFINED_N_GROUPS_RANGE`.
    pair_probability : float, optional
        Probability that a given group is a pair rather than a single.
        Default: :data:`CONFINED_PAIR_PROBABILITY`.
    sep_range : tuple[float, float], optional
        ``[low, high]`` bounds on a pair's lateral core separation, in
        pixels.  Default: :data:`CONFINED_SEPARATION_RANGE`.
    a_range, n_p_range, z_range, margin : optional
        As in :func:`_random_particles`.
    footprint_radius : float, optional
        Conservative per-particle box radius used for the clearance
        check, in pixels.  Default: :data:`CONFINED_FOOTPRINT_RADIUS`.
    clearance : float, optional
        Extra gap required between two groups' footprints, in pixels.
        Default: :data:`CONFINED_CLEARANCE`.
    max_attempts : int, optional
        Placement attempts per group before it is skipped.
        Default: :data:`CONFINED_MAX_ATTEMPTS`.

    Returns
    -------
    particles : list[Sphere]
        One or two spheres per placed group.
    '''
    rng = np.random.default_rng(seed)
    n_groups = rng.integers(*n_range)
    centers: list[tuple[float, float]] = []
    radii: list[float] = []
    particles: list[Sphere] = []
    for _ in range(n_groups):
        is_pair = rng.random() < pair_probability
        separation = rng.uniform(*sep_range) if is_pair else 0.
        radius = footprint_radius + separation / 2.

        for _attempt in range(max_attempts):
            x = rng.uniform(margin, shape[1] - margin)
            y = rng.uniform(margin, shape[0] - margin)
            if all(np.hypot(x - cx, y - cy) >= radius + r + clearance
                  for (cx, cy), r in zip(centers, radii)):
                break
        else:
            continue

        centers.append((x, y))
        radii.append(radius)
        offsets = [(0., 0.)]
        if is_pair:
            angle = rng.uniform(0., 2. * np.pi)
            dx = separation / 2. * np.cos(angle)
            dy = separation / 2. * np.sin(angle)
            offsets = [(dx, dy), (-dx, -dy)]
        for ox, oy in offsets:
            particles.append(_sphere(x + ox, y + oy,
                                     rng.uniform(*z_range),
                                     rng.uniform(*a_range),
                                     rng.uniform(*n_p_range)))
    return particles


# Lateral core separation, in pixels.  The tightest pair needs the
# Localizer run with diameter=41 (the default 31 splits the blended
# core into spurious sub-detections); the other two work at defaults.
CASES = {
    'overlapping_pair.png': _pair(50.),
    'close_pair.png':       _pair(110.),
    'separated_pair.png':   _pair(230.),
    'multi_pair.png':       _multi(110., GRID),
}

# Cases whose frame is not the standard SHAPE.  Looked up by _render,
# make(), and the ground-truth JSON.  Keyed the same way as CASES: a
# path relative to OUTDIR -- a bare filename, or 'subdir/name.png' for
# a case rendered into its own subfolder.
CASE_SHAPES = {}


def _numbered_name(i: int, base: str = 'random_particles') -> str:
    '''``<base>.png`` for ``i == 1``, else ``<base>_<i>.png``.'''
    return f'{base}.png' if i == 1 else f'{base}_{i}.png'


# One random_particles[_N].png per entry in RANDOM_SEEDS, in
# 'random particles/': fully independent particles, no placement
# constraint, so some fall into a discarded 3+ cluster.
for _i, _seed in enumerate(RANDOM_SEEDS, start=1):
    _path = f'random particles/{_numbered_name(_i)}'
    CASES[_path] = _random_particles(seed=_seed)
    CASE_SHAPES[_path] = RANDOM_SHAPE

# One random_particles_confined[_N].png per entry in CONFINED_SEEDS, in
# 'random particles (confined)/': random singles and pairs only, no
# group of 3+ -- see _confined_scatter.
for _i, _seed in enumerate(CONFINED_SEEDS, start=1):
    _name = _numbered_name(_i, base='random_particles_confined')
    _path = f'random particles (confined)/{_name}'
    CASES[_path] = _confined_scatter(seed=_seed)
    CASE_SHAPES[_path] = RANDOM_SHAPE

# random_particles_confined_6.png -- see CONFINED_6_SEPARATION_RANGE.
_path6 = 'random particles (confined)/random_particles_confined_6.png'
CASES[_path6] = _confined_scatter(seed=CONFINED_6_SEED,
                                  sep_range=CONFINED_6_SEPARATION_RANGE)
CASE_SHAPES[_path6] = RANDOM_SHAPE

# random_particles_confined_7.png -- see CONFINED_7_SEED.
_path7 = 'random particles (confined)/random_particles_confined_7.png'
CASES[_path7] = _confined_scatter(seed=CONFINED_7_SEED,
                                  sep_range=CONFINED_6_SEPARATION_RANGE)
CASE_SHAPES[_path7] = RANDOM_SHAPE


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
        path = outdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), counts)
        truth['holograms'][name] = dict(shape=list(shape),
                                        particles=_describe(particles))
        span = f'[{data.min():.2f}, {data.max():.2f}]'
        print(f'{name:<40} {shape}  {span}')
        for p in truth['holograms'][name]['particles']:
            print('    ' + '  '.join(f'{k}={v:.3f}' for k, v in p.items()))
    (outdir / 'ground_truth.json').write_text(json.dumps(truth, indent=2))
    print(f'\nwrote {len(CASES)} holograms + ground_truth.json to {outdir}')


if __name__ == '__main__':  # pragma: no cover
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR)
