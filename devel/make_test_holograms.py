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


def _pair(separation: float) -> list[Sphere]:
    '''SPHERE_A and SPHERE_B centered in the frame, split by ``separation``
    pixels along the diagonal (B up-and-left of A, as in speed_example).'''
    cx, cy = CENTER
    d = separation / 2. / 2. ** 0.5
    return [_sphere(cx + d, cy + d,
                    SPHERE_A['z_p'], SPHERE_A['a_p'], SPHERE_A['n_p']),
            _sphere(cx - d, cy - d,
                    SPHERE_B['z_p'], SPHERE_B['a_p'], SPHERE_B['n_p'])]


# Lateral core separation, in pixels.  The tightest pair needs the
# Localizer run with diameter=41 (the default 31 splits the blended
# core into spurious sub-detections); the other two work at defaults.
CASES = {
    'overlapping_pair.png': _pair(50.),
    'close_pair.png':       _pair(110.),
    'separated_pair.png':   _pair(230.),
}


def _render(particles: list[Sphere], seed: int) -> np.ndarray:
    '''Noisy hologram of ``particles`` on the standard grid.'''
    model = LorenzMie(coordinates=LMObject.meshgrid(SHAPE),
                      instrument=Instrument(**INSTRUMENT))
    model.particle = particles
    hologram = model.hologram().reshape(SHAPE)
    noise = np.random.default_rng(seed).normal(0., INSTRUMENT['noise'], SHAPE)
    return hologram + noise


def _describe(particles: list[Sphere]) -> list[dict]:
    return [dict(x_p=p.r_p[0], y_p=p.r_p[1], z_p=p.r_p[2],
                 a_p=p.a_p, n_p=p.n_p) for p in particles]


def make(outdir: Path = OUTDIR) -> None:
    '''Render every case in :data:`CASES` and write PNGs plus a JSON key.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    truth = {'instrument': INSTRUMENT, 'shape': list(SHAPE), 'holograms': {}}
    for seed, (name, particles) in enumerate(CASES.items()):
        data = _render(particles, seed)
        counts = np.clip(data * 100., 0, 255).astype(np.uint8)
        cv2.imwrite(str(outdir / name), counts)
        truth['holograms'][name] = _describe(particles)
        span = f'[{data.min():.2f}, {data.max():.2f}]'
        print(f'{name:<22} {SHAPE}  {span}')
        for p in truth['holograms'][name]:
            print('    ' + '  '.join(f'{k}={v:.3f}' for k, v in p.items()))
    (outdir / 'ground_truth.json').write_text(json.dumps(truth, indent=2))
    print(f'\nwrote {len(CASES)} holograms + ground_truth.json to {outdir}')


if __name__ == '__main__':  # pragma: no cover
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR)
