'''
Torch implementation of mtd.py

Author: Sarah Odeh

Goal: make training data using holograms calculated by torchLM.py
'''
'''
torch implementation of mtd.py

Author: Sarah Odeh (NYU)
'''

import json
import numpy as np
from pathlib import Path
import cv2
import shutil
import torch
import os
import sys

from pylorenzmie.theory import Sphere
from pylorenzmie.theory import AberratedLorenzMie as _NumpyLM
from pylorenzmie.lib import LMObject
from pylorenzmie.theory import torchAberrated
from pylorenzmie.theory.torchLorenzMie import LorenzMieBatch

# create a particle that pre-computes the ab coefficients as torch tensors to be used by torchLM.py

class TorchParticle:

    def __init__(self, sphere, n_m, wavelength):
        r_p_np   = (sphere.r_p + sphere.r_0).astype(np.float32)     
        self.r_p = torch.from_numpy(r_p_np)

    def spheres_to_torch(spheres, n_m, wavelength):
        return [TorchParticle(s, n_m, wavelength) for s in spheres]

# borrowed from mtd.py
def feature_extent(sphere: Sphere, config: dict,
                   nfringes: int = 20, maxrange: int = 300) -> float:
    '''Radius of holographic feature in pixels.'''
    x = np.arange(0, maxrange)
    h = _NumpyLM(coordinates=x)
    h.instrument.properties = config['instrument']
    h.spherical = config['spherical']
    h.pupil     = config['pupil']
    h.particle.a_p = sphere.a_p
    h.particle.n_p = sphere.n_p
    h.particle.z_p = sphere.z_p
    b   = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    return maxrange if (len(ndx) <= nfringes) else float(ndx[nfringes])


def format_yolo(spheres: list[Sphere], config: dict) -> str:
    '''Returns a string of YOLO annotations.'''
    h, w   = config['shape']
    label  = 0                                    # one class
    fmt    = '{}' + 4 * ' {:.6f}' + '\n'
    annotation = ''
    for sphere in spheres:
        diameter = 2. * feature_extent(sphere, config)
        annotation += fmt.format(label,
                                 sphere.x_p / w,
                                 sphere.y_p / h,
                                 diameter   / w,
                                 diameter   / h)
    return annotation


def format_json(spheres: list[Sphere]) -> str:
    '''Returns a JSON string of sphere parameters.'''
    return json.dumps([s.to_json() for s in spheres], indent=4)

def make_value(range, decimals: int = 3):
    if np.isscalar(range):
        return range
    if range[0] == range[1]:
        return range[0]
    return np.around(np.random.uniform(range[0], range[1]), decimals)


def too_close(s1: Sphere, s2: Sphere, mpp: float) -> bool:
    d = mpp * np.square(s1.r_p - s2.r_p).sum()
    return d < (s1.a_p + s2.a_p)


def make_sphere(config: dict) -> Sphere:
    s = Sphere()
    for p in 'a_p n_p k_p x_p y_p z_p'.split():
        setattr(s, p, make_value(config[p]))
    return s


def make_spheres(config: dict) -> list[Sphere]:
    c        = config['particle']
    mpp      = config['instrument']['magnification']
    nspheres = np.random.randint(*c['nspheres'])
    spheres  = [make_sphere(c) for _ in range(nspheres)]
    for j in range(1, nspheres):
        for i in range(j):
            while too_close(spheres[i], spheres[j], mpp):
                spheres[j].x_p = make_value(c['x_p'])
                spheres[j].y_p = make_value(c['y_p'])
    return spheres

# hologram generation
def torch_mtd(configfile: str = 'mtd.json'):

    with open(configfile, 'r') as f:
        config = json.load(f)

    shape       = config['shape']
    nframes     = config['nframes']
    batch_size  = config.get('batch_size', 16)
    noise_sigma = config['noise']
    imgtype     = config['imgtype']

    inst        = config['instrument']
    wavelength  = inst['wavelength']
    n_m         = inst['n_m']
    NA          = inst['NA']
    magnification = inst['magnification']
    spherical   = config['spherical']

    model = LorenzMieBatch(
        shape         = shape,
        wavelength    = wavelength,
        n_m           = n_m,
        magnification = magnification,
        NA            = NA,
        spherical     = spherical,
        device        = None  
    )

    directory = Path(config['directory']).expanduser()
    jsondir   = directory / 'params'
    yolodir   = directory / 'images_labels'
    jsondir.mkdir(parents=True, exist_ok=True)
    yolodir.mkdir(parents=True, exist_ok=True)
    shutil.copy(configfile, directory)

    with open(directory / 'filenames.txt', 'w') as flist:

        frame_idx = 0
        while frame_idx < nframes:

            this_batch = min(batch_size, nframes - frame_idx)

            all_spheres  = [make_spheres(config) for _ in range(this_batch)]

            all_particles = [
                spheres_to_torch(spheres, n_m, wavelength)
                for spheres in all_spheres
            ]

            holos = model.batch_holograms(all_particles)   

            for b in range(this_batch):
                n     = frame_idx + b
                name  = f'image{n:04d}'
                jname = jsondir / (name + '.json')
                yname = yolodir / (name + '.txt')
                iname = yolodir / (name + '.' + imgtype)
                print(iname)

                frame = holos[b].cpu().numpy()                   
                frame += np.random.normal(0, noise_sigma, shape)
                frame  = np.clip(100 * frame, 0, 255).astype(np.uint8)
                cv2.imwrite(str(iname), frame)

                spheres = all_spheres[b]
                with open(jname, 'w') as f:
                    f.write(format_json(spheres))
                with open(yname, 'w') as f:
                    f.write(format_yolo(spheres, config))
                flist.write(name + '\n')

            frame_idx += this_batch
            print(f'Completed {frame_idx}/{nframes} frames')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='mtd.json',
                        help='configuration file')
    args = parser.parse_args()

    torch_mtd(args.configfile)
  
