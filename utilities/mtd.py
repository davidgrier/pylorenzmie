#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Make Training Data'''

import json
from pylorenzmie.theory import Sphere
from pylorenzmie.theory import AberratedLorenzMie as LorenzMie
from pylorenzmie.utilities import coordinates
import numpy as np
from pathlib import Path
import cv2
import shutil


def feature_extent(sphere, config, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''
    x = np.arange(0, maxrange)
    h = LorenzMie(coordinates=x)
    h.instrument.properties = config['instrument']
    h.spherical = config['spherical']
    h.pupil = config['pupil']
    h.particle.a_p = sphere.a_p
    h.particle.n_p = sphere.n_p
    h.particle.z_p = sphere.z_p
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    return maxrange if (len(ndx) <= nfringes) else float(ndx[nfringes])


def format_yolo(spheres, config):
    '''Returns a string of YOLO annotations'''
    h, w = config['shape']
    type = 0  # one class for now
    fmt = '{}' + 4 * ' {:.6f}' + '\n'
    annotation = ''
    for sphere in spheres:
        diameter = 2. * feature_extent(sphere, config)
        x_p = sphere.x_p / w
        y_p = sphere.y_p / h
        w_p = diameter / w
        h_p = diameter / h
        annotation += fmt.format(type, x_p, y_p, w_p, h_p)
    return annotation


def format_json(spheres):
    '''Returns a string of JSON annotations'''
    annotation = [s.to_json() for s in spheres]
    return json.dumps(annotation, indent=4)


def make_value(range, decimals=3):
    '''Returns the value for a property'''
    if np.isscalar(range):
        value = range
    elif range[0] == range[1]:
        value = range[0]
    else:
        value = np.random.uniform(range[0], range[1])
    return np.around(value, decimals=decimals)


def too_close(s1, s2, mpp):
    d = mpp*np.square(s1.r_p - s2.r_p).sum()
    return d < (s1.a_p + s2.a_p)


def make_sphere(config):
    s = Sphere()
    for p in 'a_p n_p k_p x_p y_p z_p'.split():
        setattr(s, p, make_value(config[p]))
    return s


def make_spheres(config):
    c = config['particle']
    mpp = config['instrument']['magnification']
    nrange = c['nspheres']
    nspheres = np.random.randint(*nrange)
    spheres = [make_sphere(c) for _ in range(nspheres)]
    for j in range(1, nspheres):
        for i in range(0, j):
            while too_close(spheres[i], spheres[j], mpp):
                spheres[j].x_p = make_value(c['x_p'])
                spheres[j].y_p = make_value(c['y_p'])
    return spheres


def mtd(configfile='mtd.json'):
    '''Make Training Data'''
    # read configuration
    with open(configfile, 'r') as f:
        config = json.load(f)

    # set up pipeline for hologram calculation
    shape = config['shape']
    coords = coordinates(shape)
    holo = LorenzMie(coordinates=coords)
    holo.instrument.properties = config['instrument']
    holo.spherical = config['spherical']
    holo.pupil = config['pupil']

    # create directories and filenames
    directory = Path(config['directory']).expanduser()
    jsondir = directory / 'params'
    yolodir = directory / 'images_labels'
    jsondir.mkdir(parents=True, exist_ok=True)
    yolodir.mkdir(parents=True, exist_ok=True)

    shutil.copy(configfile, directory)

    flist = open(directory / 'filenames.txt', 'w')
    for n in range(config['nframes']):
        name = f'image{n:04d}'
        jname = jsondir / (name + '.json')
        yname = yolodir / (name + '.txt')
        iname = yolodir / (name + '.' + config['imgtype'])
        print(iname)

        spheres = make_spheres(config)
        holo.particle = spheres
        frame = holo.hologram().reshape(shape)
        frame += np.random.normal(0, config['noise'], shape)
        frame = np.clip(100*frame, 0, 255).astype(np.uint8)
        cv2.imwrite(str(iname), frame)
        with open(jname, 'w') as f:
            f.write(format_json(spheres))
        with open(yname, 'w') as f:
            f.write(format_yolo(spheres, config))
        flist.write(name + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='mtd.json',
                        help='configuration file')
    args = parser.parse_args()

    mtd(args.configfile)
