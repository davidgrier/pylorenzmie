#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Make Training Data'''
import sys
sys.path.append('/home/lea336/')
import json
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.Sphere import Sphere
import numpy as np

import cv2
import os
import shutil


def feature_extent(sphere, config, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''
    s = Sphere()
    s.a_p = sphere.a_p
    s.n_p = sphere.n_p
    s.z_p = sphere.z_p
    h = LMHologram(coordinates=np.arange(maxrange))
    h.instrument.properties = config['instrument']
    h.particle = sphere
    # roughly estimate radii of zero crossings
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    if len(ndx) < (nfringes+1):
        return maxrange
    else:
        return float(ndx[nfringes])


def format_yolo(sample, config):
    '''Returns a string of YOLO annotations'''
    (h, w) = config['shape']
    type = 0  # one class for now
    fmt = '{}' + 4 * ' {:.6f}' + '\n'
    annotation = ''
    for sphere in sample:
        diameter = 2. * feature_extent(sphere, config)
        x_p = sphere.x_p / w
        y_p = sphere.y_p / h
        w_p = diameter / w
        h_p = diameter / h
        annotation += fmt.format(type, x_p, y_p, w_p, h_p)
    return annotation


def format_json(sample, config):
    '''Returns a string of JSON annotations'''
    annotation = []
    for s in sample:
        annotation.append(s.dumps(sort_keys=True))
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


def make_sample(config):
    '''Returns an array of Sphere objects'''
    particle = config['particle']
    nrange = particle['nspheres']
    nspheres = np.random.randint(nrange[0], nrange[1])
    sample = []
    for n in range(nspheres):
        sphere = Sphere()
        for prop in ('a_p', 'n_p', 'k_p', 'x_p', 'y_p', 'z_p'):
            setattr(sphere, prop, make_value(particle[prop]))
        sample.append(sphere)
    return sample


def mtd(configfile='mtd.json'):
    '''Make Training Data'''
    # read configuration
    with open(configfile, 'r') as f:
        config = json.load(f)

    # set up pipeline for hologram calculation
    shape = config['shape']
    holo = LMHologram(coordinates=coordinates(shape))
    holo.instrument.properties = config['instrument']

    # create directories and filenames
    directory = os.path.expanduser(config['directory'])
    imgtype = config['imgtype']
    for dir in ('images', 'params', 'labels'):
        if not os.path.exists(os.path.join(directory, dir)):
            os.makedirs(os.path.join(directory, dir))
    shutil.copy2(configfile, directory)
    imgname = os.path.join(directory, 'images', 'image{:04d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:04d}.json')
    yoloname = os.path.join(directory, 'labels', 'image{:04d}.txt')

    for n in range(config['nframes']):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config)   # ... get params for particles
        # ... calculate hologram
        frame = np.random.normal(0, config['noise'], shape)
        if len(sample) > 0:
            holo.particle = sample
            frame += holo.hologram().reshape(shape)
        else:
            frame += 1.
        frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
        # ... and save the results
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config))
        with open(yoloname.format(n), 'w') as fp:
            fp.write(format_yolo(sample, config))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', type=str,
                        nargs='?', default='mtd.json',
                        help='configuration file')
    args = parser.parse_args()

    mtd(args.configfile)
