'''Render the hologram on the splash screen of lmtool.

The splash screen appears before pylorenzmie.theory is imported, so the
hologram is computed once, here, and shipped as lmtool/splash.png.

Usage: python devel/make_splash.py
'''

from pathlib import Path

import numpy as np
from PIL import Image

from pylorenzmie.theory import Instrument, LorenzMie, Sphere

WIDTH, HEIGHT = 960, 480      # twice the size on the screen (retina)
CENTER = (300, 150)           # of the particle: upper left
CONTRAST = 0.7                # of the fringes: low, so the text stays legible
PURPLE = np.array([87, 6, 140]) / 255.   # NYU purple

particle = Sphere(x_p=CENTER[0], y_p=CENTER[1], z_p=500., a_p=1.0, n_p=1.45)
instrument = Instrument(wavelength=0.447, magnification=0.048, n_m=1.34)
coordinates = LorenzMie.meshgrid((HEIGHT, WIDTH))
model = LorenzMie(coordinates=coordinates, particle=particle,
                  instrument=instrument)
hologram = model.hologram().reshape(HEIGHT, WIDTH)

# fringes modulate the brightness of the background color, and a vignette
# darkens the corners
y, x = np.mgrid[:HEIGHT, :WIDTH]
r = np.hypot((x - WIDTH/2) / WIDTH, (y - HEIGHT/2) / HEIGHT)
vignette = 1. - 0.8 * r**2
brightness = (1. + CONTRAST * (hologram - 1.)) * vignette
rgb = np.clip(PURPLE * brightness[..., None] * 1.15, 0., 1.)

path = Path(__file__).parent.parent / 'lmtool' / 'splash.png'
Image.fromarray((255 * rgb).astype(np.uint8)).save(path, optimize=True)
print(path, path.stat().st_size, 'bytes')
