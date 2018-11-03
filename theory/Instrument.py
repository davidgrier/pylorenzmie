# /usr/bin/env python
# -*- coding: utf-8 -*-


class Instrument(object):
    '''Abstraction of an in-line holographic microscope for
    Lorenz-Mie microscopy'''

    def __init__(self,
                 wavelength=None,
                 magnification=None,
                 mm=None):
        if wavelength is None:
            self.wavelength = 0.532
        else:
            self.wavelength = wavelength

        if magnification is None:
            self.magnification = 0.135
        else:
            self.magnification = magnification

        if mm is None:
            self.mm = 1.335
        else:
            self.mm = mm

    @property
    def wavelength(self):
        '''Wavelength of light in vacuum [um]'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = float(wavelength)

    @property
    def magnification(self):
        '''Magnification of microscope [um/pixel]'''
        return self._magnification

    @magnification.setter
    def magnification(self, magnification):
        self._magnification = float(magnification)

    @property
    def mm(self):
        '''Complex refractive index of medium'''
        return self._mm

    @mm.setter
    def mm(self, mm):
        self._mm = complex(mm)


if __name__ == '__main__':
    a = Instrument()
    print(a.wavelength, a.magnification)
