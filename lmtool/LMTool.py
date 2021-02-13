# /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from scipy.interpolate import (BSpline, splrep)
import os
import json
import cv2
import numpy as np
import pyqtgraph as pg
import pylorenzmie as pylm

from pylorenzmie.theory import LMHologram
from pylorenzmie.analysis import Feature
from pylorenzmie.utilities import (coordinates, azistd)

from LMTool_Ui import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtCore

logger = logging.getLogger('LMTool')
logger.setLevel(logging.INFO)


class LMTool(QtWidgets.QMainWindow):

    def __init__(self,
                 filename=None,
                 background=None,
                 normalization=None,
                 data=None):
        super(LMTool, self).__init__()
        if background is not None:
            self.openBackground(background)
        elif normalization is not None:
            self.background = normalization
        else:
            self.background = 1.
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupParameters()
        self.maxrange = int(self.ui.bbox.value() // 2)
        self.setupImageTab()
        self.setupProfileTab()
        self.setupFitTab()
        self.setupTheory()
        if data is None:
            self.openFile(filename)
        else:
            self.data = data.astype(np.float)
        self.connectSignals()
        self.updateRp()

    #
    # Set up widgets
    #
    def setupImageTab(self):
        self.ui.imageTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.viewbox = self.ui.imageTab.addViewBox(enableMenu=False,
                                                   enableMouse=False,
                                                   invertY=False,
                                                   lockAspect=True)
        self.image = pg.ImageItem(border=pg.mkPen('k'))
        self.viewbox.addItem(self.image)
        self.overlay = pg.ScatterPlotItem(symbol='+', pen=pg.mkPen('y'))
        self.viewbox.addItem(self.overlay)

    def setupProfileTab(self):
        plot = self.ui.profilePlot
        plot.setXRange(0., self.maxrange)
        plot.showGrid(True, True, 0.2)
        plot.setLabel('bottom', 'r [pixel]')
        plot.setLabel('left', 'b(r)')
        pen = pg.mkPen('k', width=3, style=QtCore.Qt.DashLine)
        plot.addLine(y=1., pen=pen)
        pen = pg.mkPen('k', width=3)
        plot.getAxis('bottom').setPen(pen)
        plot.getAxis('left').setPen(pen)
        pen = pg.mkPen('r', width=3)
        self.theoryProfile = pg.PlotCurveItem(pen=pen)
        plot.addItem(self.theoryProfile)
        pen = pg.mkPen('k', width=3)
        self.dataProfile = pg.PlotCurveItem(pen=pen)
        plot.addItem(self.dataProfile)
        pen = pg.mkPen('k', width=1, style=QtCore.Qt.DashLine)
        self.regionUpper = pg.PlotCurveItem(pen=pen)
        self.regionLower = pg.PlotCurveItem(pen=pen)
        self.dataRegion = pg.FillBetweenItem(
            self.regionUpper, self.regionLower,
            brush=pg.mkBrush(255, 165, 0, 128))
        plot.addItem(self.dataRegion)

    def setupFitTab(self):
        self.ui.fitTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        options = {'enableMenu': False,
                   'enableMouse': False,
                   'invertY': False,
                   'lockAspect': True}
        self.region = pg.ImageItem(pen=pg.mkPen('k'))
        self.fit = pg.ImageItem(pen=pg.mkPen('k'))
        self.residuals = pg.ImageItem(pen=pg.mkPen('k'))
        self.ui.fitTab.addViewBox(**options).addItem(self.region)
        self.ui.fitTab.addViewBox(**options).addItem(self.fit)
        self.ui.fitTab.addViewBox(**options).addItem(self.residuals)

    def setupParameters(self):
        folder = os.path.dirname(pylm.__file__)
        folder += str('/lmtool')
        with open(folder+'/LMTool.json', 'r') as file:
            settings = json.load(file)
        names = ['wavelength', 'magnification', 'n_m',
                 'a_p', 'n_p', 'k_p', 'x_p', 'y_p', 'z_p',
                 'bbox']
        for name in names:
            if hasattr(self.ui, name):
                prop = getattr(self.ui, name)
                setting = settings[name]
                if 'text' in setting:
                    prop.setText(setting['text'])
                if 'suffix' in setting:
                    prop.spinbox.setSuffix(setting['suffix'])
                if 'range' in setting:
                    range = setting['range']
                    prop.setRange(range[0], range[1])
                if 'decimals' in setting:
                    prop.setDecimals(setting['decimals'])
                if 'step' in setting:
                    prop.setSingleStep(setting['step'])
                if 'value' in setting:
                    prop.setValue(setting['value'])
                if 'fixed' in setting:
                    prop.fixed = setting['fixed']
        self.ui.bbox.checkbox.hide()

    def setupTheory(self):
        self.profile_coords = np.arange(self.maxrange)
        self._coordinates = self.profile_coords
        self.feature = Feature()
        self.theory = self.feature.model
        self.theory.coordinates = self.coordinates
        self.theory.instrument.wavelength = self.ui.wavelength.value()
        self.theory.instrument.magnification = self.ui.magnification.value()
        self.theory.instrument.n_m = self.ui.n_m.value()
        self.theory.particle.a_p = self.ui.a_p.value()
        self.theory.particle.n_p = self.ui.n_p.value()
        self.theory.particle.z_p = self.ui.z_p.value()
        self.updateTheoryProfile()

    def connectSignals(self):
        self.ui.actionOpen.triggered.connect(self.openFile)
        self.ui.actionSave_Parameters.triggered.connect(self.saveParameters)
        self.ui.tabs.currentChanged.connect(self.handleTabChanged)
        self.ui.wavelength.valueChanged['double'].connect(
            self.updateInstrument)
        self.ui.magnification.valueChanged['double'].connect(
            self.updateInstrument)
        self.ui.n_m.valueChanged['double'].connect(self.updateInstrument)
        self.ui.a_p.valueChanged['double'].connect(self.updateParticle)
        self.ui.n_p.valueChanged['double'].connect(self.updateParticle)
        self.ui.x_p.valueChanged['double'].connect(self.updateRp)
        self.ui.y_p.valueChanged['double'].connect(self.updateRp)
        self.ui.z_p.valueChanged['double'].connect(self.updateParticle)
        self.ui.bbox.valueChanged['double'].connect(self.updateBBox)
        self.ui.optimizeButton.clicked.connect(self.optimize)

    #
    # Slots for handling user interaction
    #
    @pyqtSlot(int)
    def handleTabChanged(self, tab):
        #self.ui.a_p.fixed = (tab != 1)
        #self.ui.n_p.fixed = (tab != 1)
        #self.ui.k_p.fixed = (tab != 1)
        #self.ui.z_p.fixed = (tab != 1)
        #self.ui.x_p.fixed = (tab != 0)
        #self.ui.y_p.fixed = (tab != 0)
        if (tab == 1):
            self.updateDataProfile()
        if (tab == 2):
            self.updateFit()

    @pyqtSlot(float)
    def updateInstrument(self, count):
        self.theory.instrument.wavelength = self.ui.wavelength.value()
        self.theory.instrument.magnification = self.ui.magnification.value()
        self.theory.instrument.n_m = self.ui.n_m.value()
        self.updateTheoryProfile()
        if self.ui.tabs.currentIndex() == 2:
            self.updateFit()

    @pyqtSlot(float)
    def updateParticle(self, count):
        self.theory.particle.a_p = self.ui.a_p.value()
        self.theory.particle.n_p = self.ui.n_p.value()
        self.theory.particle.z_p = self.ui.z_p.value()
        self.updateTheoryProfile()
        if self.ui.tabs.currentIndex() == 2:
            self.updateFit()

    @pyqtSlot(float)
    def updateHologram(self, count):
        self.updateTheoryProfile()
        if self.ui.tabs.currentIndex() == 2:
            self.updateFit()

    @pyqtSlot(float)
    def updateRp(self, r_p=0):
        x_p = [self.ui.x_p.value()]
        y_p = [self.ui.y_p.value()]
        self.overlay.setData(x_p, y_p)
        self.updateDataProfile()
        if self.ui.tabs.currentIndex() == 2:
            self.updateFit()

    @pyqtSlot(float)
    def updateBBox(self, count):
        self.maxrange = int(self.ui.bbox.value() / 2)
        self.profile_coords = np.arange(self.maxrange)
        self.ui.profilePlot.setXRange(0., self.maxrange)
        self.updateDataProfile()
        self.updateTheoryProfile()
        self.updateFit()

    @pyqtSlot()
    def optimize(self):
        logger.info("Starting optimization...")
        self.updateFit()
        method = 'lm' if self.ui.LMButton.isChecked() else 'amoeba-lm'
        for prop in self.feature.optimizer.params:
            if hasattr(self.ui, prop):
                propUi = getattr(self.ui, prop)
                self.feature.optimizer.vary[prop] = not propUi.fixed
        result = self.feature.optimize(method=method)
        self.updateParameterUi()
        self.updateFit()
        self.updateDataProfile()
        self.updateTheoryProfile()
        logger.info("Finished!\n{}".format(str(result)))

    def updateParameterUi(self):
        # Disconnect
        self.ui.wavelength.valueChanged['double'].disconnect(
            self.updateInstrument)
        self.ui.magnification.valueChanged['double'].disconnect(
            self.updateInstrument)
        self.ui.n_m.valueChanged['double'].disconnect(self.updateInstrument)
        self.ui.a_p.valueChanged['double'].disconnect(self.updateParticle)
        self.ui.n_p.valueChanged['double'].disconnect(self.updateParticle)
        self.ui.z_p.valueChanged['double'].disconnect(self.updateParticle)
        # Update
        particle, instrument = (self.theory.particle,
                                self.theory.instrument)
        for p in self.feature.optimizer.params:
            if hasattr(self.ui, p):
                attrUi = getattr(self.ui, p)
                if p in particle.properties:
                    attrUi.setValue(getattr(particle, p))
                elif p in instrument.properties:
                    attrUi.setValue(getattr(instrument, p))
                else:
                    attrUi.setValue(getattr(self.theory, p))
        # Reconnect
        self.ui.wavelength.valueChanged['double'].connect(
            self.updateInstrument)
        self.ui.magnification.valueChanged['double'].connect(
            self.updateInstrument)
        self.ui.n_m.valueChanged['double'].connect(self.updateInstrument)
        self.ui.a_p.valueChanged['double'].connect(self.updateParticle)
        self.ui.n_p.valueChanged['double'].connect(self.updateParticle)
        self.ui.z_p.valueChanged['double'].connect(self.updateParticle)

    @pyqtSlot()
    def openFile(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open Hologram', '', 'Images (*.png)')
        data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if data is None:
            return
        self.data = data.astype(float)

    @pyqtSlot()
    def openBackground(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open Background', '', 'Images (*.png)')
        self.background = cv2.imread(
            filename, cv2.IMREAD_GRAYSCALE).astype(float)

    @pyqtSlot()
    def saveParameters(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Parameters', '', 'JSON (*.json)')
        names = ['x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p',
                 'magnification', 'wavelength', 'n_m']
        parameters = {name: getattr(self.ui, name).value()
                      for name in names}
        try:
            with open(filename, 'w') as file:
                json.dump(parameters, file, indent=4, sort_keys=True)
        except IOError:
            print('error')

    #
    # Routines to update plots
    #
    def updateDataProfile(self):
        self.coordinates = self.profile_coords
        center = (self.ui.x_p.value(), self.ui.y_p.value())
        avg, std = azistd(self.data, center)
        self.dataProfile.setData(avg)
        self.regionUpper.setData(avg + std)
        self.regionLower.setData(avg - std)

    def updateTheoryProfile(self):
        self.coordinates = self.profile_coords
        self.theory.particle.x_p, self.theory.particle.y_p = (0, 0)
        xsmooth = np.linspace(0, self.maxrange - 1, 300)
        y = self.theory.hologram()
        t, c, k = splrep(self.coordinates, y)
        spline = BSpline(t, c, k)
        ysmooth = spline(xsmooth)
        self.theoryProfile.setData(xsmooth, ysmooth)

    def updateFit(self):
        dim = self.maxrange
        x_p = self.ui.x_p.value()
        y_p = self.ui.y_p.value()
        h, w = self.data.shape
        x0 = int(np.clip(x_p - dim, 0, w - 2))
        y0 = int(np.clip(y_p - dim, 0, h - 2))
        x1 = int(np.clip(x_p + dim, x0 + 1, w - 1))
        y1 = int(np.clip(y_p + dim, y0 + 1, h - 1))
        img = self.data[y0:y1, x0:x1]
        xcoords = self.data_coords[0].reshape(self.data.shape)
        xcoords = xcoords[y0:y1, x0:x1].flatten()
        ycoords = self.data_coords[1].reshape(self.data.shape)
        ycoords = ycoords[y0:y1, x0:x1].flatten()
        self.coordinates = np.stack((xcoords, ycoords))
        self.theory.particle.x_p = x_p
        self.theory.particle.y_p = y_p
        hol = self.theory.hologram().reshape(img.shape)
        self.feature.data = img
        self.region.setImage(img)
        self.fit.setImage(hol)
        self.residuals.setImage(self.feature.residuals())

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data / self.background
        self._data /= np.mean(self._data)
        self.image.setImage(self._data)
        self.data_coords = coordinates(data.shape)
        self.ui.x_p.setRange(0, data.shape[1]-1)
        self.ui.y_p.setRange(0, data.shape[0]-1)
        self.ui.bbox.setRange(0, min(data.shape[0]-1, data.shape[1]-1))
        self.updateDataProfile()

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates
        self.theory.coordinates = coordinates


def main():
    import sys
    import argparse

    fn = '../docs/tutorials/crop.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=fn,
                        nargs='?', action='store')
    parser.add_argument('-b', '--background', metavar='filename',
                        dest='background', type=str, default=None,
                        action='store',
                        help='name of background image file')
    parser.add_argument('-n', '--normalization', metavar='value',
                        dest='normalization', type=float, default=1.,
                        action='store',
                        help='Ignored if background is supplied.')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    app = QtWidgets.QApplication(qt_args)
    lmtool = LMTool(args.filename, args.background, args.normalization)
    lmtool.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
