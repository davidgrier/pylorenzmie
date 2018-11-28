# /usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from LMTool_Ui import Ui_MainWindow
import pyqtgraph as pg
import numpy as np
import cv2
import json
from scipy.interpolate import spline
from pylorenzmie.theory.LMHologram import LMHologram


def aziavg(data, center):
    x_p, y_p = center
    y, x = np.indices((data.shape))
    d = data.ravel()
    r = np.hypot(x - x_p, y - y_p).astype(np.int).ravel()
    nr = np.bincount(r)
    ravg = np.bincount(r, d) / nr
    avg = ravg[r]
    rstd = np.sqrt(np.bincount(r, (d - avg)**2) / nr)
    return ravg, rstd


class LMTool(QtWidgets.QMainWindow):

    def __init__(self, filename=None):
        super(LMTool, self).__init__()
        self.maxrange = 100
        self.coordinates = np.arange(self.maxrange)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupImageTab()
        self.setupProfileTab()
        self.setupFitTab()
        self.setupParameters()
        self.setupTheory()
        self.connectSignals()
        self.openFile(filename)
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
        plot = self.ui.plot
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
        with open('LMTool.json', 'r') as file:
            settings = json.load(file)
        names = ['wavelength', 'magnification', 'n_m',
                 'a_p', 'n_p', 'k_p', 'x_p', 'y_p', 'z_p']
        for name in names:
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

    def setupTheory(self):
        self.theory = LMHologram(coordinates=self.coordinates)
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

    #
    # Slots for handling user interaction
    #
    @pyqtSlot(int)
    def handleTabChanged(self, tab):
        self.ui.a_p.fixed = (tab != 1)
        self.ui.n_p.fixed = (tab != 1)
        self.ui.k_p.fixed = (tab != 1)
        self.ui.z_p.fixed = (tab != 1)
        self.ui.x_p.fixed = (tab != 0)
        self.ui.y_p.fixed = (tab != 0)
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

    @pyqtSlot(float)
    def updateParticle(self, count):
        self.theory.particle.a_p = self.ui.a_p.value()
        self.theory.particle.n_p = self.ui.n_p.value()
        self.theory.particle.z_p = self.ui.z_p.value()
        self.updateTheoryProfile()

    @pyqtSlot(float)
    def updateRp(self, r_p=0):
        x_p = [self.ui.x_p.value()]
        y_p = [self.ui.y_p.value()]
        self.overlay.setData(x_p, y_p)

    @pyqtSlot()
    def openFile(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open Hologram', '', 'Images (*.png)')
        self.data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

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
        center = (self.ui.x_p.value(), self.ui.y_p.value())
        avg, std = aziavg(self.data, center)
        self.dataProfile.setData(avg)
        self.regionUpper.setData(avg + std)
        self.regionLower.setData(avg - std)

    def updateTheoryProfile(self):
        xsmooth = np.linspace(0, self.maxrange - 1, 300)
        y = self.theory.hologram()
        ysmooth = spline(self.coordinates, y, xsmooth)
        self.theoryProfile.setData(xsmooth, ysmooth)

    def updateFit(self):
        dim = 100
        h, w = self.data.shape
        x_p = self.ui.x_p.value()
        y_p = self.ui.y_p.value()
        x0 = int(np.clip(x_p - dim, 0, w - 2))
        y0 = int(np.clip(y_p - dim, 0, h - 2))
        x1 = int(np.clip(x_p + dim, x0 + 1, w - 1))
        y1 = int(np.clip(y_p + dim, y0 + 1, h - 1))
        img = self.data[y0:y1, x0:x1]
        self.region.setImage(img)
        self.fit.setImage(img)
        self.residuals.setImage(img)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        data = data.astype(float)
        med = np.median(data)
        if med > 2:
            data /= med
        self._data = data
        self.image.setImage(self._data)
        self.ui.x_p.setRange(0, data.shape[1])
        self.ui.y_p.setRange(0, data.shape[0])
        self.updateDataProfile()


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default='sample.png',
                        nargs='?', action='store')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    app = QtWidgets.QApplication(qt_args)
    lmtool = LMTool(args.filename)
    lmtool.show()
    sys.exit(app.exec_())
