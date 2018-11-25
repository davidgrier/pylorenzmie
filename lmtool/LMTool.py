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

    def __init__(self):
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
        self.openFile('sample.png')
        self.updateRp()

    #####
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
        self.region = pg.ImageItem()
        regionbox = pg.ViewBox(**options)
        regionbox.addItem(self.region)
        self.fit = pg.ImageItem()
        fitbox = pg.ViewBox(**options)
        regionbox.addItem(self.fit)
        self.residuals = pg.ImageItem()
        residualbox = pg.ViewBox(**options)
        residualbox.addItem(self.residuals)
        self.ui.fitTab.addItem(regionbox, 0, 0)
        self.ui.fitTab.addItem(fitbox, 0, 1)
        self.ui.fitTab.addItem(residualbox, 0, 2)

    def setupParameters(self):
        self.ui.wavelength.setText('wavelength')
        self.ui.wavelength.spinbox.setSuffix(' μm')
        self.ui.wavelength.setRange(0.405, 1.070)
        self.ui.wavelength.setValue(0.447)
        self.ui.wavelength.fixed = True

        self.ui.magnification.setText('magnification')
        self.ui.magnification.spinbox.setSuffix(' μm/pixel')
        self.ui.magnification.setRange(0.046, 0.135)
        self.ui.magnification.setValue(0.135)
        self.ui.magnification.fixed = True

        self.ui.n_m.setText('n<sub>m</sub>')
        self.ui.n_m.setRange(1.330, 1.342)
        self.ui.n_m.setValue(1.340)
        self.ui.n_m.fixed = True

        self.ui.a_p.setText('a<sub>p</sub>')
        self.ui.a_p.spinbox.setSuffix(' μm')
        self.ui.a_p.setRange(0.3, 10.)
        self.ui.a_p.setValue(0.75)

        self.ui.n_p.setText('n<sub>p</sub>')
        self.ui.n_p.setRange(1.345, 2.5)
        self.ui.n_p.setValue(1.45)

        self.ui.k_p.setText('k<sub>p</sub>')
        self.ui.k_p.setRange(0., 10.)
        self.ui.k_p.setValue(0.)
        self.ui.k_p.fixed = True

        self.ui.x_p.setText('x<sub>p</sub>')
        self.ui.x_p.spinbox.setSuffix(' pixel')
        self.ui.x_p.setDecimals(2)
        self.ui.x_p.setValue(100)

        self.ui.y_p.setText('y<sub>p</sub>')
        self.ui.y_p.spinbox.setSuffix(' pixel')
        self.ui.y_p.setDecimals(2)
        self.ui.y_p.setValue(100)

        self.ui.z_p.setText('z<sub>p</sub>')
        self.ui.z_p.spinbox.setSuffix(' pixel')
        self.ui.z_p.setDecimals(2)
        self.ui.z_p.setRange(20, 600)
        self.ui.z_p.setValue(100)

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

    #####
    #
    # Slots for handling user interaction
    #
    @pyqtSlot(int)
    def handleTabChanged(self, tab):
        self.ui.a_p.fixed = (tab == 0)
        self.ui.n_p.fixed = (tab == 0)
        self.ui.k_p.fixed = (tab == 0)
        self.ui.z_p.fixed = (tab == 0)
        self.ui.x_p.fixed = (tab == 1)
        self.ui.y_p.fixed = (tab == 1)
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

    #####
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
        xsmooth = np.linspace(0, self.maxrange-1, 300)
        y = self.theory.hologram()
        ysmooth = spline(self.coordinates, y, xsmooth)
        self.theoryProfile.setData(xsmooth, ysmooth)

    def updateFit(self):
        dim = 100
        h, w = self.data.shape
        x_p = self.ui.x_p.value()
        y_p = self.ui.y_p.value()
        x0 = int(np.clip(x_p - dim, 0, w - 1))
        y0 = int(np.clip(y_p - dim, 0, h - 1))
        x1 = int(np.clip(x_p + dim, 0, w - 1))
        y1 = int(np.clip(y_p + dim, 0, h - 1))
        print(x0, x1, y0, y1)
        img = self.data[x0:x1, y0:y1]
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

    app = QtWidgets.QApplication(sys.argv)
    lmtool = LMTool()
    lmtool.show()
    sys.exit(app.exec_())
