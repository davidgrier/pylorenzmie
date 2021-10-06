# /usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtCore import (pyqtProperty, pyqtSlot, Qt)
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QApplication)
from PyQt5 import uic
import pyqtgraph as pg
from matplotlib import cm
import os
import pandas as pd
import cv2
import numpy as np
import logging

try:
    import cupy
except ImportError:
    pass
from pylorenzmie.theory import (Sphere, Instrument, ZernikeCoefficients,
                                LMHologram)
from pylorenzmie.analysis import Frame
from pylorenzmie.utilities import azistd


logger = logging.getLogger('LMTool')
logger.setLevel(logging.WARNING)


class LMTool(QMainWindow):

    def __init__(self,
                 data=None,
                 aberrations=False,
                 background=None,
                 percentpix=0.3):
        super(LMTool, self).__init__()

        self.setupPyQtGraph()
        uic.loadUi('LMTool.ui', self)
        self.setupControls(aberrations)
        self.setupTabs()
        self.setupTheory(percentpix)
        self.autonormalize = True         # FIXME: should be a UI option
        self.setupData(data, background)
        self.connectSignals()
        self.loadDefaults()

    @pyqtProperty(int)
    def maxrange(self):
        return self.bbox.value() // 2

    @pyqtProperty(list)
    def parameters(self):
        p = ['wavelength', 'magnification', 'n_m',
             'a_p', 'n_p', 'k_p',
             'x_p', 'y_p', 'z_p',
             'bbox',
             'pupil', 'xtilt', 'ytilt',
             'defocus', 'xastigmatism', 'yastigmatism',
             'xcoma', 'ycoma', 'spherical']
        return p

    def setupPyQtGraph(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')

    #
    # Set up widgets
    #

    def setupControls(self, aberrations):
        self.bbox.checkbox.hide()
        self.x_p.setStep(1.)
        self.y_p.setStep(1.)
        self.z_p.setStep(1.)
        if not aberrations:
            self.aberrationFrame.hide()

    def setupTabs(self):
        options = dict(enableMenu=False,
                       enableMouse=False,
                       invertY=False,
                       lockAspect=True)
        self.setupImageTab(options)
        self.setupProfileTab()
        self.setupFitTab(options)

    def setupImageTab(self, options):
        self.imageTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.image = pg.ImageItem(border=pg.mkPen('k'))
        self.imageTab.addViewBox(**options).addItem(self.image)
        pen = pg.mkPen('w', width=3)
        hoverPen = pg.mkPen('y', width=3)
        self.roi = pg.CircleROI([100, 100], radius=100., parent=self.image)
        self.roi.setPen(pen)
        self.roi.hoverPen = hoverPen
        self.roi.removeHandle(0)

    def setupProfileTab(self):
        plot = self.profilePlot
        plot.setXRange(0., self.maxrange)
        plot.showGrid(True, True, 0.2)
        plot.setLabel('bottom', 'r [pixel]')
        plot.setLabel('left', 'b(r)')
        pen = pg.mkPen('k', width=3, style=Qt.DashLine)
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
        pen = pg.mkPen('k', width=1, style=Qt.DashLine)
        self.regionUpper = pg.PlotCurveItem(pen=pen)
        self.regionLower = pg.PlotCurveItem(pen=pen)
        self.dataRegion = pg.FillBetweenItem(
            self.regionUpper, self.regionLower,
            brush=pg.mkBrush(255, 165, 0, 128))
        plot.addItem(self.dataRegion)

    def setupFitTab(self, options):
        self.fitTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        pen = pg.mkPen('k')
        self.region = pg.ImageItem(pen=pen)
        self.fit = pg.ImageItem(pen=pen)
        self.residuals = pg.ImageItem(pen=pen)
        self.fitTab.addViewBox(**options).addItem(self.region)
        self.fitTab.addViewBox(**options).addItem(self.fit)
        self.fitTab.addViewBox(**options).addItem(self.residuals)
        map = cm.get_cmap('bwr')
        map._init()
        lut = (map._lut * 255).view(np.ndarray)
        self.residuals.setLookupTable(lut)

    def setupTheory(self, percentpix):
        # Profile and Frame share components
        self.particle = Sphere()
        self.instrument = Instrument()
        self.coefficients = ZernikeCoefficients()
        components = dict(particle=self.particle,
                          instrument=self.instrument,
                          coefficients=self.coefficients)
        # Theory for radial profile
        self.theory = LMHologram(**components)
        self.theory.coordinates = np.arange(self.maxrange)
        # Theory for image
        self.frame = Frame(**components, percentpix=percentpix)

    #
    # Routines for loading data
    #

    def setupData(self, data, background):
        if type(data) is str:
            self.data = self.readData(data)
        else:
            self.data = data.astype(float)
        if self.autonormalize:
            return
        if type(background) is str:
            self.background = self.readData(background)
        elif type(background) is int:
            self.background = background

    def readData(self, filename=None):
        if filename is None:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Hologram', '', 'Images (*.png)')
        if filename is None:
            return None
        return cv2.imread(filename, 0).astype(float)

    @pyqtSlot()
    def openHologram(self):
        self.data = self.readData()

    @pyqtSlot()
    def openBackground(self):
        self.background = self.readData()

    @pyqtProperty(np.ndarray)
    def data(self):
        return self.frame.data

    @data.setter
    def data(self, data):
        if data is None:
            return
        if self.autonormalize:
            self.frame.data = data / np.median(data)
        else:
            self.frame.data = data / self.background
        self.image.setImage(self.frame.data)
        self.x_p.setRange((0, data.shape[1]-1))
        self.y_p.setRange((0, data.shape[0]-1))
        self.bbox.setRange((0, min(data.shape[0]-1, data.shape[1]-1)))
        self.updateDataProfile()

    def loadDefaults(self):
        basedir = os.path.dirname(os.path.abspath(__file__))
        settings = pd.read_json(os.path.join(basedir, 'LMTool.json'))
        for parameter in self.parameters:
            widget = getattr(self, parameter)
            for setting, value in settings[parameter].items():
                if (value is np.nan):
                    continue
                logger.debug(f'{parameter}: {setting}: {value}')
                setter_name = f'set{setting.capitalize()}'
                setter = getattr(widget, setter_name)
                setter(value)
    #
    # Slots for handling user interaction
    #

    def connectSignals(self):
        self.actionOpen.triggered.connect(self.openHologram)
        self.actionSave_Parameters.triggered.connect(self.saveParameters)
        self.tabs.currentChanged.connect(self.handleTabChanged)
        for parameter in self.parameters:
            widget = getattr(self, parameter)
            widget.valueChanged['double'].connect(self.updateParameter)
        self.optimizeButton.clicked.connect(self.optimize)
        self.roi.sigRegionChanged.connect(self.handleROIChanged)

    @pyqtSlot(int)
    def handleTabChanged(self, tab):
        if (tab == 1):
            self.updateDataProfile()
        if (tab == 2):
            self.updateFit()

    @pyqtSlot(object)
    def handleROIChanged(self, roi):
        x0, y0 = roi.pos()
        dim, _ = roi.size()
        self.x_p.setValue(x0 + dim/2)
        self.y_p.setValue(y0 + dim/2)
        self.bbox.setValue(dim)

    @pyqtSlot(float)
    def updateParameter(self, value):
        parameter = self.sender().objectName()
        if parameter == 'bbox':
            self.profilePlot.setXRange(0., self.maxrange)
        for subsys in [self.instrument, self.particle, self.coefficients]:
            if hasattr(subsys, parameter):
                setattr(subsys, parameter, value)
                break
        if parameter in ['x_p', 'y_p', 'bbox']:
            self.updateROI()
        self.updatePlots()

    #
    # Routines to update plots
    #
    def updatePlots(self):
        self.updateTheoryProfile()
        self.updateDataProfile()
        if self.tabs.currentIndex() == 2:
            self.updateFit()

    def updateDataProfile(self):
        avg, std = azistd(self.frame.data, self.particle.r_p[0:2])
        self.dataProfile.setData(avg)
        self.regionUpper.setData(avg + std)
        self.regionLower.setData(avg - std)

    def updateTheoryProfile(self):
        x = np.arange(self.maxrange, dtype=float)
        y = np.full_like(x, self.particle.y_p)
        coordinates = np.stack((x + self.particle.x_p, y))
        self.theory.coordinates = coordinates
        profile = self.theory.hologram()
        self.theoryProfile.setData(x, profile)

    def updateFit(self):
        feature = self.frame.features[0]
        self.region.setImage(feature.data)
        self.fit.setImage(feature.hologram())
        self.residuals.setImage(feature.residuals())

    def updateROI(self):
        dim = self.maxrange
        x_p = self.particle.x_p
        y_p = self.particle.y_p
        h, w = self.frame.shape
        x0 = int(np.clip(x_p - dim, 0, w - 2))
        y0 = int(np.clip(y_p - dim, 0, h - 2))
        x1 = int(np.clip(x_p + dim, x0 + 1, w - 1))
        y1 = int(np.clip(y_p + dim, y0 + 1, h - 1))
        self.roi.setSize((2*dim+1, 2*dim+1), (0.5, 0.5), update=False)
        self.roi.setPos(x0, y0, update=False)
        self.frame.bboxes = [((x0, y0), x1-x0, y1-y0)]

    #
    # Routines associated with fitting
    #
    @pyqtSlot()
    def optimize(self):
        self.statusBar().showMessage('Optimizing parameters...')
        logger.info('Starting optimization...')
        feature = self.frame.features[0]
        feature.particle = self.particle
        if self.LMButton.isChecked():
            feature.optimizer.settings['method'] = 'lm'
            feature.optimizer.settings['loss'] = 'linear'
        else:
            feature.optimizer.settings['method'] = 'dogbox'
            feature.optimizer.settings['loss'] = 'cauchy'
        fixed = [p for p in self.parameters if getattr(self, p).fixed]
        feature.optimizer.fixed = fixed
        result = feature.optimize()
        self.updateUiValues()
        self.updatePlots()
        logger.info(f'Finished!\n{str(result)}')
        self.statusBar().showMessage('Optimization complete')

    def updateUiValues(self):
        '''Update Ui with parameters from particle and instrument'''
        for parameter in self.parameters:
            widget = getattr(self, parameter)
            widget.blockSignals(True)
            if hasattr(self.particle, parameter):
                widget.setValue(getattr(self.particle, parameter))
            elif hasattr(self.instrument, parameter):
                widget.setValue(getattr(self.instrument, parameter))
            elif hasattr(self.coefficients, parameter):
                widget.setValue(getattr(self.coefficients, parameter))
            widget.blockSignals(False)

    @pyqtSlot()
    def saveParameters(self, filename=None):
        if filename is None:
            get = QFileDialog.getSaveFileName
            filename, _ = get(self, 'Save Parameters', '', 'JSON (*.json)')
        params = {p: getattr(self, p).value() for p in self.parameters}
        parameters = pd.DataFrame(params)
        try:
            parameters.to_json(filename, orient='index', indent=4)
        except IOError as ex:
            logger.debug(f'Could not save settings: {ex}')


def main():
    import sys
    import argparse

    basedir = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(basedir, '..', 'docs', 'tutorials', 'crop.png')

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=fn,
                        nargs='?', action='store')
    parser.add_argument('-a', '--aberrations', dest='aberrations',
                        action='store_true',
                        help='adjust aberration values')
    parser.set_defaults(aberration=False)
    parser.add_argument('-b', '--background', dest='background',
                        default=None, action='store',
                        help='background value or file name')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    background = args.background
    if background is not None and background.isdigit():
        background = int(background)

    app = QApplication(qt_args)
    lmtool = LMTool(args.filename, args.aberrations, background)
    lmtool.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
