# /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import json
import cv2
import numpy as np
import pyqtgraph as pg

from pylorenzmie.theory import (Sphere, Instrument, LMHologram)
from pylorenzmie.analysis import Frame
from pylorenzmie.utilities import (coordinates, azistd)

from LMTool_Ui import Ui_MainWindow
from PyQt5.QtCore import (pyqtProperty, pyqtSlot)
from PyQt5 import (QtWidgets, QtCore)

logger = logging.getLogger('LMTool')
logger.setLevel(logging.INFO)


class LMTool(QtWidgets.QMainWindow):

    def __init__(self,
                 data=None,
                 background=None,
                 percentpix=0.3):
        super(LMTool, self).__init__()

        self.setupPyQtGraph()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.bbox.checkbox.hide()
        self.setupTabs()
        self.setupTheory(percentpix)
        self.autonormalize = True         # FIXME: should be a UI option
        self.setupData(data, background)
        self.connectSignals()
        self.loadDefaults()

    @pyqtProperty(int)
    def maxrange(self):
        return self.ui.bbox.value() // 2

    @pyqtProperty(list)
    def parameters(self):
        p = ['wavelength', 'magnification', 'n_m',
             'a_p', 'n_p', 'k_p',
             'x_p', 'y_p', 'z_p', 'bbox']
        return p
    
    def setupPyQtGraph(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')

    
    #
    # Set up widgets
    #
    def setupTabs(self):
        options = dict(enableMenu=False,
                       enableMouse=False,
                       invertY=False,
                       lockAspect=True)
        self.setupImageTab(options)
        self.setupProfileTab()
        self.setupFitTab(options)
        
    def setupImageTab(self, options):
        self.ui.imageTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.image = pg.ImageItem(border=pg.mkPen('k'))
        self.ui.imageTab.addViewBox(**options).addItem(self.image)
        pen = pg.mkPen('w', width=3)
        hoverPen = pg.mkPen('y', width=3)
        self.roi = pg.CircleROI([100,100], radius=100., parent=self.image)
        self.roi.setPen(pen)
        self.roi.hoverPen = hoverPen
        self.roi.removeHandle(0)

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

    def setupFitTab(self, options):
        self.ui.fitTab.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.region = pg.ImageItem(pen=pg.mkPen('k'))
        self.fit = pg.ImageItem(pen=pg.mkPen('k'))
        self.residuals = pg.ImageItem(pen=pg.mkPen('k'))
        self.ui.fitTab.addViewBox(**options).addItem(self.region)
        self.ui.fitTab.addViewBox(**options).addItem(self.fit)
        self.ui.fitTab.addViewBox(**options).addItem(self.residuals)
    
    def setupTheory(self, percentpix):
        # Profile and Frame use the same particle and instrument
        self.particle = Sphere()
        self.instrument = Instrument()
        # Theory for radial profile
        self.theory = LMHologram(particle=self.particle,
                                 instrument=self.instrument)
        self.theory.coordinates = np.arange(self.maxrange)
        # Theory for image
        self.frame = Frame(particle=self.particle,
                           instrument=self.instrument,
                           percentpix=percentpix)

    #
    # Routines for loading data
    #
    def setupData(self, data, background):
        if type(data) is str:
            self.openHologram(data)
        else:
            self.data = data.astype(float)
        if not self.autonormalize:
            if type(background) is str:
                self.openBackground(background)
            elif type(background) is int:
                self.frame.background = background

    @pyqtSlot()
    def openHologram(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open Hologram', '', 'Images (*.png)')
        data = cv2.imread(filename, 0).astype(float)
        if data is None:
            return
        self.data = data
        
    @pyqtSlot()
    def openBackground(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open Background', '', 'Images (*.png)')
        background = cv2.imread(filename, 0).astype(float)
        if background is None:
            return
        self.frame.background = background

    @pyqtProperty(object)
    def data(self):
        return self.frame.data

    @data.setter
    def data(self, data):
        self.frame.data = data / np.mean(data)
        self.image.setImage(self.frame.data)
        self.ui.x_p.setRange((0, data.shape[1]-1))
        self.ui.y_p.setRange((0, data.shape[0]-1))
        self.ui.bbox.setRange((0, min(data.shape[0]-1, data.shape[1]-1)))
        self.updateDataProfile()

    def loadDefaults(self):
        basedir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(basedir, 'LMTool.json'), 'r') as file:
            settings = json.load(file)
        for parameter in self.parameters:
            widget = getattr(self.ui, parameter)
            for setting, value in settings[parameter].items():
                logger.debug('{}: {}: {}'.format(parameter, setting, value))
                setter_name = 'set{}'.format(setting.capitalize())
                setter = getattr(widget, setter_name)
                setter(value)

    #
    # Slots for handling user interaction
    #
    def connectSignals(self):
        self.ui.actionOpen.triggered.connect(self.openHologram)
        self.ui.actionSave_Parameters.triggered.connect(self.saveParameters)
        self.ui.tabs.currentChanged.connect(self.handleTabChanged)
        for parameter in self.parameters:
            widget = getattr(self.ui, parameter)
            widget.valueChanged['double'].connect(self.updateParameter)
        self.ui.optimizeButton.clicked.connect(self.optimize)
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
        self.ui.bbox.setValue(dim)
        self.ui.x_p.setValue(x0 + dim/2)
        self.ui.y_p.setValue(y0 + dim/2)

    @pyqtSlot(float)
    def updateParameter(self, value):
        parameter = self.sender().parent().objectName()
        if parameter == 'bbox':
            self.ui.profilePlot.setXRange(0., self.maxrange)
        elif hasattr(self.instrument, parameter):
            setattr(self.instrument, parameter, value)
        elif hasattr(self.particle, parameter):
            setattr(self.particle, parameter, value)
        if parameter in ['x_p', 'y_p', 'bbox']:
            self.updateROI()
        self.updatePlots()

    #
    # Routines to update plots
    #
    def updatePlots(self):
        self.updateTheoryProfile()
        self.updateDataProfile()
        if self.ui.tabs.currentIndex() == 2:
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
        logger.info('Starting optimization...')
        feature = self.frame.features[0]
        feature.particle = self.particle
        if self.ui.LMButton.isChecked():
            feature.optimizer.method = 'lm'
        else:
            feature.optimizer.method = 'amoeba-lm'
        fixed = [p for p in self.parameters if getattr(self.ui, p).fixed]
        feature.optimizer.fixed = fixed
        print('Before: ', self.particle)
        result = feature.optimize()
        print('After: ', self.particle)
        self.updateUiValues()
        self.updatePlots()
        logger.info("Finished!\n{}".format(str(result)))

    def updateUiValues(self):
        '''Update Ui with parameters from particle and instrument'''
        for parameter in self.parameters:
            widget = getattr(self.ui, parameter)
            widget.blockSignals(True)
            if hasattr(self.particle, parameter):
                widget.setValue(getattr(self.particle, parameter))
            elif hasattr(self.instrument, parameter):
                widget.setValue(getattr(self.instrument, parameter))
            widget.blockSignals(False)

    @pyqtSlot()
    def saveParameters(self, filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Parameters', '', 'JSON (*.json)')
        parameters = {name: getattr(self.ui, name).value()
                      for name in self.parameters}
        try:
            with open(filename, 'w') as file:
                json.dump(parameters, file, indent=4, sort_keys=True)
        except IOError:
            print('error')

 
def main():
    import sys
    import argparse

    basedir = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(basedir, '..', 'docs', 'tutorials', 'crop.png')

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=fn,
                        nargs='?', action='store')
    parser.add_argument('-b', '--background', dest='background',
                        default=None, action='store',
                        help='background value or file name')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    background = args.background
    if background is not None and background.isdigit():
        background = int(background)

    app = QtWidgets.QApplication(qt_args)
    lmtool = LMTool(args.filename, background)
    lmtool.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
