#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import (pyqtProperty, pyqtSlot, QRectF,
                                  QSignalBlocker)
from pyqtgraph.Qt.QtWidgets import (QMainWindow, QFileDialog, QProgressBar)

from pylorenzmie.analysis import Estimator
from pylorenzmie.lib import (Azimuthal, LMObject)
from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.utilities import Normalizer

_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)

# TODO: interactive residuals (currently only updates after fits)
# TODO: support for cuda-accelerated kernels
# TODO: toml config file
# TODO: support for gamma correction


class LMTool(QMainWindow):
    '''Main window for the LMTool hologram-fitting application.

    Parameters
    ----------
    controls : type[LMWidget]
        LMWidget subclass (not an instance) providing parameter controls.
    filename : str, optional
        Path to a hologram image to load on startup.
    normalizer : Normalizer, optional
        Background normalization strategy. Defaults to median-filter
        normalization with a 51-pixel kernel.
    '''

    uiFile = 'LMTool.ui'

    def __init__(self,
                 controls: type[LMWidget],
                 filename: str | None = None,
                 normalizer: Normalizer | None = None):
        super().__init__()
        uic.loadUi(_DIR / self.uiFile, self)
        self.normalizer = normalizer if normalizer is not None else Normalizer()
        self._raw = None
        self._data = None
        self._pre_estimate = None
        self._setupTheory(controls())
        self._setupStatusBar()
        self.readHologram(filename)
        self._connectSignals()

    def _setupTheory(self, controls: LMWidget) -> None:
        layout = self.controls.parent().layout()
        layout.replaceWidget(self.controls, controls)
        self.controls.close()
        self.controls = controls
        self.profileWidget.model = self.controls.cls()
        self.profileWidget.properties = self.controls.properties
        self.profileWidget.radius = self.imageWidget.radius
        self.fitWidget.model = controls.model
        self.fitWidget.properties = self.controls.properties
        self.optimizerWidget.settings = self.fitWidget.optimizer.settings

    def _setupStatusBar(self) -> None:
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(150)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)

    def _connectSignals(self) -> None:
        self.imageWidget.roiChanged.connect(self._handleROIChanged)
        self.imageWidget.radiusChanged.connect(self._handleRadiusChanged)
        self.controls.propertyChanged.connect(self._handlePropertyChanged)
        self.actionOpen.triggered.connect(self.readHologram)
        self.actionOpenBackground.triggered.connect(self.readBackground)
        self.actionEstimate.triggered.connect(self.estimate)
        self.actionUndoEstimate.triggered.connect(self.undoEstimate)
        self.actionSaveParameters.triggered.connect(self.saveParameters)
        self.saveResult.triggered.connect(self.fitWidget.saveResult)
        self.saveResultAs.triggered.connect(self.fitWidget.saveResultAs)
        self.actionRobust.toggled.connect(self.setRobust)
        self.actionOptimize.triggered.connect(self.optimize)
        connect = self.optimizerWidget.settingChanged.connect
        connect(self.fitWidget.setSetting)
        self.fitWidget.optimizationStarted.connect(self._onOptimizationStarted)
        self.fitWidget.optimizationFinished.connect(self._onOptimizationFinished)
        self.fitWidget.optimizationError.connect(self._onOptimizationError)

    @pyqtProperty(np.ndarray)
    def data(self) -> NDArray[float]:
        return self._data

    @data.setter
    def data(self, data: NDArray[float]) -> None:
        self._raw = data
        self._data = self.normalizer(data)
        self.coordinates = LMObject.meshgrid(data.shape, flatten=False)
        self.controls.x_p.setRange((0, data.shape[1]-1))
        self.controls.y_p.setRange((0, data.shape[0]-1))
        self.imageWidget.data = self._data
        self._updateProfile()
        self.fitWidget.setData(*self.crop())

    @pyqtSlot()
    def readHologram(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Hologram', '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        self.fitWidget.datafile = filename
        self.data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)

    @pyqtSlot()
    def readBackground(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Background', '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        bg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float)
        self.normalizer.method = 'reference'
        self.normalizer.reference = bg
        if self._raw is not None:
            self.data = self._raw

    @pyqtSlot()
    def saveParameters(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getSaveFileName
            filename, _ = get(self, 'Save Parameters', '', 'JSON (*.json)')
        if not filename:
            return
        properties = self.controls.properties
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=4, sort_keys=True)

    def _updateProfile(self) -> None:
        x_p = self.controls.x_p.value()
        y_p = self.controls.y_p.value()
        self.profileWidget.data = Azimuthal.std(self._data, (x_p, y_p))

    @pyqtSlot(float, float)
    def _handleROIChanged(self, x_p: float, y_p: float) -> None:
        if self._data is None:
            return
        with QSignalBlocker(self.controls):
            self.controls.properties = dict(x_p=x_p, y_p=y_p)
        self._updateProfile()
        self.fitWidget.setData(*self.crop())

    @pyqtSlot(int)
    def _handleRadiusChanged(self, radius: int) -> None:
        if self._data is None:
            return
        self.profileWidget.radius = radius
        self.fitWidget.setData(*self.crop())

    @pyqtSlot(str, float)
    def _handlePropertyChanged(self, name: str, value: float) -> None:
        if name == 'x_p':
            self.imageWidget.x_p = value
        elif name == 'y_p':
            self.imageWidget.y_p = value
        self.profileWidget.properties = {name: value}
        self.fitWidget.refreshPreview()

    def crop(self) -> tuple[NDArray[float], QRectF, NDArray[float]]:
        get = self.imageWidget.roi.getArraySlice
        (sy, sx), _ = get(self._data, self.imageWidget.image)
        return (self._data[sy, sx],
                self.imageWidget.rect(),
                self.coordinates[:, sy, sx])

    @pyqtSlot()
    def estimate(self) -> None:
        if self._data is None:
            return
        props = self.controls.properties
        self._pre_estimate = {k: props[k] for k in ('z_p', 'a_p')
                              if k in props}
        self.actionUndoEstimate.setEnabled(True)
        data, _, _ = self.crop()
        instrument = self.fitWidget.optimizer.model.instrument
        result = Estimator(instrument=instrument).estimate(data)
        updates = {k: float(result[k]) for k in ('z_p', 'a_p')
                   if result[k] is not None and np.isfinite(result[k])}
        if updates:
            self.controls.properties = updates
            self.fitWidget.refreshPreview()
        logger.info(f'Estimate: {result}')
        self.statusBar().showMessage('Estimation complete', 2000)

    @pyqtSlot()
    def undoEstimate(self) -> None:
        if self._pre_estimate is None:
            return
        self.controls.properties = self._pre_estimate
        self.fitWidget.refreshPreview()
        self._pre_estimate = None
        self.actionUndoEstimate.setEnabled(False)

    @pyqtSlot(bool)
    def setRobust(self, state: bool) -> None:
        optimizer = self.fitWidget.optimizer
        optimizer.robust = state
        self.optimizerWidget.settings = optimizer.settings

    def closeEvent(self, event) -> None:
        self.fitWidget.shutdown()
        super().closeEvent(event)

    @pyqtSlot()
    def optimize(self) -> None:
        if self._data is None:
            return
        optimizer = self.fitWidget.optimizer
        optimizer.model.properties = self.controls.properties
        optimizer.fixed = self.controls.fixed
        optimizer.robust = self.actionRobust.isChecked()
        data, _, coordinates = self.crop()
        self.fitWidget.optimizeAsync(data, coordinates)

    @pyqtSlot()
    def _onOptimizationStarted(self) -> None:
        self.actionOptimize.setEnabled(False)
        self.actionEstimate.setEnabled(False)
        self.controls.setEnabled(False)
        self.imageWidget.setEnabled(False)
        self._progress.setVisible(True)
        self.statusBar().showMessage('Optimizing...')

    @pyqtSlot(object)
    def _onOptimizationFinished(self, result: pd.Series) -> None:
        self._progress.setVisible(False)
        self.controls.setEnabled(True)
        self.imageWidget.setEnabled(True)
        self.actionOptimize.setEnabled(True)
        self.actionEstimate.setEnabled(True)
        self.controls.properties = self.fitWidget.optimizer.model.properties
        logger.info(f'Optimization complete\n{result}')
        self.statusBar().showMessage('Optimization complete', 2000)

    @pyqtSlot(str)
    def _onOptimizationError(self, message: str) -> None:
        self._progress.setVisible(False)
        self.controls.setEnabled(True)
        self.imageWidget.setEnabled(True)
        self.actionOptimize.setEnabled(True)
        self.actionEstimate.setEnabled(True)
        logger.error(f'Optimization failed: {message}')
        self.statusBar().showMessage(f'Optimization failed: {message}', 5000)


def lmtool() -> None:
    from pylorenzmie.lmtool.ALMWidget import ALMWidget
    from pyqtgraph.Qt.QtWidgets import QApplication
    import sys
    import argparse

    logging.basicConfig(level=logging.WARNING,
                        format='%(name)s: %(levelname)s: %(message)s')

    default_file = str(_DIR.parent / 'docs' / 'tutorials' / 'crop.png')

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, default=default_file,
                        nargs='?', action='store')
    parser.add_argument('-b', '--background', dest='background',
                        default=None, action='store',
                        help='background: image path or intensity value')
    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    normalizer = Normalizer()
    if args.background is not None:
        try:
            value = float(args.background)
            normalizer = Normalizer(method='reference', reference=value)
        except ValueError:
            bg_path = Path(args.background)
            if bg_path.exists():
                bg = cv2.imread(str(bg_path),
                                cv2.IMREAD_GRAYSCALE).astype(float)
                normalizer = Normalizer(method='reference', reference=bg)
            else:
                logger.warning(
                    f'Background not found: {args.background!r}; '
                    'using median filter')

    app = QApplication(qt_args)
    tool = LMTool(ALMWidget, args.filename, normalizer)
    tool.show()
    sys.exit(app.exec())


if __name__ == '__main__':  # pragma: no cover
    lmtool()
