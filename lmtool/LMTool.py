#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyqtgraph.Qt import uic
from pyqtgraph.Qt.QtCore import pyqtProperty, pyqtSlot, QRectF
from pyqtgraph.Qt.QtGui import QCloseEvent
from pyqtgraph.Qt.QtWidgets import (QMainWindow, QFileDialog, QProgressBar)

from pylorenzmie.analysis import Localizer
from pylorenzmie.lib import (Azimuthal, LMObject)
from pylorenzmie.lib.lmtypes import Coordinates, Image
from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.lmtool.Preview import Preview
from pylorenzmie.lmtool.tasks import (FitResult, Snapshot, TaskRunner,
                                       run_estimate, run_fit)
from pylorenzmie.utilities import Normalizer, estimate_noise

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

    #: Number of candidate particles that Find Particles keeps, the
    #: strongest first.  A noisy image can have hundreds of candidates.
    MAX_CANDIDATES = 10

    def __init__(self,
                 controls: type[LMWidget],
                 filename: str | None = None,
                 normalizer: Normalizer | None = None) -> None:
        super().__init__()
        uic.loadUi(_DIR / self.uiFile, self)
        self.normalizer = normalizer if normalizer is not None else Normalizer()
        self._raw = None
        self._data = None
        self.coordinates = None
        self._pre_estimate = None
        self._moving = False
        self._candidates = None
        self._candidate = -1
        self.tasks = TaskRunner()
        self._setupTheory(controls())
        self._setupStatusBar()
        self._connectSignals()
        self.readHologram(filename)

    def _setupTheory(self, controls: LMWidget) -> None:
        layout = self.controls.parent().layout()
        layout.replaceWidget(self.controls, controls)
        self.controls.close()
        self.controls = controls
        self.session = controls.session
        self.preview = Preview(self.session)
        self.preview.updated.connect(self._showPreview)
        self.profileWidget.model = None       # the preview has the model
        self.profileWidget.radius = self.imageWidget.radius
        self.preview.setRadius(self.imageWidget.radius)
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
        self.session.parametersChanged.connect(self._onParametersChanged)
        self.actionOpen.triggered.connect(self.readHologram)
        self.actionOpenBackground.triggered.connect(self.readBackground)
        self.actionEstimateNoise.triggered.connect(self.estimateNoise)
        self.actionFindParticles.triggered.connect(self.findParticles)
        self.actionNextParticle.triggered.connect(self.nextParticle)
        self.actionEstimate.triggered.connect(self.estimate)
        self.actionUndoEstimate.triggered.connect(self.undoEstimate)
        self.actionSaveParameters.triggered.connect(self.saveParameters)
        self.saveResult.triggered.connect(self.fitWidget.saveResult)
        self.saveResultAs.triggered.connect(self.fitWidget.saveResultAs)
        self.actionRobust.toggled.connect(self.setRobust)
        self.actionOptimize.triggered.connect(self.optimize)
        self.optimizerWidget.settingChanged.connect(
            self.fitWidget.setSetting)
        self.actionCancel.triggered.connect(self.tasks.cancel)
        self.tasks.started.connect(self._onTaskStarted)
        self.tasks.finished.connect(self._onTaskFinished)
        self.tasks.failed.connect(self._onTaskFailed)
        self.tasks.cancelled.connect(self._onTaskCancelled)
        self.fitWidget.resultSaved.connect(self._onResultSaved)
        self.fitWidget.saveFailed.connect(self._onSaveFailed)

    def _setUIBusy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self.controls.setEnabled(not busy)
        self.imageWidget.setEnabled(not busy)
        self.actionOptimize.setEnabled(not busy)
        self.actionEstimate.setEnabled(not busy)
        self.actionFindParticles.setEnabled(not busy)
        self.actionEstimateNoise.setEnabled(not busy)
        self.actionNextParticle.setEnabled(not busy)
        self.actionCancel.setEnabled(busy)

    @pyqtProperty(np.ndarray)
    def data(self) -> Image:
        return self._data

    @data.setter
    def data(self, data: Image) -> None:
        self._raw = data
        self._data = self.normalizer(data)
        self.coordinates = LMObject.meshgrid(data.shape, flatten=False)
        self.controls.x_p.setRange((0, data.shape[1]-1))
        self.controls.y_p.setRange((0, data.shape[0]-1))
        self.imageWidget.data = self._data
        self._setNoise(self._data)
        self._updateProfile()
        self._showCrop()

    @pyqtSlot()
    def readHologram(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Hologram', '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        self.fitWidget.datafile = filename
        data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if data is None:
            logger.error(f'Could not read image: {filename!r}')
            self.statusBar().showMessage(f'Could not read: {filename}', 5000)
            return
        self.data = data.astype(float)
        self.findParticles()

    def _setNoise(self, image: Image) -> None:
        '''Set the noise of the model to the estimate for ``image``.'''
        low, high = (self.controls.noise.range()
                     if hasattr(self.controls, 'noise') else (0., np.inf))
        noise = float(np.clip(estimate_noise(image), low, high))
        self.session.update(dict(noise=noise))

    @pyqtSlot()
    def estimateNoise(self) -> None:
        '''Estimate the noise from the region of interest.

        The estimate is the width of the pixel-to-pixel variation, which
        does not include the fringes or a smooth background.  Every
        residual in the Fit tab is divided by it, so the reduced chi-squared
        of a fit is meaningful only when it is right.
        '''
        if self._data is None:
            return
        data, _, _ = self.crop()
        self._setNoise(data)
        self.statusBar().showMessage(
            f'Noise: {self.session.get("noise"):.3f}', 5000)

    @pyqtSlot()
    def findParticles(self) -> None:
        '''Look for particles in the image and select the first one.

        The ROI moves to the selected particle and takes the size of the
        region that encloses its fringes.  :meth:`nextParticle` selects
        the others.
        '''
        if self._data is None:
            return
        self.statusBar().showMessage('Looking for particles ...')
        try:
            found = Localizer().localize(self._data,
                                         maxfeatures=self.MAX_CANDIDATES)
        except Exception as error:
            logger.exception('Localization failed')
            self._candidates = None
            self.statusBar().showMessage(
                f'Localization failed: {type(error).__name__}: {error}', 8000)
            return
        self._candidates = found
        self._candidate = -1
        if len(found) == 0:
            self.statusBar().showMessage(
                'No particles found: move the ROI by hand', 8000)
            return
        self.nextParticle()

    @pyqtSlot()
    def nextParticle(self) -> None:
        '''Select the next particle that :meth:`findParticles` found.'''
        if self._candidates is None or len(self._candidates) == 0:
            self.findParticles()
            return
        self._candidate = (self._candidate + 1) % len(self._candidates)
        candidate = self._candidates.iloc[self._candidate]
        _, width, height = candidate['bbox']
        radius = int(np.clip(max(width, height) // 2, 30, 250))
        self.imageWidget.setRegion(candidate['x_p'], candidate['y_p'],
                                   radius)
        n = len(self._candidates)
        nfound = self._candidates.attrs.get('nfound', n)
        where = f'({candidate["x_p"]:.0f}, {candidate["y_p"]:.0f})'
        if n == 1 and nfound == 1:
            message = f'Only one particle found, at {where}'
        else:
            message = f'Particle {self._candidate + 1} of {n} at {where}'
            if nfound > n:
                message += (f', the strongest of {nfound} candidates: '
                            'the image may be noisy')
        self.statusBar().showMessage(message, 8000)

    @pyqtSlot()
    def readBackground(self, filename: str | None = None) -> None:
        if not filename:
            get = QFileDialog.getOpenFileName
            filename, _ = get(self, 'Open Background', '',
                              'Images (*.png *.tif *.tiff)')
        if not filename:
            return
        bg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            logger.error(f'Could not read background: {filename!r}')
            self.statusBar().showMessage(
                f'Could not read: {filename}', 5000)
            return
        self.normalizer.method = 'reference'
        bg = bg.astype(float)
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
        properties = self.session.properties
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=4, sort_keys=True)

    def _updateProfile(self) -> None:
        x_p = self.session.get('x_p')
        y_p = self.session.get('y_p')
        self.profileWidget.data = Azimuthal.std(self._data, (x_p, y_p))

    @pyqtSlot(float, float)
    def _handleROIChanged(self, x_p: float, y_p: float) -> None:
        if self._data is None:
            return
        self._moving = True               # the ROI moved, so do not move it
        try:
            self.session.update(dict(x_p=x_p, y_p=y_p))
        finally:
            self._moving = False
        self._updateProfile()
        self._showCrop()

    @pyqtSlot(int)
    def _handleRadiusChanged(self, radius: int) -> None:
        if self._data is None:
            return
        self.profileWidget.radius = radius
        self.preview.setRadius(radius)
        self._showCrop()

    @pyqtSlot(list)
    def _onParametersChanged(self, names: list[str]) -> None:
        '''Show parameters that changed in the ROI, the profile and the fit.'''
        if self._moving:
            return
        for name in ('x_p', 'y_p'):
            if name in names:
                value = self.session.get(name)
                if abs(getattr(self.imageWidget, name) - value) > 1e-6:
                    setattr(self.imageWidget, name, value)
        # the preview shows the other parameters when the session announces
        # them

    def _showCrop(self) -> None:
        '''Show the region of interest and ask for the prediction for it.'''
        data, rect, coordinates = self.crop()
        self.fitWidget.setData(data, rect, coordinates)
        self.preview.setCoordinates(coordinates)

    @pyqtSlot(object, object, float)
    def _showPreview(self, hologram, profile, noise: float) -> None:
        self.fitWidget.showModel(hologram, noise)
        self.profileWidget.showTheory(profile)

    def crop(self) -> tuple[Image, QRectF, Coordinates]:
        get = self.imageWidget.roi.getArraySlice
        (sy, sx), _ = get(self._data, self.imageWidget.image)
        return (self._data[sy, sx],
                self.imageWidget.rect(),
                self.coordinates[:, sy, sx])

    def _snapshot(self) -> Snapshot:
        '''A copy of what a task needs: the session and the region.'''
        data, _, coordinates = self.crop()
        optimizer = self.fitWidget.optimizer
        return Snapshot.take(self.session.cls, self.session.properties,
                             data, coordinates, self.session.fixed,
                             optimizer.settings, optimizer.fraction)

    @pyqtSlot()
    def estimate(self) -> None:
        '''Estimate the height, radius and index of the particle.

        The search runs in a thread, on a copy of the session.
        '''
        if self._data is None or self.tasks.busy:
            return
        props = self.session.properties
        # Save only the scalar physical parameters for undo; x_p/y_p are
        # excluded because the user may have moved the ROI since estimating.
        self._pre_estimate = {k: props[k] for k in ('z_p', 'a_p', 'n_p')
                              if k in props}
        self.actionUndoEstimate.setEnabled(True)
        self.tasks.start('estimate', run_estimate, self._snapshot())

    def _onEstimated(self, result: pd.Series) -> None:
        updates = {k: float(result[k]) for k in ('z_p', 'a_p', 'n_p')
                   if k in result
                   and result[k] is not None
                   and np.isfinite(result[k])}
        if updates:
            self.session.update(updates)
        logger.info(f'Estimate: {result}')
        self.statusBar().showMessage('Estimation complete', 2000)

    @pyqtSlot()
    def undoEstimate(self) -> None:
        if self._pre_estimate is None:
            return
        self.session.update(self._pre_estimate)
        self._pre_estimate = None
        self.actionUndoEstimate.setEnabled(False)

    @pyqtSlot(bool)
    def setRobust(self, state: bool) -> None:
        optimizer = self.fitWidget.optimizer
        optimizer.robust = state
        self.optimizerWidget.settings = optimizer.settings

    def closeEvent(self, event: QCloseEvent) -> None:
        self.tasks.shutdown()
        super().closeEvent(event)

    @pyqtSlot()
    def optimize(self) -> None:
        '''Fit the model to the region of interest.

        The fit runs in a thread, on a copy of the session; the result
        arrives when it is done, and Cancel (Esc) stops it.
        '''
        if self._data is None or self.tasks.busy:
            return
        self.tasks.start('optimize', run_fit, self._snapshot())

    def _onOptimized(self, fit: FitResult) -> None:
        self.session.update(fit.properties)
        self.fitWidget.setResult(fit.result, fit.metadata)
        logger.info(f'Optimization complete\n{fit.result}')
        message = 'Optimization complete'
        if 'redchi' in fit.result:
            message += f': reduced \u03c7\u00b2 = {fit.result["redchi"]:.2f}'
        self.statusBar().showMessage(message, 8000)

    @pyqtSlot(str)
    def _onTaskStarted(self, name: str) -> None:
        self._setUIBusy(True)
        verb = 'Estimating' if name == 'estimate' else 'Optimizing'
        self.statusBar().showMessage(f'{verb} ... (Esc cancels)')

    @pyqtSlot(str, object)
    def _onTaskFinished(self, name: str, result: object) -> None:
        self._setUIBusy(False)
        if name == 'estimate':
            self._onEstimated(result)
        else:
            self._onOptimized(result)

    @pyqtSlot(str, str)
    def _onTaskFailed(self, name: str, message: str) -> None:
        self._setUIBusy(False)
        what = 'Estimation' if name == 'estimate' else 'Optimization'
        logger.error(f'{what} failed: {message}')
        self.statusBar().showMessage(f'{what} failed: {message}', 8000)

    @pyqtSlot(str)
    def _onTaskCancelled(self, name: str) -> None:
        self._setUIBusy(False)
        if name == 'estimate':            # nothing changed: nothing to undo
            self._pre_estimate = None
            self.actionUndoEstimate.setEnabled(False)
        self.statusBar().showMessage('Cancelled', 3000)

    @pyqtSlot(str)
    def _onResultSaved(self, filename: str) -> None:
        self.statusBar().showMessage(f'Saved {filename}', 5000)

    @pyqtSlot(str)
    def _onSaveFailed(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)


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
