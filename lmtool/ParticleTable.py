'''The particles of an image that lmtool knows, and what it has fitted.'''

from __future__ import annotations

import numpy as np
import pandas as pd
from pyqtgraph.Qt.QtCore import (QAbstractTableModel, QModelIndex, Qt,
                                 pyqtSignal)


def radius_of(bbox: tuple, low: int = 30, high: int = 250) -> int:
    '''Radius of the region of interest for a bounding box, in pixels.'''
    _, width, height = bbox
    return int(np.clip(max(width, height) // 2, low, high))


class ParticleTable(QAbstractTableModel):
    '''Table of the particles in an image: where, whether to use, the result.

    A particle is either found by the localizer or added by hand, from the
    region of interest.  It can be left out (its check box) and, once the user
    has estimated or fitted it, it carries the parameters and the result.  The
    table is the model of the view in the Particles tab; it holds no widgets.

    Rows are numbered from zero.  One row at a time is the current one, the
    particle that the region of interest is on.
    '''

    #: Emitted with the row of the current particle, or -1 for none.
    currentChanged = pyqtSignal(int)

    COLUMNS = ('', 'x_p', 'y_p', 'source', 'status', 'z_p', 'a_p', 'n_p',
               'chi2')

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict] = []
        self._current = -1

    # ------------------------------------------------------------------
    # the Qt model
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.DisplayRole
                and orientation == Qt.Orientation.Horizontal):
            return self.COLUMNS[section].replace('chi2', 'χ²')
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        flags = (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index: QModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = self._rows[index.row()], index.column()
        if column == 0:
            if role == Qt.ItemDataRole.CheckStateRole:
                return (Qt.CheckState.Checked if row['include']
                        else Qt.CheckState.Unchecked)
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        name = self.COLUMNS[column]
        if name in ('x_p', 'y_p'):
            return f'{self._position(row)[name == "y_p"]:.1f}'
        if name in ('source', 'status'):
            return row[name]
        if name in ('z_p', 'a_p', 'n_p'):
            properties = row['properties']
            if properties is None:
                return ''
            return f'{properties[name]:.{3 if name != "z_p" else 1}f}'
        result = row['result']
        return '' if result is None else f'{result["redchi"]:.2f}'

    def setData(self, index: QModelIndex, value,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            self._rows[index.row()]['include'] = (
                value == Qt.CheckState.Checked.value
                or value == Qt.CheckState.Checked)
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    # ------------------------------------------------------------------
    # the particles
    # ------------------------------------------------------------------

    @staticmethod
    def _position(row: dict) -> tuple[float, float]:
        result = row['result']
        if result is not None:
            return float(result['x_p']), float(result['y_p'])
        return row['x_p'], row['y_p']

    @property
    def count(self) -> int:
        '''Number of particles.'''
        return len(self._rows)

    @property
    def current(self) -> int:
        '''Row of the current particle, or -1.'''
        return self._current

    def setCurrent(self, row: int) -> None:
        '''Make a particle the current one.'''
        row = row if 0 <= row < len(self._rows) else -1
        if row != self._current:
            self._current = row
            self.currentChanged.emit(row)

    def entry(self, row: int) -> dict:
        '''A copy of what the table holds about a particle.

        Keys: ``x_p``, ``y_p``, ``radius``, ``source``, ``include``,
        ``status``, ``properties`` and ``result``.
        '''
        return dict(self._rows[row])

    def position(self, row: int) -> tuple[float, float]:
        '''Where the particle is: the fit if there is one, else where it was
        found.'''
        return self._position(self._rows[row])

    def clear(self) -> None:
        '''Forget every particle.'''
        self.beginResetModel()
        self._rows = []
        self._current = -1
        self.endResetModel()
        self.currentChanged.emit(-1)

    def setFound(self, candidates: pd.DataFrame) -> None:
        '''Replace the particles that the localizer found.

        Particles that were added by hand stay.
        '''
        added = [row for row in self._rows if row['source'] == 'added']
        self.beginResetModel()
        self._rows = [self._new(c.x_p, c.y_p, radius_of(c.bbox), 'found')
                      for c in candidates.itertuples()] + added
        self._current = -1
        self.endResetModel()
        self.currentChanged.emit(-1)

    def add(self, x_p: float, y_p: float, radius: int,
            source: str = 'added') -> int:
        '''Add a particle; returns its row.'''
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(self._new(x_p, y_p, radius, source))
        self.endInsertRows()
        return row

    def remove(self, row: int) -> None:
        '''Remove a particle.'''
        if not 0 <= row < len(self._rows):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._rows[row]
        self.endRemoveRows()
        if self._current == row:
            self._current = -1
            self.currentChanged.emit(-1)
        elif self._current > row:
            self._current -= 1

    @staticmethod
    def _new(x_p: float, y_p: float, radius: int, source: str) -> dict:
        return dict(x_p=float(x_p), y_p=float(y_p), radius=int(radius),
                    source=source, include=True, status='new',
                    properties=None, result=None)

    def rowAt(self, x_p: float, y_p: float,
              tolerance: float = 5.) -> int | None:
        '''Row of the particle nearest to a position, if it is within
        ``tolerance`` pixels.'''
        best, distance = None, tolerance
        for row in range(len(self._rows)):
            x, y = self._position(self._rows[row])
            d = float(np.hypot(x - x_p, y - y_p))
            if d <= distance:
                best, distance = row, d
        return best

    def next(self) -> int | None:
        '''Row of the next particle that is included, after the current one,
        wrapping around; None if none is included.'''
        n = len(self._rows)
        for step in range(1, n + 1):
            row = (self._current + step) % n
            if self._rows[row]['include']:
                return row
        return None

    # ------------------------------------------------------------------
    # what has been done
    # ------------------------------------------------------------------

    def setEstimate(self, row: int, properties: dict) -> None:
        '''Record the parameters that an estimate gave a particle.'''
        entry = self._rows[row]
        entry['properties'] = dict(properties)
        if entry['result'] is None:
            entry['status'] = 'estimated'
        self._changed(row)

    def setResult(self, row: int, properties: dict,
                  result: pd.Series) -> None:
        '''Record the parameters and the result of a fit of a particle.'''
        entry = self._rows[row]
        entry['properties'] = dict(properties)
        entry['result'] = result.copy()
        entry['status'] = 'fitted'
        self._changed(row)

    def _changed(self, row: int) -> None:
        self.dataChanged.emit(self.index(row, 0),
                              self.index(row, len(self.COLUMNS) - 1))

    def results(self, included: bool = True) -> pd.DataFrame:
        '''The fits, one row per particle (empty if there are none).

        Parameters
        ----------
        included : bool, optional
            Leave out the particles that are not checked.  Default: True.
        '''
        rows = {}
        for number, entry in enumerate(self._rows):
            if entry['result'] is None or (included and not entry['include']):
                continue
            rows[number] = entry['result'].to_dict() | dict(
                source=entry['source'])
        frame = pd.DataFrame.from_dict(rows, orient='index')
        frame.index.name = 'particle'
        return frame
