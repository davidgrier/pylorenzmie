from numpy.typing import NDArray
import pandas as pd
from pyqtgraph.Qt.QtCore import pyqtProperty
from pyqtgraph.Qt.QtWidgets import (QTableWidget, QTableWidgetItem,
                                    QSizePolicy)


class ResultWidget(QTableWidget):
    '''Table widget displaying optimizer results as a pandas DataFrame.'''

    def __init__(self, *args,
                 data: NDArray[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Minimum,
                           QSizePolicy.Policy.Minimum)
        self.data = data

    @pyqtProperty(pd.DataFrame)
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame | None) -> None:
        self._data = data
        if data is None:
            return
        headers = list(data)
        self.setRowCount(data.shape[0])
        self.setColumnCount(data.shape[1])
        self.setHorizontalHeaderLabels(headers)

        values = data.values
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                self.setItem(row, col,
                             QTableWidgetItem(str(values[row, col])))

        self.resizeColumnsToContents()
        self.setFixedSize(self.horizontalHeader().length() +
                          self.verticalHeader().width(),
                          self.verticalHeader().length() +
                          self.horizontalHeader().height())

    @classmethod
    def example(cls) -> None:
        from pyqtgraph import mkQApp

        app = mkQApp()
        data = pd.DataFrame.from_dict({'a_p': [1.01], 'n_p': [1.42]})
        widget = cls(data=data)
        widget.show()
        app.exec()


if __name__ == '__main__':  # pragma: no cover
    ResultWidget.example()
