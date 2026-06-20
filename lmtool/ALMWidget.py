from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.theory import AberratedLorenzMie


class ALMWidget(LMWidget):
    '''Parameter-control panel for an AberratedLorenzMie model.'''

    cls = AberratedLorenzMie
    uiFile = 'ALMWidget.ui'


if __name__ == '__main__':  # pragma: no cover
    ALMWidget.example()
