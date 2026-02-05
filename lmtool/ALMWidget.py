from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.theory import AberratedLorenzMie


class ALMWidget(LMWidget):

    cls = AberratedLorenzMie
    uiFile = 'ALMWidget.ui'


if __name__ == '__main__':
    ALMWidget.example()
