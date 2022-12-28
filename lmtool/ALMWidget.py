from pylorenzmie.lmtool.LMWidget import (LMWidget, example)
from pylorenzmie.theory import AberratedLorenzMie


class ALMWidget(LMWidget):

    cls = AberratedLorenzMie
    uifile = 'ALMWidget.ui'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    example(ALMWidget)
