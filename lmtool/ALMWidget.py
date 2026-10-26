from pylorenzmie.lmtool.LMWidget import LMWidget
from pylorenzmie.theory import AberratedLorenzMie


class ALMWidget(LMWidget):
    '''Parameter-control panel for an AberratedLorenzMie model.

    The panel works with any model that has the aberration parameters, such
    as a :func:`~pylorenzmie.theory.Distorted.Distorted` one; see
    :meth:`with_model`.
    '''

    cls = AberratedLorenzMie
    uiFile = 'ALMWidget.ui'

    @classmethod
    def with_model(cls, model: type) -> type['ALMWidget']:
        '''The panel class for a model class.

        Parameters
        ----------
        model : type
            A :class:`~pylorenzmie.theory.LorenzMie.LorenzMie` calculator
            with the aberration parameters of the panel.

        Returns
        -------
        type
            A subclass of this class whose sessions use *model*.
        '''
        return type(cls.__name__, (cls,), {'cls': model})


if __name__ == '__main__':  # pragma: no cover
    ALMWidget.example()
