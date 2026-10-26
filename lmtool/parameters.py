'''The parameters that lmtool shows, and how.

Each :class:`ParameterSpec` describes one parameter of the model: the label,
the range of its slider, the step of its spin box, the number of decimals, its
unit, the value it starts with and whether a fit holds it fixed until the user
frees it.  :data:`PARAMETERS` lists them in the order that the panel uses.

The layout of the panel is in a Qt Designer file, which has a control for the
parameters that are shown.  A parameter in the list that the panel has no
control for is hidden: it is part of the model and of the session, a fit holds
it fixed, and the user cannot change it.  A test checks that every control of
every panel has a specification, and that every specification is valid.
'''

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpec:
    '''Description of one parameter of the model.

    Parameters
    ----------
    name : str
        Name of the property of the model.
    text : str
        Label of the control; may contain HTML, such as subscripts.
    range : tuple[float, float]
        Smallest and largest value of the slider and the spin box.
    value : float
        Value that the control starts with.
    decimals : int, optional
        Number of decimals that the spin box shows.  Default: 2.
    step : float, optional
        Step of the spin box and the slider.  Default: the control's own.
    suffix : str, optional
        Unit shown after the value, with its leading space.  Default: none.
    prefix : str, optional
        Text shown before the value.  Default: none.
    fixed : bool, optional
        True if a fit holds the parameter fixed until the user frees it.
        Default: False.
    '''

    name: str
    text: str
    range: tuple[float, float]
    value: float
    decimals: int = 2
    step: float | None = None
    suffix: str = ''
    prefix: str = ''
    fixed: bool = False

    def __post_init__(self) -> None:
        low, high = self.range
        if not low < high:
            raise ValueError(f'{self.name}: the range {self.range} is empty')
        if not low <= self.value <= high:
            raise ValueError(f'{self.name}: the value {self.value} is not in '
                             f'the range {self.range}')
        if self.step is not None and self.step <= 0.:
            raise ValueError(f'{self.name}: the step must be positive')

    def settings(self) -> dict:
        '''Settings for the ``setSettings`` method of a control.'''
        settings = dict(text=self.text, range=list(self.range),
                        value=self.value, decimals=self.decimals,
                        fixed=self.fixed)
        if self.step is not None:
            settings['step'] = self.step
        if self.suffix:
            settings['suffix'] = self.suffix
        if self.prefix:
            settings['prefix'] = self.prefix
        return settings


#: The parameters of the model, in the order of the panel.
PARAMETERS = (
    ParameterSpec(
        name='wavelength',
        text='wavelength',
        range=(0.405, 1.07),
        value=0.447,
        decimals=3,
        suffix=' μm',
        fixed=True),
    ParameterSpec(
        name='magnification',
        text='magnification',
        range=(0.046, 0.135),
        value=0.048,
        decimals=3,
        step=0.001,
        suffix=' μm/pixel',
        fixed=True),
    ParameterSpec(
        name='n_m',
        text='n<sub>m</sub>',
        range=(1.3, 1.7),
        value=1.34,
        decimals=3,
        step=0.001,
        fixed=True),
    ParameterSpec(
        name='a_p',
        text='a<sub>p</sub>',
        range=(0.1, 10.0),
        value=0.75,
        decimals=3,
        step=0.005,
        suffix=' μm'),
    ParameterSpec(
        name='n_p',
        text='n<sub>p</sub>',
        range=(1.0, 2.5),
        value=1.45,
        decimals=3,
        step=0.001),
    ParameterSpec(
        name='k_p',
        text='k<sub>p</sub>',
        range=(0.0, 10.0),
        value=0.0,
        decimals=4,
        step=0.1,
        fixed=True),
    ParameterSpec(
        name='x_p',
        text='x<sub>p</sub>',
        range=(0.0, 1240.0),
        value=100.0,
        decimals=2,
        step=0.1,
        suffix=' pixel'),
    ParameterSpec(
        name='y_p',
        text='y<sub>p</sub>',
        range=(0.0, 1080.0),
        value=100.0,
        decimals=2,
        step=0.1,
        suffix=' pixel'),
    ParameterSpec(
        name='z_p',
        text='z<sub>p</sub>',
        range=(20.0, 2000.0),
        value=100.0,
        decimals=2,
        step=0.1,
        suffix=' pixel'),
    ParameterSpec(
        name='defocus',
        text='defocus',
        range=(-10.0, 10.0),
        value=0.0,
        decimals=1,
        step=0.1),
    ParameterSpec(
        name='spherical',
        text='spherical',
        range=(-10.0, 10.0),
        value=0.0,
        decimals=1,
        step=0.1,
        fixed=True),
    ParameterSpec(
        name='coma',
        text='coma',
        range=(-10.0, 10.0),
        value=0.0,
        decimals=1,
        step=0.1),
    ParameterSpec(
        name='astigmatism',
        text='astigmatism',
        range=(-10.0, 10.0),
        value=0.0,
        decimals=1,
        step=0.1),
    ParameterSpec(
        name='distortion',
        text='distortion',
        range=(-10.0, 10.0),
        value=0.0,
        decimals=1,
        step=0.1),
    ParameterSpec(
        name='piston',
        text='piston',
        range=(-0.5, 0.5),
        value=0.0,
        decimals=2,
        step=0.01,
        fixed=True),
    ParameterSpec(
        name='x_c',
        text='x<sub>c</sub>',
        range=(0.0, 1240.0),
        value=0.0,
        decimals=2,
        step=0.1,
        suffix=' pixel',
        fixed=True),
    ParameterSpec(
        name='y_c',
        text='y<sub>c</sub>',
        range=(0.0, 1080.0),
        value=0.0,
        decimals=2,
        step=0.1,
        suffix=' pixel',
        fixed=True),
    ParameterSpec(
        name='noise',
        text='noise',
        range=(0.01, 1.0),
        value=0.05,
        decimals=3,
        step=0.01,
        fixed=True),
)


def specification(name: str) -> ParameterSpec:
    '''The specification of the parameter ``name``.'''
    for spec in PARAMETERS:
        if spec.name == name:
            return spec
    raise KeyError(name)
