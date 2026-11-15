'''Shared type aliases for pylorenzmie.'''

from typing import Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray


# what a ``properties`` dict holds: numbers and strings, ``None`` for a
# value that is not known yet, and the lists and dicts of a configuration
Property = bool | int | float | str | list | dict | None
Properties = dict[str, Property]
Bounds = dict[str, tuple[float, float]]
# keyword arguments of a solver, which take any kind of value
Settings = dict[str, Any]
Image = NDArray[np.floating] | NDArray[np.integer]
Images = Image | list[Image]
Coordinates = NDArray[np.floating]
Coefficients = NDArray[np.complexfloating]
Field = NDArray[np.complexfloating]
Result = pd.Series | pd.DataFrame
Results = Result | list[Result]
