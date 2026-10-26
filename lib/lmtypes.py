'''Shared type aliases for pylorenzmie.'''

import pandas as pd
import numpy as np
from numpy.typing import NDArray


Property = bool | int | float | str
Properties = dict[str, Property]
Image = NDArray[np.floating] | NDArray[np.integer]
Images = Image | list[Image]
Coordinates = NDArray[np.floating]
Coefficients = NDArray[np.complexfloating]
Field = NDArray[np.complexfloating]
Result = pd.Series | pd.DataFrame
Results = Result | list[Result]
