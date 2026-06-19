'''Shared type aliases for pylorenzmie.'''

import pandas as pd
from numpy.typing import NDArray


Property = bool | int | float | str
Properties = dict[str, Property]
Image = NDArray[float] | NDArray[int]
Images = Image | list[Image]
Coordinates = NDArray[float]
Coefficients = NDArray[complex]
Field = NDArray[complex]
Result = pd.Series | pd.DataFrame
Results = Result | list[Result]
