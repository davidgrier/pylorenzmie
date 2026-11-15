# lib

The base classes and helpers that the rest of the package is built on.

- `LMObject.py`: the abstract base class of every major class, with the
  `properties` interface, JSON and pandas serialization, and `meshgrid`.
- `meshgrid.py`: pixel coordinates of an image.
- `lmtypes.py`: the type aliases (`Image`, `Coordinates`, `Properties`).
- `Azimuthal.py` and `CircleTransform.py`: re-export the azimuthal
  statistics and the ring-finding transform from `dgimage`, which is where
  they are edited.

Update this file when the role or contents of the directory change.
