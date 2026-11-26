# utilities

Helpers that are not part of the model or the pipeline.

- `normalizer.py` divides holograms by their background and finds the
  saturated pixels, and `vmedian.py` is a fast approximate running median
  for that background.
- `noise.py` estimates the noise in an image.
- `readimage.py` reads an image file in gray at its own bit depth, with
  Pillow.
- `water.py` has the physical properties of water.
- `example_hologram.py` loads the holograms in `data/`, which the tests
  and the tutorials use.

Update this file when the role or contents of the directory change.
