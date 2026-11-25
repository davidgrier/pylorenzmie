# theory

The physical model: scatterers, the instrument, and the Lorenz-Mie
calculators that turn them into holograms.

- `Instrument.py`, `Particle.py`, `Sphere.py`, `LayeredSphere.py`,
  `Cluster.py`, `Dimer.py` and `Pair.py` describe the scatterers and the
  microscope.  `_mie.py` has the recurrences of the Mie coefficients.
- `LorenzMie.py` is the NumPy calculator.  `numbaLorenzMie.py`,
  `cupyLorenzMie.py`, `torchLorenzMie.py` (with `_tritonkernel.py` and
  `torchLorenzMieBatch.py`) and `jaxLorenzMie.py` are accelerated
  calculators with the same interface.
- `Aberrated.py` and `Distorted.py` add the Seidel aberrations of the
  imaging system, with `_warp.py` holding the kernels of the distortion.
- `generators.py` has the generating functions of the Seidel aberrations.
- `_mie.py` holds the recurrences of the Mie coefficients of a sphere, which
  Numba compiles when it is installed.

`pylorenzmie.theory.LorenzMie` is the fastest calculator that works in the
environment; set `PYLORENZMIE_BACKEND` to choose another.

Update this file when the role or contents of the directory change.
