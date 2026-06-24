from pylorenzmie.theory.LorenzMie import LorenzMie


def best_model(**kwargs) -> LorenzMie:
    '''Return the best available Lorenz-Mie model instance.

    Tries backends in priority order: JAX → CuPy → Numba → NumPy.
    JAX is preferred because it offers an analytical Jacobian on every
    platform, including CPU.  CuPy is next for CUDA GPU throughput.
    Numba provides JIT-compiled CPU acceleration.  NumPy is the
    always-available fallback.

    The first backend whose runtime is importable and functional is
    used; the rest are skipped silently.

    Parameters
    ----------
    **kwargs
        Forwarded to the chosen model constructor (e.g. ``coordinates``,
        ``particle``, ``instrument``).

    Returns
    -------
    model : LorenzMie
        Instance of the best available subclass.

    Examples
    --------
    >>> from pylorenzmie.theory import best_model
    >>> model = best_model()
    >>> print(type(model).__name__)
    '''
    # JAX: analytical Jacobian + cross-platform GPU (Metal / CUDA / TPU)
    try:
        from pylorenzmie.theory.jaxLorenzMie import jaxLorenzMie, _jax_available
        if _jax_available:
            return jaxLorenzMie(**kwargs)
    except Exception:
        pass

    # CuPy: NVIDIA GPU via CUDA
    try:
        from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
        return cupyLorenzMie(**kwargs)
    except Exception:
        pass

    # Numba: JIT-compiled CPU
    try:
        from pylorenzmie.theory.numbaLorenzMie import numbaLorenzMie, _numba_available
        if _numba_available:
            return numbaLorenzMie(**kwargs)
    except Exception:
        pass

    # NumPy: always available
    return LorenzMie(**kwargs)
