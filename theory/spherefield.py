from pylorenzmie.theory.sphere_coefficients import sphere_coefficients
from pylorenzmie.theory.sphericalfield import sphericalfield
import numpy as np


def spherefield(x, y, z, a_p, n_p, n_m=complex(1.3326, 1.5e-8),
                lamb=0.447, mpp=0.135, precision=False,
                cartesian=False):
    """
    Calculate the complex electric field scattered by a sphere illuminated
    by a plane wave linearly polarized in the x direction.

    Args:
        x:  [npts] array of pixel coordinates [pixels]
        y:  [npts] array of pixel coordinates [pixels]
        z:  If field is required in a single plane, z is the plane's distance 
            from the sphere's center [pixels]. Otherwise, z is an [npts] array of 
            coordinates.
        a_p:  radius of sphere [micrometers]
        n_p: (complex) refractive index of sphere

    Keywords:
        n_m:        refractive index of medium (default water)
        lamb:       vacuum wavelength of light (default 0.6328)
        mpp:        micrometers per pixel      (default 0.101)
        precision:  accuracy of scattering coefficients
        k:          scaled wavenumber in medium
        cartesian:  project to cartesian coordinates

    Returns:
        field: [3,npts]
    """
    ab = sphere_coefficients(a_p, n_p, n_m, lamb)

    if precision:
        # retain first coefficient for bookkeeping
        fac = abs(ab[:, 1])
        w = np.where(fac > precision*max(fac))
        w = np.concatenate((np.array([0]), w[0]))
        ab = ab[w, :]

    lamb_m = lamb/np.real(n_m)/mpp  # medium wavelength [pixel]
    field = sphericalfield(x, y, z, ab, lamb_m, cartesian=cartesian,
                           strength=False)

    return field
