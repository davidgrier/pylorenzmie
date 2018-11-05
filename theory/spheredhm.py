import numpy as np
from pylorenzmie.theory.spherefield import spherefield


def spheredhm(rp, a_p, n_p, n_m, dim, mpp=0.135, lamb=.447, alpha=False,
              precision=False,  lut=False):
    """
    Compute holographic microscopy image of a sphere immersed in a transparent 
    medium.

    Args:
        rp  : [x, y, z] 3 dimensional position of sphere relative to center
              of image.
        a_p  : radius of sphere [micrometers]
        n_p  : (complex) refractive index of sphere
        n_m  : (complex) refractive index of medium
        dim : [nx, ny] dimensions of image [pixels]

    NOTE: The imaginary parts of the complex refractive indexes
    should be positive for absorbing materials.  This follows the
    convention used in SPHERE_COEFFICIENTS.

    Keywords:
        precision: 
        alpha: fraction of incident light scattered by particle.
            Default: 1.
        lamb:  vacuum wavelength of light [micrometers]
        mpp: micrometers per pixel
        precision: relative precision with which fields are calculated.

    Returns:
        dhm: [nx, ny] holographic image                
    """

    nx, ny = dim
    x = np.tile(np.arange(nx, dtype=float), ny)
    y = np.repeat(np.arange(ny, dtype=float), nx)
    x -= float(nx)/2. + float(rp[0])
    y -= float(ny)/2. + float(rp[1])

    if lut:
        rho = np.sqrt(x**2 + y**2)
        x = np.arange(np.fix(rho).max()+1)
        y = 0. * x

    zp = float(rp[2])

    field = spherefield(x, y, zp, a_p, n_p, n_m=n_m, cartesian=True, mpp=mpp,
                        lamb=lamb, precision=precision)
    if alpha:
        field *= alpha

    k = 2.0*np.pi/(lamb/np.real(n_m)/mpp)

    # Compute the sum of the incident and scattered fields, then square.
    field *= np.exp(np.complex(0., -k*zp))
    field[0, :] += 1.0
    image = np.sum(np.real(field*np.conj(field)), axis=0)

    if lut:
        image = np.interpolate(image, rho, cubic=-0.5)

    return image.reshape(int(ny), int(nx))


def test_spheredhm():
    '''Produces a test hologram resulting from a spherical scatterer.'''
    # Particle and imaging properties.
    mpp = 0.135
    z = 100.
    rp = [0, 0, z]
    a_p = 0.75
    n_p = 1.5
    n_m = 1.339
    dim = [201, 201]
    lamb = 0.447

    # Produce Image.
    image = spheredhm(rp, a_p, n_p, n_m, dim, lamb=lamb, mpp=mpp)

    # Plot the hologram.
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title("Test Hologram")
    plt.gray()
    plt.show()


if __name__ == '__main__':
    test_spheredhm()
