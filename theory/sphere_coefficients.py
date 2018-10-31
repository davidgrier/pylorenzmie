import numpy as np


def n_stop(x, m, method='both'):
    '''
    Return the number of terms to keep in partial wave expansion
    Args:
        x: [nlayers] list of size parameters for each layer
        m: relative refractive index
     to Wiscombe (1980) and Yang (2003).
    '''

    # Wiscombe (1980)
    xl = x[-1]
    if xl <= 8.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 1.)
    elif xl <= 4200.:
        ns = np.floor(xl + 4.05 * xl**(1./3.) + 2.)
    elif xl > 4200.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 2.)

    # Yang (2003) Eq. (30)
    xm = abs(x*m)
    xm_1 = abs(np.roll(x, -1)*m)
    nstop = max(ns, xm, xm_1)
    return int(nstop + 15)


def sphere_coefficients(a_p, n_p, n_m, lamb, resolution=0):
    """
    Calculate the Mie scattering coefficients for a multilayered sphere
    illuminated by a coherent plane wave linearly polarized in the x direction.

    Args:
        a_p: [nlayers] radii of layered sphere [micrometers]
            NOTE: a_p and n_p are reordered automatically so that
            a_p is in ascending order.
        n_p: [nlayers] (complex) refractive indexes of sphere's layers
        n_m: (complex) refractive index of medium
        lamb: wavelength of light [micrometers]

    Keywrods:
        resolution: minimum magnitude of Lorenz-Mie coefficients to retain.
              Default: See references
    Returns:
        ab: the coefficients a,b
    """

    a_p = np.array([a_p])
    n_p = np.array([n_p])
    nlayers = a_p.ndim

    assert n_p.ndim == nlayers, \
        'a_p and n_p must have the same number of elements'

    # arrange shells in size order
    '''
    if nlayers > 1:
        order = a_p.argsort()
        a_p = a_p[order]
        n_p = n_p[order]
    '''
    # size parameters for layers
    x = [abs(2.0 * np.pi * n_m * a_j) for a_j in a_p]
    m = n_p/n_m               # relative refractive index [array]
    nmax = n_stop(x, m)       # number of terms in partial-wave expansion

    # arrays for storing results
    # Note:  May be faster not to use zeros
    ab = np.zeros([nmax+1, 2], complex)
    d1 = np.zeros(nmax+2, complex)
    d1_a = np.zeros([nmax+2, nlayers], complex)
    d1_am1 = np.zeros([nmax+2, nlayers], complex)

    d3 = np.zeros(nmax+1, complex)
    d3_a = np.zeros([nmax+1, nlayers], complex)
    d3_am1 = np.zeros([nmax+1, nlayers], complex)

    psi = np.zeros(nmax+1, complex)
    zeta = np.zeros(nmax+1, complex)
    psiZeta = np.zeros(nmax+1, complex)
    psiZeta_a = np.zeros([nmax+1, nlayers], complex)
    psiZeta_am1 = np.zeros([nmax+1, nlayers], complex)

    q = np.zeros([nmax+1, nlayers], complex)
    ha = np.zeros([nmax+1, nlayers], complex)
    hb = np.zeros([nmax+1, nlayers], complex)

    # Calculate D1, D3 and PsiZeta for z1 in the first layer
    z1 = x[0] * m[0]

    # D1_a[0, nmax + 1] = dcomplex(0) # Eq. (16a)
    for n in range(nmax+1, 0, -1):    # downward recurrence Eq. (16b)
        d1_a[n-1, 0] = n/z1 - 1.0/(d1_a[n, 0] + n/z1)

    psiZeta_a[0, 0] = 0.5 * (1. - np.exp(2j * z1))  # Eq. (18a)
    d3_a[0, 0] = 1j                                 # Eq. (18a)
    for n in range(1, nmax+1):        # upward recurrence Eq. (18b)
        psiZeta_a[n, 0] = psiZeta_a[n-1, 0] * \
            (n/z1 - d1_a[n-1, 0]) * (n/z1 - d3_a[n-1, 0])
        d3_a[n, 0] = d1_a[n, 0] + 1j/psiZeta_a[n, 0]

    # Ha and Hb in the core
    ha[:, 0] = d1_a[0:-1, 0]     # Eq. (7a)
    hb[:, 0] = d1_a[0:-1, 0]     # Eq. (8a)

    # Iterate from layer 2 to layer L
    for ii in range(1, nlayers):
        z1 = x[ii] * m[ii]
        z2 = x[ii-1] * m[ii]
        # Downward recurrence for D1, Eqs. (16a) and (16b)
        #   D1_a[ii, nmax+1]   = dcomplex(0)      # Eq. (16a)
        #   D1_am1[ii, nmax+1] = dcomplex(0)
        for n in range(nmax+1, 0, -1):  # Eq. (16 b)
            d1_a[n-1, ii] = n/z1 - 1./(d1_a[n, ii] + n/z1)
            d1_am1[n-1, ii] = n/z2 - 1./(d1_am1[n, ii] + n/z2)

        # Upward recurrence for PsiZeta and D3, Eqs. (18a) and (18b)
        psiZeta_a[0, ii] = 0.5 * (1. - np.exp(2.j * z1))  # Eq. (18a)
        psiZeta_am1[0, ii] = 0.5 * (1. - np.exp(2.j * z2))
        d3_a[0, ii] = 1j
        d3_am1[0, ii] = 1j
        for n in range(1, nmax+1):    # Eq. (18b)
            psiZeta_a[n, ii] = psiZeta_a[n-1, ii] * \
                (n/z1 - d1_a[n-1, ii]) * (n/z1 - d3_a[n-1, ii])
            psiZeta_am1[n, ii] = psiZeta_am1[n-1, ii] * \
                (n/z2 - d1_am1[n-1, ii]) * (n/z2 - d3_am1[n-1, ii])
            d3_a[n, ii] = d1_a[n, ii] + 1j/psiZeta_a[n, ii]
            d3_am1[n, ii] = d1_am1[n, ii] + 1j/psiZeta_am1[n, ii]

        # Upward recurrence for Q
        q[0, ii] = (np.exp(-2j * z2) - 1.) / (np.exp(-2j * z1) - 1.)
        for n in range(1, nmax+1):
            num = (z1 * d1_a[n, ii] + n) * (n - z1 * d3_a[n-1, ii])
            den = (z2 * d1_am1[n, ii] + n) * (n - z2 * d3_am1[n-1, ii])
            q[n, ii] = (x[ii-1]/x[ii])**2 * q[n-1, ii] * num/den

        # Upward recurrence for Ha and Hb, Eqs. (7b), (8b) and (12) - (15)
        for n in range(1, nmax+1):
            g1 = m[ii] * ha[n, ii-1] - m[ii-1] * d1_am1[n, ii]
            g2 = m[ii] * ha[n, ii-1] - m[ii-1] * d3_am1[n, ii]
            temp = q[n, ii] * g1
            num = g2 * d1_a[n, ii] - temp * d3_a[n, ii]
            den = g2 - temp
            ha[n, ii] = num/den

            g1 = m[ii-1] * hb[n, ii-1] - m[ii] * d1_am1[n, ii]
            g2 = m[ii-1] * hb[n, ii-1] - m[ii] * d3_am1[n, ii]
            temp = q[n, ii] * g1
            num = g2 * d1_a[n, ii] - temp * d3_a[n, ii]
            den = g2 - temp
            hb[n, ii] = num/den
    # ii (layers)

    z1 = complex(x[-1])
    # Downward recurrence for D1, Eqs. (16a) and (16b)
    # D1[nmax+1] = dcomplex(0)          # Eq. (16a)
    for n in range(nmax, 0, -1):        # Eq. (16b)
        d1[n-1] = n/z1 - (1./(d1[n] + n/z1))

    # Upward recurrence for Psi, Zeta, PsiZeta and D3, Eqs. (18a) and (18b)
    psi[0] = np.sin(z1)                 # Eq. (18a)
    zeta[0] = -1j * np.exp(1j * z1)
    psiZeta[0] = 0.5 * (1. - np.exp(2j * z1))
    d3[0] = 1j
    for n in range(1, nmax+1):          # Eq. (18b)
        psi[n] = psi[n-1] * (n/z1 - d1[n-1])
        zeta[n] = zeta[n-1] * (n/z1 - d3[n-1])
        psiZeta[n] = psiZeta[n-1] * (n/z1 - d1[n-1]) * (n/z1 - d3[n-1])
        d3[n] = d1[n] + 1j/psiZeta[n]

    # Scattering coefficients, Eqs. (5) and (6)
    n = np.arange(nmax+1)
    ab[:, 0] = (ha[:, -1]/m[-1] + n/x[-1]) * psi - np.roll(psi,  1)  # Eq. (5)
    ab[:, 0] /= (ha[:, -1]/m[-1] + n/x[-1]) * zeta - np.roll(zeta, 1)
    ab[:, 1] = (hb[:, -1]*m[-1] + n/x[-1]) * psi - np.roll(psi,  1)  # Eq. (6)
    ab[:, 1] /= (hb[:, -1]*m[-1] + n/x[-1]) * zeta - np.roll(zeta, 1)
    ab[0, :] = complex(0., 0.)
    if (resolution > 0):
        w = abs(ab).sum(axis=1)
        ab = ab[(w > resolution), :]

    return ab
