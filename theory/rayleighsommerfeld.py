import numpy as np

def hanning(nx, ny):
    """
    Calculates the Hanning Window of size (nx,ny)
    """ 
    if ny <= 0:
        print("Array dimensions must be >= 0")
        raise TypeError
    if nx <= 0:
        print("Array dimensions must be >= 0")
        raise TypeError
    
    row_window = .5*(1-np.cos(2*np.pi*np.arange(0,int(nx))/nx))
    col_window = .5*(1-np.cos(2*np.pi*np.arange(0,int(ny))/ny))
    if ny > 0:
        return np.outer(row_window,col_window)
    else:
        return row_window  

def rayleighsommerfeld(a, z, lamb = 0.447, mpp = 0.135, nozphase = False, 
                       hanning_win = False): 
    """
    Compute electric fields propagated by a distance or set of distances above 
    the imaging plane via Rayleigh-Sommerfeld approximation.

    Args:
        a: A two dimensional intensity array.
        z: displacement(s) from the focal plane [pixels].

    Keywords:
        lamb: Wavelength of light in medium [micrometers].
            Default: 0.447
        mpp: Micrometers per pixel.
            Default: 0.135

    Returns:
        Complex electric fields at a plane or set of planes z.
    """

    # Check if a and z are appropriates types
    if type(a) != np.ndarray:
        print('a must be a numpy array')
        raise TypeError

    if type(z) == int or type(z) == float:
        z = [z]

    if type(z) != list and type(z) != np.ndarray:
        print('z must be an int, float, list or numpy array')
        raise TypeError

    # Check image is 2D
    if a.ndim != 2 : 
        print("a must be two-dimensional hologram")
        raise TypeError
    
    ny, nx = map(float, a.shape)
  
    # A single slice or volumetric slices?
    nz = 1 if type(z) == int else len(z)

    # important factors
    ci = complex(0., 1.)
    k = 2.*np.pi*mpp/lamb      # wavenumber in radians/pixels

    # phase factor for Rayleigh-Sommerfeld propagator in Fourier space
    # Compute factor k*sqrt(1-qx**2+qy**2)
    # (FIXME MDH): Do I need to neglect the endpoint?
    qx = np.linspace(-0.5, 0.5, nx, endpoint = False, dtype = complex)
    qy = np.linspace(-0.5, 0.5, ny, endpoint = False, dtype = complex)
    
    qx, qy = np.meshgrid(qx, qy)
    qsq = qx**2+qy**2
    qsq *= (lamb/mpp)**2 

    qfactor = k * np.sqrt(1. - qsq)

    if nozphase:
        qfactor -= k

    if hanning_win:
        qfactor *= hanning(ny, nx)

    # Account for propagation and absorption
    ikappa = ci * np.real(qfactor)
    gamma = np.imag(qfactor) 
  
    # Go to fourier space and apply RS propagation operator
    a = np.array(a,dtype=complex)
    E = np.fft.ifft2(a-1.) # avg(a-1) should = 0. Better fft accuracy with a-1.
    E = np.fft.fftshift(E)
    res = np.zeros([ny, nx, nz],dtype = complex)
    for j in xrange(0, nz):
        Hqz = np.exp((ikappa * z[j] - gamma * abs(z[j])))
        thisE = E * Hqz                        # convolve with propagator
        thisE = np.fft.ifftshift(thisE)        # Shift Center
        thisE = np.fft.fft2(thisE)             # transform back to real space
        res[:,:,j] = thisE                     # save result
        
    return res + 1. # undo the previous reduction by 1.
