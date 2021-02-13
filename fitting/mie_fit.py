import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylorenzmie.utilities import coordinates
from lmfit import Minimizer, Parameters, report_fit
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except Exception:
    logging.info("Could not load Cuda LMHologram. Falling back to CPU")
    from pylorenzmie.theory.LMHologram import LMHologram


class Mie_Fitter(object):

    '''
    Mie_Fitter provides a method for fitting holographic images to the
    Lorenz-Mie theory of light scattering via the Levenberg-Marquardt
    algorithm.

    Inputs:
        init_params: a dictionary containing the initial values for
            each parameter.

    Attributes:
        p: lmfit parameters relevant to the scattering calculation.
            p['x']: x position [pix]
            p['y']: y position [pix]
            p['z']: z position [pix]
            p['a_p']: radius [um]
            p['n_p']: refractive index of scatterer [unitless]
            p['n_m']: refractive index of medium [unitless]
                (Default: 1.3371 water w/ lambda ~ 447 nm)
            p['mpp']: camera pixel calibration [um/pix]
            p['lamb']: wavelength [um]

        result: lmfit result object (or None before fitting procedure).
            result contains the parameter estimates, standard devs,
            covariances, . (See lmfit result object docs).
    '''
    def __init__(self, init_params, fixed=['n_m', 'mpp', 'lamb'], noise=.05):
        # Instantiate parameters.
        self.__init_params__()

        # Set initial values.
        for name, value in init_params.items():
            self.set_param(name, value)

        # Set parameters which should NOT be varied.
        for name in fixed:
            self.fix_param(name)

        # Set noise estimate
        self.noise = noise
        
        self.result = None

    def __init_params__(self):
        self.p = Parameters()
        params = ['x', 'y', 'z', 'a_p', 'n_p', 'n_m', 'mpp', 'lamb']
        for p in params:
            self.p.add(p)

    def set_param(self, name, value):
        '''Set parameter 'name' to 'value'
        '''
        self.p[name].value = value

    def fix_param(self, name, choice=False):
        '''Fix parameter 'name' to not vary during fitting'''
        self.p[name].vary = choice

    def mie_loss(self, params, image, dim):
        '''Returns the residual between the image and our Mie model.'''
        p = params.valuesdict()
        h = LMHologram(coordinates=coordinates(dim))
        h.particle.r_p = [p['x'] + dim[0] // 2, p['y'] + dim[1] // 2, p['z']]
        h.particle.a_p = p['a_p']
        h.particle.n_p = p['n_p']
        h.instrument.wavelength = p['lamb']
        h.instrument.magnification = p['mpp']
        h.instrument.n_m = p['n_m']
        hologram = h.hologram().reshape(dim)
        return (hologram - image) / self.noise

    def fit(self, image):
        '''Fit a image of a hologram with the current attribute 
        parameters.

        Example:
        >>> p = {'x':0, 'y':0, 'z':100, 'a_p':0.5, 'n_p':1.5, 'n_m':1.337, 
        ...      'mpp':0.135, 'lamb':0.447}
        >>> mie_fit = Mie_Fitter(p)
        >>> mit_fit.result(image)
        '''
        dim = image.shape
        minner = Minimizer(self.mie_loss, self.p, fcn_args=(image, dim))
        self.result = minner.minimize()
        return self.result

      
def example():
    '''
    Make a "noisy" hologram. Then fit the noisy hologram. Plot the results.
    '''
    ## Make Noisy Hologram.
    # Create hologram to be fitted.
    from time import time
    x,y,z = 0., 0., 100.
    a_p = 0.5
    n_p = 1.5
    n_m = 1.339
    dim = [201,201]
    lamb = 0.447
    mpp = 0.048
    h = LMHologram(coordinates=coordinates(dim))
    h.particle.r_p = [x + dim[0] // 2, y + dim[1] // 2, z]
    h.particle.a_p = a_p
    h.particle.n_p = n_p
    h.instrument.wavelength = lamb
    h.instrument.magnification = mpp
    h.instrument.n_m = n_m
    hologram = h.hologram().reshape(dim)
    
    # Add noise.
    std = 0.05
    noise = np.random.normal(size=hologram.shape)*std
    noisy_hologram = hologram + noise

    # Fit the noisy hologram.
    init_params = {'x':x, 'y':y, 'z':z+8, 'a_p':a_p-.3, 'n_p':n_p+.03, 'n_m':n_m,
                   'mpp':mpp, 'lamb':lamb}
    mie_fit = Mie_Fitter(init_params)
    t = time()
    result = mie_fit.fit(noisy_hologram)
    print("Time to fit: {:.05f}".format(time()-t))

    # Calculate the resulting image.
    residual = result.residual.reshape(*dim)
    final = hologram

    # Write error report.
    report_fit(result)

    ## Make plots.
    # Plot images.
    sns.set(style='white', font_scale=1.4)
    plt.imshow(np.hstack([noisy_hologram, final, residual+1]))
    plt.title('Image, Fit, Residual')
    plt.gray()
    plt.show()

    # Plot Covariance.
    f, ax = plt.subplots()
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.set(font_scale=1.5)
    plt.title('Log Covariance Matrix')
    sns.heatmap(np.log(result.covar), cmap='PuBu',
                square=True, cbar_kws={}, ax=ax)
    ax.set_xticklabels(['x', 'y', 'z', r'a$_p$', r'n$_p$'])
    ax.set_yticklabels([r'n$_p$', r'a$_p$', 'z', 'y', 'x'])
    plt.show()

if __name__ == '__main__':
    example()

