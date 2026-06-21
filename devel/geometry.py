import numpy as np

class CartesianCoordinates(object):
    def __init__(self, nx, ny, origin=[0.,0.], scale=[1.,1.], mask=None):
        '''
        Encapsulate the method and attributes associated with a cartesian coordinate
        system.
        '''
        self.origin = origin
        self.xx, self.yy = self._compute_coords(nx, ny, origin, scale)
        self.shape = self.xx.shape

    def _compute_coords(self, nx, ny, origin, scale):
        '''Initializes coordinate system by translation THEN scaling (the order matters).'''
        xy = np.ogrid[0.:nx, 0.:ny]

        # Translate.
        x = xy[0] - origin[0]
        y = xy[1] - origin[1]
        
        # Scale.
        x *= scale[0]
        y *= scale[1]
        
        return np.meshgrid(x, y)

    def scale(self, factor = 1, x_factor = 1, y_factor = 1):
        '''Scales the coordinates by an amount factor.'''
        self.xx *= factor*x_factor
        self.yy *= factor*y_factor
    
    def translate(self, x_shift, y_shift):
        '''Translate the coordinate system by an amount (x_shift, y_shift).'''
        self.xx -= x_shift
        self.yy -= y_shift
        self.origin -= (x_shift, y_shift)

    def extrema(self):
        return (self.xx.min(), self.yy.min(), self.xx.max(), self.yy.max())
    
    def acquire_spherical(self, z):
        xx = self.xx
        yy = self.yy

        # convert to spherical coordinates centered on the sphere.
        # (r, theta, phi) is the spherical coordinate of the pixel
        # at (x,y) in the imaging plane at distance z from the
        # center of the sphere.
        self.z = z
        self.rho   = np.sqrt(xx**2 + yy**2)
        self.r     = np.sqrt(self.rho**2 + z**2)
        theta = np.arctan2(self.rho, z)
        phi   = np.arctan2(yy, xx)
        self.costheta = np.cos(theta)
        self.sintheta = np.sin(theta)
        self.cosphi   = np.cos(phi)
        self.sinphi   = np.sin(phi)
    
class VectorField(object):
    def __init__(self, coordinates, dim):
        self.coords = coordinates
        self.dim = dim
        self.shape = (dim, coordinates.shape[0], coordinates.shape[1])

    def evaluate_field(self, function):
        '''
        Evaluate field over coordinates.
        '''
        pass


class SphericalVectorField(VectorField):
    def __init__(self, coordinates, dim):
        VectorField.__init__(self, coordinates, dim)
    

def spherical_to_cartesian(es_cam, geom):
    sintheta = geom.sintheta
    costheta = geom.costheta
    sinphi = geom.sinphi
    cosphi = geom.cosphi

    es_cam_cart = np.zeros(es_cam.shape, dtype = complex)
    es_cam_cart += es_cam
    
    es_cam_cart[0,:,:] =  es_cam[0,:,:] * sintheta * cosphi
    es_cam_cart[0,:,:] += es_cam[1,:,:] * costheta * cosphi
    es_cam_cart[0,:,:] -= es_cam[2,:,:] * sinphi
    
    es_cam_cart[1,:,:] =  es_cam[0,:,:] * sintheta * sinphi
    es_cam_cart[1,:,:] += es_cam[1,:,:] * costheta * sinphi
    es_cam_cart[1,:,:] += es_cam[2,:,:] * cosphi
    
    es_cam_cart[2,:,:] =  es_cam[0,:,:] * costheta - es_cam[1,:,:] * sintheta
    
    return es_cam_cart

def test():
    xy = np.ogrid[0:10,0:10]
    origin = [0.0, 0.0]
    cart = CartesianCoordinates(xy, origin, units = [1.0, 'um'])
    
    x = np.tile(np.arange(10, dtype = float), 10)
    y = np.repeat(np.arange(10, dtype = float), 10)

def test_spherical():
    
    # Parameters
    nm_obj = 1.339
    nm_img = 1.339 
    NA = 1.45 
    mpp = 0.135 
    M = 1.

    # Arbitrary size of image
    p, q = 20, 20

    # Compute the two geometries, s_img, s_obj, n_img
    # Origins for the coordinate systems.
    origin = [.5*(p-1.), .5*(q-1.)]
    
    # Scale factors for the coordinate systems.
    img_factor = 2*NA/(M*nm_img)
    obj_factor = 2*NA/nm_obj
    img_scale = [img_factor*1./p, img_factor*1./q]
    obj_scale = [obj_factor*1./p, obj_factor*1./q]

    # Cartesian Geometries.
    # FIXME (MDH): is it necessary to have s_obj_cart?
    s_img_cart = CartesianCoordinates(p, q, origin, img_scale)
    s_obj_cart = CartesianCoordinates(p, q, origin, obj_scale)

    # Spherical geometries.
    s_img_cart.acquire_spherical(1.)
    s_obj_cart.acquire_spherical(1.)

    # Plot the cartesian quantities
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(s_img_cart.xx.ravel(), s_img_cart.yy.ravel(), color = 'r')
    ax.scatter(s_obj_cart.xx.ravel(), s_obj_cart.yy.ravel(), color = 'b')
    plt.show()

    # Plot the spherical quantities
    fig, ax = plt.subplots()
    ax.scatter(s_img_cart.costheta, s_img_cart.sintheta.ravel(), color = 'b')
    ax.scatter(s_obj_cart.costheta, s_obj_cart.sintheta.ravel(), color = 'r')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    plt.plot(s_obj_cart.costheta)
    plt.show()

if __name__ == '__main__':
    test_spherical()
