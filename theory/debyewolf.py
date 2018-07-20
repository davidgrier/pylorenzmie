import numpy as np
from lorenzmie.theory.sphericalfield import sphericalfield
from lorenzmie.theory.sphere_coefficients import sphere_coefficients
import matplotlib.pyplot as plt
from lorenzmie.utilities import geometry as g
from lorenzmie.utilities import azimedian as azi

def round_to_even(num):
    return int(np.ceil(num/2.)*2)

def map_abs(data):
    #return np.hstack(map(np.abs, data[:]))
    return np.hstack([np.abs(datum) for datum in data])

def verbose(data, title, gray=False, outfile=None, **kwargs):
    plt.imshow(data, **kwargs)
    plt.title(title)
    if gray:
        plt.gray()
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()

def aperture(field, geom, r_max):
    '''Sets field to zero wherever x**2+y**2 >= rmax.'''
    rho_sq = geom.xx**2 + geom.yy**2
    indices = np.where(rho_sq >= r_max**2)
    field[:, indices] = 0

    return field

def discretize_plan(NA, M, lamb, nm_img, mpp):
    '''Discretizes a plane according to Eq. 130 - 131.'''

    # Suppose the largest scatterer we consider is 20 lambda. Then
    # P should be larger than 40*NA.
    diam = 200 # wavelengths

    p, q = int(2*diam*NA), int(2*diam*NA) # FIXME (MDH): should you add nm_obj?

    # Pad with zeros to help dealias and to approximate mpp_r to mpp.
    pad_p = max([int((lamb - mpp*2*NA)/(mpp*2*NA)*p), 0])
    pad_q = max([int((lamb - mpp*2*NA)/(mpp*2*NA)*q), 0])

    Np = p + pad_p
    Nq = q + pad_q

    # Compute the real mpp.
    mpp_r = lamb*p/(2*NA*Np) # FIXME (MDH): Will this result in mpp_x, mpp_y?

    return mpp_r, Np, Nq, p, q

def propagate_plane_wave(amplitude, k, path_len, shape):
    '''Propagates a plane with wavenumber k through a distance path_len. 
    The wave is polarized in the x direction. The field is given as a 
    cartesian vector field.'''
    e_inc = np.zeros(shape, dtype = complex)
    e_inc[0, :, :] += amplitude*np.exp(1.j * k * path_len)
    return e_inc

def scatter(s_obj_cart, a_p, n_p, nm, lamb, r, mpp):
    '''Compute the angular spectrum arriving at the entrance pupil.'''
    
    p, q = s_obj_cart.shape

    lamb_m = lamb/np.real(nm)/mpp
    ab = sphere_coefficients(a_p, n_p, nm, lamb)
    sx = s_obj_cart.xx.ravel()
    sy = s_obj_cart.yy.ravel()
    rho_sq = sx**2 + sy**2
    inds = rho_sq < 1.

    costheta = np.zeros(sx.shape)
    costheta[inds] = np.sqrt(1. - rho_sq[inds])

    # Compute the electromagnetic strength factor on the object side 
    # (Eq 40 Ref[1]).
    ang_spec = np.zeros((3,p*q), dtype = complex)
    ang_spec[:, inds] = sphericalfield(sx[inds]*r, sy[inds]*r, costheta[inds]*r,
                                       ab, lamb_m, cartesian=False, 
                                       str_factor=True)

    return ang_spec.reshape(3, p, q)

def displacement(geom, z, k):
    '''Returns the displacement phase accumulated by an angular spectrum
    propagating a distance z. 
    
    Ref[2]: J. Goodman, Introduction to Fourier Optics, 2nd Edition, 1996
            [See 3.10.2 Propagation of the Angular Spectrum]
    '''

    rho_sq = geom.xx**2 + geom.yy**2
    inside = rho_sq < 1.
    disp = np.zeros(geom.xx.shape, dtype = complex)
    disp[inside] = np.exp(1.0j * k * z * np.sqrt( 1. - rho_sq[inside]))
    return disp

def collection(es, s_obj, s_img, nm_obj, nm_img, M):
    '''Modulates the electric field strength factor as it passes through
    a microscope. See Eq. 108 of Ref. 1.
    '''
    return es*-M*np.sqrt(nm_img*s_img.costheta/(nm_obj*s_obj.costheta))

def refocus(es_img, s_img, n_disc_grid, p, q, Np, Nq, NA, M, lamb, nm_img,
            aber=None):
    '''Propagates the electric field from the exit pupil to the image plane.'''

    if aber is None:
        aber  = np.zeros([3, p, q], complex) 

    # Compute auxiliary (Eq. 133) with zero padding!
    # The lower dimensional s_img.costheta broadcasts to es_img.
    g_aux = es_img * np.exp(-1.j*2*np.pi*aber/lamb) / s_img.costheta 

    # Apply discrete Fourier Transform (Eq. 135).
    es_m_n = np.fft.ifft2(g_aux, s=(Np,Nq))*(Np*Nq)
    es_m_n = np.fft.fftshift(es_m_n, axes = (1,2))

    # Compute the electric field at plane 3.
    # Eq. 139 of reference.
    es_cam  = (-1.j*NA**2/(M**2*lamb*nm_img))*(4./(p*q))*es_m_n

    # Accounting for aliasing.
    mm, nn = n_disc_grid.xx, n_disc_grid.yy
    es_cam *= np.exp(-1.j*np.pi*( mm*(p-1.)/Np + nn*(q-1.)/Nq))

    return es_cam

def image_formation(es_cam, e_inc_cam):
    '''Produces an image from the electric fields present.'''
    fields = es_cam + e_inc_cam
    image = np.sum(np.real(fields*np.conjugate(fields)), axis = 0)
    return image

def propagate_ang_spec_microscope(ang_spec, s_obj_cart, s_img_cart, nm_obj, 
                                  nm_img, M, n_disc_grid, p, q, Np, Nq, NA, lamb,
                                  mpp, quiet=True):
    """Propagate angular spectrum through microscope to get field in camera."""

    # Compute the electric field strength factor leaving the tube lens.
    es_img = collection(ang_spec, s_obj_cart, s_img_cart, nm_obj, nm_img, M)

    if not quiet:
        verbose(map_abs(es_img), r'After Collection ($r$, $\theta$, $\phi$)')

    # Input the electric field strength into the debye-wolf formalism to
    # compute the scattered field at the camera plane.
    es_img = g.spherical_to_cartesian(es_img, s_img_cart)

    if not quiet:
        verbose(map_abs(es_img), r'Before Refocusing $(x, y, z)$')

    es_cam = refocus(es_img, s_img_cart, n_disc_grid, p, q, Np, Nq, NA, M,
                     lamb/mpp, nm_img)

    if not quiet:
        verbose(map_abs(es_cam), r'After Refocusing $(x, y, z)$')

    return es_cam

def incident_field_camera_plane(nm, nm_obj, nm_img, lamb, mpp, NA, M, z):
    """Calculate the incident field in the camera plane."""
    
    # Devise a discretization plan.
    mpp_r, Np, Nq, p, q = discretize_plan(NA, M, lamb, nm_img, mpp)

    # Necessary constants.
    k_med = 2*np.pi*nm*mpp_r/lamb # [pix**-1]

    z *= mpp/mpp_r

    # Propagate the incident field to the camera plane.
    e_inc = propagate_plane_wave(-1.0 / M * np.sqrt(nm_obj/nm_img), k_med, z, (3, Np, Nq))

    return e_inc

def particle_field_camera_plane(z, a_p, n_p, nm, nm_obj=1.339, nm_img=1.0, NA=1.45,
                       lamb=0.447, mpp=0.135, M=100, f=2.E5, dim=[201,201],
                       quiet=True):
    """Calculate the field of a scattering particle in the camera plane."""

    # Devise a discretization plan.
    mpp_r, Np, Nq, p, q = discretize_plan(NA, M, lamb, nm_img, mpp)
    z *= mpp/mpp_r # TOTAL HACK FIXME (MDH)
    mpp = mpp_r

    # Necessary constants.
    k = 2*np.pi*nm*mpp/lamb
    # k_img = 2*np.pi*nm_img/lamb*mpp # [pix**-1]
    # k_obj = 2 * np.pi * nm_obj * mpp / lamb   # [pix**-1]
    r_max = 1000.  # [pix]

    # Compute the three geometries, s_img, s_obj, n_img
    # Origins for the coordinate systems.
    origin = [.5*(p-1.), .5*(q-1.)]
    
    # Scale factors for the coordinate systems.
    img_factor = 2*NA/(M*nm_img)
    obj_factor = 2*NA/nm_obj
    img_scale = [img_factor*1./p, img_factor*1./q]
    obj_scale = [obj_factor*1./p, obj_factor*1./q]

    # Cartesian Geometries.
    # FIXME (MDH): is it necessary to have s_obj_cart?
    s_img_cart = g.CartesianCoordinates(p, q, origin, img_scale)
    s_obj_cart = g.CartesianCoordinates(p, q, origin, obj_scale)
    n_disc_grid = g.CartesianCoordinates(Np, Nq, origin=[.5*(Np-1), .5*(Nq-1.)])
    #n_disc_grid = g.CartesianCoordinates(Np, Nq, origin=[0, 0])
    

    # Spherical Geometries.
    s_obj_cart.acquire_spherical(1.)
    s_img_cart.acquire_spherical(1.)

    # Compute the angular spectrum incident on entrance pupil of the objective.
    ang_spec = scatter(s_obj_cart, a_p, n_p, nm, lamb, r_max, mpp)

    if not quiet:
        verbose(map_abs(ang_spec), r'After Scatter $(r,\theta,\phi)$')

    # Propagate the angular spectrum a distance z_p.
    disp = displacement(s_obj_cart, z, k)
    ang_spec[1:, :] *= disp

    if not quiet:
        verbose(np.real(disp), r'Displacement Field')

    # Collection and refocus.
    es_cam = propagate_ang_spec_microscope(ang_spec, s_obj_cart, s_img_cart, 
                                           nm_obj, nm_img, M, n_disc_grid,
                                           p, q, Np, Nq, NA, lamb, mpp, 
                                           quiet=True)
    return es_cam


def image_camera_plane(z, a_p, n_p, nm, nm_obj=1.5, nm_img=1.0, NA=1.45, 
                       lamb=0.447, mpp=0.135, M=100, dim=None, 
                       quiet=True):
    '''
    Returns an image in the camera plane due to a spherical scatterer with 
    radius a_p and refractive index n_p at a height z above the focal plane. 

    Args:
        z:     [pix] scatterer's distance from the focal plane.
        a_p:   [um] sets the radius of the spherical scatterer.
        n_p:   [unitless] sets the refractive index of the scatterer.
        nm:    sets the refractive index of the medium immersing the 
               scatterer.
               Default: 1.339 (Water)
        nm_obj:[unitless] sets the refractive index of the medium immersing the
               objective.
               Default: 1.5 (Immersion Oil)
        nm_img:[unitless] sets the refractive index of the medium immersing the 
               camera.
               Default: 1.00 (Air)
        NA:    [unitless] The numerical aperture of the optical train.
               Default: 1.45 (100x Nikon Lambda Series)
        lamb:  [um] wavelength of the incident illumination.
               Default: 0.447 (Coherent Cube.. blue)
        mpp:   [um/pix] sets the size of a pixel.
               Default: 0.135.
        M:     [unitless] Magnification of the optical train.
`              Default: 100
        dim:   [nx, ny]: crop the resulting image to size [nx, ny].
               Default: None. No croppping done.

    Return:
        image: [?, ?] - Currently dim is not implemented. The resulting image 
               size is dictated by the padding chose for the fourier transform.

    Ref[1]: Capoglu et al. (2012). "The Microscope in a Computer:...", 
               Applied Optics, 38(34), 7085.
    '''

    e_inc = incident_field_camera_plane(nm, nm_obj, nm_img, lamb, mpp, NA, M, z)

    es_cam = particle_field_camera_plane(z, a_p, n_p, nm, nm_obj=nm_obj, 
                                         nm_img=nm_img, NA=NA,
                                         lamb=lamb, mpp=mpp, M=M,
                                         quiet=quiet)*-1

    image = image_formation(es_cam, e_inc)

    if dim is not None:
        xc, yc = list(map(lambda x:x//2, image.shape))
        dim = list(map(lambda x:x//2, dim))
        image = image[xc-dim[0]:xc+dim[0], yc-dim[1]:yc+dim[1]]

    return image

def test_image(z=10.0, quiet=False):
    from spheredhm import spheredhm

    # Necessary parameters.
    a_p = 0.5
    n_p = 1.59
    nm = 1.339
    NA = 1.339
    lamb = 0.447
    dim = [400,400] # FIXME: Does nothing.
    nm_obj = 1.339
    nm_img = 1.339
    M = 1
    mpp = 0.135
    
    # Produce image with Debye-Wolf Formalism.
    cam_image = image_camera_plane(z/mpp, a_p, n_p, nm, nm_obj=nm_obj, 
                                   nm_img=nm_img,  NA=NA, lamb=lamb, 
                                   mpp=mpp, M=M, dim=dim, 
                                   quiet=quiet)

    # Produce image in the focal plane.
    dim = cam_image.shape
    image = spheredhm([-0.,-0., z/mpp], a_p, n_p, nm_obj, dim, mpp, lamb)

    # Visually compare the two.
    cam_image *= M**2*nm_img/nm_obj
    diff = cam_image - image
    print("Maximum difference between two images: {}".format(np.max(diff)))
    verbose(np.hstack([cam_image, image, diff+1]), 
            r'Camera Plane Image, Focal Plane Image and their Difference.', 
            gray=True)

    # Plot the radii.
    cam_rad = azi.azimedian(cam_image)
    focal_rad = azi.azimedian(image) 
    # FIXME (MDH): center is not correct...

    end = 150
    plt.plot(cam_rad[:end], 'r', label = 'Camera Plane')
    plt.plot(focal_rad[:end], 'black', label = 'Focal Plane')
    plt.xlabel('Radial distance [pix]')
    plt.ylabel('Normalized Intensity [arb]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true', help='If set only plot last figure.')
    parser.add_argument('-z', type=float, help='Height of test particle.', default=10.0)
    args = parser.parse_args()
    test_image(z=args.z, quiet=args.quiet)
