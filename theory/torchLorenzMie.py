''' 
Torch implementation of LorenzMie.py

Author: Sarah Odeh (NYU)

Goal: Given instrument settings and a list of particles with variable parameters position, size, and refractive index, generate a hologram image. 

Plan:
1. lorenzmie(ab,kdr) to generate the field, given Mie coefficients and the displacement coordinates. 
2. field(particles, coordinates) to loop over a list of particles, calling lorenzmie for each of them to calculate the field for each, applies the phase factors, and finally adds all the fields together into one total field
3. hologram(particles, coordinates, k) calls field and adds the incident beam, and then computes the intensity at every pixel to generate a hologram. 
4. LorenzMieBatch to generate batches of holograms with a variable number of particles
'''
import torch
from torchAberrated import spherical_aberration

# automatically detect and return the best available device
def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            print('Using GPU acceleration')
            return torch.device('cuda')
        except Exception as e: 
            print(f'CUDA available but not working: {e}')
            print('Falling back to CPU')
    return torch.device('cpu')

# generate the field (copied from the LorenzMie.lorenzmie() method)
def lorenzmie_field(ab, kdr, cartesian: bool = True, bohren: bool=True):

    device  = kdr.device
    dtype_r = torch.float32
    dtype_c = torch.complex64

    norders = ab.shape[0]
    npts    = kdr.shape[1]

    ab = ab.to(device=device, dtype=dtype_c)

# from the geometry section
    kx =  kdr[0]
    ky =  kdr[1]
    kz = -kdr[2] 

    kp        = torch.hypot(kx, ky)
    kr        = torch.hypot(kp, kz)
    sinkr     = torch.sin(kr)
    coskr     = torch.cos(kr)

    phi       = torch.atan2(ky, kx)
    cos_phi   = torch.cos(phi)
    sin_phi   = torch.sin(phi)
    theta     = torch.atan2(kp, kz)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    sign_kz = torch.sign(kz).to(dtype_c)
   
# from the special features section   
    factor  = (1j * sign_kz) if bohren else (-1j * sign_kz)
   
    sinkr_c = sinkr.to(dtype_c)
    coskr_c = coskr.to(dtype_c)
    xi_nm2 = coskr_c + factor * sinkr_c   # xi_{-1}(kr)
    xi_nm1 = sinkr_c - factor * coskr_c   # xi_0(kr)
   
    pi_nm1 = torch.zeros(npts, dtype=dtype_r, device=device)
    pi_n   = torch.ones( npts, dtype=dtype_r, device=device)

    Es   = torch.zeros(3, npts, dtype=dtype_c, device=device)
    kr_c = kr.to(dtype_c)

# from the field computation section   
    for n in range(1, norders):

        swisc = pi_n * cos_theta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc        
        
        xi_n = (2.*n - 1.) * (xi_nm1 / kr_c) - xi_nm2
        
        Dn = n * (xi_n / kr_c) - xi_nm1

        En = (1j**n) * (2.*n + 1.) / (n*n + n)
        pi_n_c  = pi_n.to(dtype_c)
        tau_n_c = tau_n.to(dtype_c)

        Mo1n_1 = pi_n_c  * xi_n
        Mo1n_2 = tau_n_c * xi_n

        Ne1n_0 = (n*n + n) * pi_n_c * xi_n
        Ne1n_1 = tau_n_c * Dn
        Ne1n_2 = pi_n_c  * Dn

        En_a = En * ab[n, 0]
        En_b = En * ab[n, 1]

        Es[0] += 1j * En_a * Ne1n_0
        Es[1] += 1j * En_a * Ne1n_1 - En_b * Mo1n_1
        Es[2] += 1j * En_a * Ne1n_2 - En_b * Mo1n_2

        pi_nm1 = pi_n
        pi_n   = swisc + (1. + 1./n) * twisc

        xi_nm2 = xi_nm1
        xi_nm1 = xi_n

    cos_phi_c   = cos_phi.to(dtype_c)
    sin_phi_c   = sin_phi.to(dtype_c)
    cos_theta_c = cos_theta.to(dtype_c)
    sin_theta_c = sin_theta.to(dtype_c)

    Es[0] *= cos_phi_c * sin_theta_c / (kr_c * kr_c)  # divide by kr_c^2 UGHHHH
    Es[1] *= cos_phi_c / kr_c
    Es[2] *= sin_phi_c / kr_c

    if not cartesian:
        return Es
    Ec = torch.zeros(3, npts, dtype=dtype_c, device=device)

    Ec[0] = (Es[0] * sin_theta_c * cos_phi_c
           + Es[1] * cos_theta_c * cos_phi_c
           - Es[2] * sin_phi_c)

    Ec[1] = (Es[0] * sin_theta_c * sin_phi_c
           + Es[1] * cos_theta_c * sin_phi_c
           + Es[2] * cos_phi_c)

    Ec[2] = Es[0] * cos_theta_c - Es[1] * sin_theta_c

    return Ec

# computing the total scattered field from all particles with aberration (taken from LorenzMie.field() and AberratedLorenzMie.scattered_field)
def field(particles, coordinates, k, NA, n_m, spherical, cartesian: bool = True, bohren: bool = True):
    
    device      = coordinates.device
    npts        = coordinates.shape[1]
    total_field = torch.zeros(3, npts, dtype=torch.complex64, device=device)

    for particle in particles:
        r_p = particle.r_p.to(device=device, dtype=torch.float32)
        ab  = particle.ab.to(device=device,  dtype=torch.complex64)

        dr  = coordinates - r_p[:, None]   
        kdr = k * dr                       

        particle_field = lorenzmie_field(ab, kdr, cartesian=cartesian, bohren=bohren)

        if spherical != 0.:
            particle_field *= spherical_aberration(coordinates, r_p,
                                                   NA, n_m, spherical)

        phase = torch.exp(torch.tensor(-1j * k * r_p[2].item(),
                                 dtype=torch.complex64, device=device))
        particle_field *= phase
        total_field  += particle_field

    return total_field

# hologram calculations (copied from LorenzMie.hologram())
def hologram(particles, coordinates, k, NA, n_m, spherical, cartesian: bool = True, bohren: bool = True):

    f = field(particles, coordinates, k, NA, n_m, spherical, cartesian=cartesian, bohren=bohren)
    f[0] += 1.0 + 0j                             
    return (f.real**2 + f.imag**2).sum(dim=0) 

# make the coordinate grid that matches the LM.meshgrid convention
def make_coordinates(shape, device = None):
    if device is None:
        device = get_device()
    
    ny, nx = shape
    x       = torch.arange(nx, dtype=torch.float32, device=device)
    y       = torch.arange(ny, dtype=torch.float32, device=device)
    yy, xx  = torch.meshgrid(y, x, indexing='ij')
    zz      = torch.zeros_like(xx)

    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=0)

# batch hologram generator
class LorenzMieBatch:

    def __init__(self, shape, wavelength, n_m, magnification, NA, spherical, device):
        self.shape          = shape
        self.device         = get_device()
        self.wavelength     = wavelength
        self.n_m            = n_m
        self.magnification  = magnification
        self.NA             = NA
        self.spherical      = spherical

        self.k = 2 * torch.pi * n_m / (wavelength / magnification)

        self.coordinates = make_coordinates(shape, device=self.device)

        print(f'LorenzMieBatch running on: {self.device}')

    def hologram(self, particles, cartesian: bool = True, bohren: bool = True):
        return hologram(particles, self.coordinates, self.k, self.NA, self.n_m, self.spherical, cartesian = cartesian, bohren = bohren)

    def batch_holograms(self, particle_lists, cartesian: bool = True, bohren: bool = True): 
        holos = [self.hologram(particles, cartesian = cartesian, bohren = bohren).reshape(self.shape)
            for particles in particle_lists]
        return torch.stack(holos, dim = 0)
