'''Compute scattered light field with CUDA acceleration using PyTorch.

Inherits LorenzMie numpy base class.
'''

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.lib.lmtypes import Image
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tl_libdevice
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# Triton CUDA kernel: computes the Lorenz-Mie scattered field for a
# single particle over npts pixels in parallel.  The entire Mie sum
# runs inside one kernel to avoid per-order launch overhead.
@triton.jit
def _lorenzmie_kernel(
        kx_ptr, ky_ptr, kz_ptr,           # [npts] float32 — k-scaled displacements
        a_re_ptr, a_im_ptr,               # [norders] float32 — Mie a coefficients
        b_re_ptr, b_im_ptr,               # [norders] float32 — Mie b coefficients
        norders: tl.constexpr,
        e1_re_ptr, e1_im_ptr,             # [npts] float32 — output field component 1
        e2_re_ptr, e2_im_ptr,
        e3_re_ptr, e3_im_ptr,
        npts,
        bohren: tl.constexpr,
        cartesian: tl.constexpr,
        BLOCK: tl.constexpr):

    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < npts

    kx = tl.load(kx_ptr + idx, mask=mask, other=0.0)
    ky = tl.load(ky_ptr + idx, mask=mask, other=0.0)
    kz = -tl.load(kz_ptr + idx, mask=mask, other=0.0)

    krho      = tl.sqrt(kx*kx + ky*ky)
    kr        = tl.sqrt(krho*krho + kz*kz)
    phi       = tl_libdevice.atan2(ky, kx)
    cos_phi   = tl.cos(phi)
    sin_phi   = tl.sin(phi)
    theta     = tl_libdevice.atan2(krho, kz)
    cos_theta = tl.cos(theta)
    sin_theta = tl.sin(theta)
    sinkr     = tl.sin(kr)
    coskr     = tl.cos(kr)

    sign_kz = tl.where(kz > 0, 1.0, tl.where(kz < 0, -1.0, 0.0))
    if bohren:
        factor_im = sign_kz
    else:
        factor_im = -sign_kz

    xi_nm2_re = coskr
    xi_nm2_im = factor_im * sinkr
    xi_nm1_re = sinkr
    xi_nm1_im = -factor_im * coskr

    pi_nm1 = tl.zeros([BLOCK], dtype=tl.float32)
    pi_n   = tl.full([BLOCK], 1.0, dtype=tl.float32)

    esr_re = tl.zeros([BLOCK], dtype=tl.float32)
    esr_im = tl.zeros([BLOCK], dtype=tl.float32)
    est_re = tl.zeros([BLOCK], dtype=tl.float32)
    est_im = tl.zeros([BLOCK], dtype=tl.float32)
    esp_re = tl.zeros([BLOCK], dtype=tl.float32)
    esp_im = tl.zeros([BLOCK], dtype=tl.float32)

    En_i_re = 1.0
    En_i_im = 0.0

    for n in range(1, norders):
        En_i_re, En_i_im = -En_i_im, En_i_re

        swisc = pi_n * cos_theta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc

        c       = (2.0 * n - 1.0) / kr
        xi_n_re = c * xi_nm1_re - xi_nm2_re
        xi_n_im = c * xi_nm1_im - xi_nm2_im

        c2    = n / kr
        Dn_re = c2 * xi_n_re - xi_nm1_re
        Dn_im = c2 * xi_n_im - xi_nm1_im

        En_mag = (2.0 * n + 1.0) / (n * n + n)
        En_re  = En_i_re * En_mag
        En_im  = En_i_im * En_mag

        a_re = tl.load(a_re_ptr + n)
        a_im = tl.load(a_im_ptr + n)
        b_re = tl.load(b_re_ptr + n)
        b_im = tl.load(b_im_ptr + n)

        En_a_re = En_re * a_re - En_im * a_im
        En_a_im = En_re * a_im + En_im * a_re
        En_b_re = En_re * b_re - En_im * b_im
        En_b_im = En_re * b_im + En_im * b_re

        iEn_a_re = -En_a_im
        iEn_a_im =  En_a_re

        pi_xi_re  = pi_n  * xi_n_re
        pi_xi_im  = pi_n  * xi_n_im
        tau_xi_re = tau_n * xi_n_re
        tau_xi_im = tau_n * xi_n_im
        tau_Dn_re = tau_n * Dn_re
        tau_Dn_im = tau_n * Dn_im
        pi_Dn_re  = pi_n  * Dn_re
        pi_Dn_im  = pi_n  * Dn_im

        nn = n * n + n

        esr_re += iEn_a_re * nn * pi_xi_re - iEn_a_im * nn * pi_xi_im
        esr_im += iEn_a_re * nn * pi_xi_im + iEn_a_im * nn * pi_xi_re

        est_re += (iEn_a_re * tau_Dn_re - iEn_a_im * tau_Dn_im
                   - En_b_re * pi_xi_re + En_b_im * pi_xi_im)
        est_im += (iEn_a_re * tau_Dn_im + iEn_a_im * tau_Dn_re
                   - En_b_re * pi_xi_im - En_b_im * pi_xi_re)

        esp_re += (iEn_a_re * pi_Dn_re - iEn_a_im * pi_Dn_im
                   - En_b_re * tau_xi_re + En_b_im * tau_xi_im)
        esp_im += (iEn_a_re * pi_Dn_im + iEn_a_im * pi_Dn_re
                   - En_b_re * tau_xi_im - En_b_im * tau_xi_re)

        pi_nm1 = pi_n
        pi_n   = swisc + (1.0 + 1.0 / n) * twisc

        xi_nm2_re, xi_nm1_re = xi_nm1_re, xi_n_re
        xi_nm2_im, xi_nm1_im = xi_nm1_im, xi_n_im

    inv_kr  = 1.0 / kr
    esr_re *= cos_phi * sin_theta * inv_kr * inv_kr
    esr_im *= cos_phi * sin_theta * inv_kr * inv_kr
    est_re *= cos_phi * inv_kr
    est_im *= cos_phi * inv_kr
    esp_re *= sin_phi * inv_kr
    esp_im *= sin_phi * inv_kr

    if cartesian:
        ec1_re = (esr_re * sin_theta * cos_phi
                  + est_re * cos_theta * cos_phi
                  - esp_re * sin_phi)
        ec1_im = (esr_im * sin_theta * cos_phi
                  + est_im * cos_theta * cos_phi
                  - esp_im * sin_phi)
        ec2_re = (esr_re * sin_theta * sin_phi
                  + est_re * cos_theta * sin_phi
                  + esp_re * cos_phi)
        ec2_im = (esr_im * sin_theta * sin_phi
                  + est_im * cos_theta * sin_phi
                  + esp_im * cos_phi)
        ec3_re = esr_re * cos_theta - est_re * sin_theta
        ec3_im = esr_im * cos_theta - est_im * sin_theta
        tl.store(e1_re_ptr + idx, ec1_re, mask=mask)
        tl.store(e1_im_ptr + idx, ec1_im, mask=mask)
        tl.store(e2_re_ptr + idx, ec2_re, mask=mask)
        tl.store(e2_im_ptr + idx, ec2_im, mask=mask)
        tl.store(e3_re_ptr + idx, ec3_re, mask=mask)
        tl.store(e3_im_ptr + idx, ec3_im, mask=mask)
    else:
        tl.store(e1_re_ptr + idx, esr_re, mask=mask)
        tl.store(e1_im_ptr + idx, esr_im, mask=mask)
        tl.store(e2_re_ptr + idx, est_re, mask=mask)
        tl.store(e2_im_ptr + idx, est_im, mask=mask)
        tl.store(e3_re_ptr + idx, esp_re, mask=mask)
        tl.store(e3_im_ptr + idx, esp_im, mask=mask)


def get_device() -> torch.device:
    '''Return the best available compute device.

    Returns
    -------
    device : torch.device
        ``cuda`` if a working GPU is available, otherwise ``cpu``.
    '''
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            logger.info('Using GPU acceleration')
            return torch.device('cuda')
        except Exception as e:
            logger.warning(
                f'CUDA available but failed: {e}. Falling back to CPU.')
    return torch.device('cpu')


class TorchLorenzMie(LorenzMie):
    '''GPU-accelerated Lorenz-Mie hologram calculator using Triton.

    Overrides :meth:`lorenzmie` with a single Triton kernel that runs
    the full Mie partial-wave sum on the GPU.  All other behavior
    (coordinate handling, multi-particle accumulation, properties
    interface) is inherited from :class:`LorenzMie`.

    Attributes
    ----------
    device : torch.device
        Compute device in use (``cuda`` or ``cpu``).
    '''

    method: str = 'torch'

    def __init__(self,
                 *args,
                 device: torch.device | None = None,
                 **kwargs) -> None:
        self._device = device if device is not None else get_device()
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        '''Compute device in use.'''
        return self._device

    def _allocate(self) -> None:
        '''Allocate GPU tensors sized to the current coordinate grid.'''
        logger.debug('Allocating torch buffers')
        shape = self.coordinates.shape
        npts = shape[1]
        self._coords   = torch.tensor(
            self.coordinates, dtype=torch.float32, device=self._device)
        self._field_t  = torch.zeros(
            shape, dtype=torch.complex64, device=self._device)
        self._tri_e_re = torch.zeros(
            3, npts, dtype=torch.float32, device=self._device)
        self._tri_e_im = torch.zeros(
            3, npts, dtype=torch.float32, device=self._device)

    def hologram(self,
                 cartesian: bool = True,
                 bohren: bool = True) -> Image:
        '''Hologram of the particle at the current coordinates.

        Parameters
        ----------
        cartesian : bool
            If True (default), project the field onto Cartesian
            coordinates. If False, return the field in spherical
            polar coordinates.
        bohren : bool
            If True (default), use +z along the propagation direction.
            If False, flip the orientation of z.

        Returns
        -------
        hologram : numpy.ndarray, shape (npts,)
            Computed hologram intensity at each coordinate.
        '''
        field = self.field(cartesian=cartesian, bohren=bohren)
        intensity = (field.real**2 + field.imag**2).sum(dim=0)
        return (intensity + 2.*field[0].real + 1.).cpu().numpy()

    def field(self,
              cartesian: bool = True,
              bohren: bool = True) -> torch.Tensor:
        '''Electric field scattered by the particle(s).

        Parameters
        ----------
        cartesian : bool
            If True (default), project the field onto Cartesian
            coordinates. If False, return the field in spherical
            polar coordinates.
        bohren : bool
            If True (default), use +z along the propagation direction.
            If False, flip the orientation of z.

        Returns
        -------
        field : torch.Tensor, shape (3, npts), dtype complex64
            Complex electric field scattered by the particle(s).
        '''
        k = float(self.instrument.wavenumber())
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength

        self._field_t.zero_()

        for particle in self.particle:
            r_p = torch.tensor(
                particle.r_p + particle.r_0,
                dtype=torch.float32,
                device=self._device)
            ab = torch.tensor(
                particle.ab(n_m, wavelength),
                dtype=torch.complex64,
                device=self._device)

            dr = self._coords - r_p[:, None]
            kdr = k * dr

            particle_field = self.lorenzmie(
                ab, kdr, cartesian=cartesian, bohren=bohren)

            phase = torch.exp(torch.tensor(
                -1j * k * float(r_p[2]),
                dtype=torch.complex64,
                device=self._device))
            self._field_t += particle_field * phase

        return self._field_t

    def lorenzmie(self,
                  ab: torch.Tensor,
                  kdr: torch.Tensor,
                  cartesian: bool = True,
                  bohren: bool = True) -> torch.Tensor:
        '''Scattered field for given Mie coefficients and geometry.

        Parameters
        ----------
        ab : torch.Tensor, shape (norders, 2), dtype complex64
            Mie scattering coefficients.
        kdr : torch.Tensor, shape (3, npts)
            Wavenumber-scaled displacement from particle to each
            coordinate point.
        cartesian : bool
            If True, return field projected onto Cartesian coordinates.
            Default: True.
        bohren : bool
            If True, use sign convention from Bohren and Huffman.
            Default: True.

        Returns
        -------
        field : torch.Tensor, shape (3, npts), dtype complex64
            Complex scattered field at each coordinate.
        '''
        norders = ab.shape[0]
        npts    = kdr.shape[1]

        a = ab[:, 0].contiguous()
        b = ab[:, 1].contiguous()

        BLOCK = 512
        grid  = (triton.cdiv(npts, BLOCK),)
        _lorenzmie_kernel[grid](
            kdr[0].contiguous(), kdr[1].contiguous(), kdr[2].contiguous(),
            a.real.contiguous(), a.imag.contiguous(),
            b.real.contiguous(), b.imag.contiguous(),
            norders,
            self._tri_e_re[0], self._tri_e_im[0],
            self._tri_e_re[1], self._tri_e_im[1],
            self._tri_e_re[2], self._tri_e_im[2],
            npts,
            bohren=bohren,
            cartesian=cartesian,
            BLOCK=BLOCK,
        )
        return torch.complex(self._tri_e_re, self._tri_e_im)


if __name__ == '__main__':  # pragma: no cover
    TorchLorenzMie.example()
