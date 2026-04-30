from pylorenzmie.theory.LorenzMie import LorenzMie
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tl_libdevice
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

@triton.jit
def _lorenzmie_kernel(
        kx_ptr, ky_ptr, kz_ptr,          # [npts] float32 — k-scaled displacements
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

    # initialise xi recurrence (complex, stored as re/im pairs)
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

    # track i^n as a running scalar — compile-time constant when norders is constexpr
    En_i_re = 1.0
    En_i_im = 0.0

    for n in range(1, norders):
        # multiply running i^n by i: (a+bi)*i = (-b, a)
        En_i_re, En_i_im = -En_i_im, En_i_re

        swisc = pi_n * cos_theta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc

        # xi recurrence (kr is real, so dividing complex by real is component-wise)
        c       = (2.0 * n - 1.0) / kr
        xi_n_re = c * xi_nm1_re - xi_nm2_re
        xi_n_im = c * xi_nm1_im - xi_nm2_im

        # Dn = n * xi_n / kr - xi_nm1
        c2    = n / kr
        Dn_re = c2 * xi_n_re - xi_nm1_re
        Dn_im = c2 * xi_n_im - xi_nm1_im

        En_mag = (2.0 * n + 1.0) / (n * n + n)
        En_re  = En_i_re * En_mag
        En_im  = En_i_im * En_mag

        # load ab[n] as scalars — one per Mie order, same for all pixels
        a_re = tl.load(a_re_ptr + n)
        a_im = tl.load(a_im_ptr + n)
        b_re = tl.load(b_re_ptr + n)
        b_im = tl.load(b_im_ptr + n)

        # En_a = En * a[n],  En_b = En * b[n]
        En_a_re = En_re * a_re - En_im * a_im
        En_a_im = En_re * a_im + En_im * a_re
        En_b_re = En_re * b_re - En_im * b_im
        En_b_im = En_re * b_im + En_im * b_re

        # i * En_a
        iEn_a_re = -En_a_im
        iEn_a_im =  En_a_re

        # vector-scalar products (pi_n, tau_n are [BLOCK] float32)
        pi_xi_re  = pi_n  * xi_n_re
        pi_xi_im  = pi_n  * xi_n_im
        tau_xi_re = tau_n * xi_n_re
        tau_xi_im = tau_n * xi_n_im
        tau_Dn_re = tau_n * Dn_re
        tau_Dn_im = tau_n * Dn_im
        pi_Dn_re  = pi_n  * Dn_re
        pi_Dn_im  = pi_n  * Dn_im

        nn = n * n + n

        # esr += i*En_a * nn * pi_xi
        esr_re += iEn_a_re * nn * pi_xi_re - iEn_a_im * nn * pi_xi_im
        esr_im += iEn_a_re * nn * pi_xi_im + iEn_a_im * nn * pi_xi_re

        # est += i*En_a * tau_Dn - En_b * pi_xi
        est_re += (iEn_a_re * tau_Dn_re - iEn_a_im * tau_Dn_im
                   - En_b_re * pi_xi_re + En_b_im * pi_xi_im)
        est_im += (iEn_a_re * tau_Dn_im + iEn_a_im * tau_Dn_re
                   - En_b_re * pi_xi_im - En_b_im * pi_xi_re)

        # esp += i*En_a * pi_Dn - En_b * tau_xi
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


@triton.jit
def _batch_lorenzmie_kernel(
        kx_ptr, ky_ptr, kz_ptr,          # [N, npts] float32 contiguous — batch-major
        a_re_ptr, a_im_ptr,               # [N, norders] float32 contiguous
        b_re_ptr, b_im_ptr,               # [N, norders] float32 contiguous
        norders: tl.constexpr,
        e1_re_ptr, e1_im_ptr,             # [N, npts] float32 output
        e2_re_ptr, e2_im_ptr,
        e3_re_ptr, e3_im_ptr,
        npts,
        bohren: tl.constexpr,
        cartesian: tl.constexpr,
        BLOCK: tl.constexpr):
    pid = tl.program_id(0)   # pixel-block index
    bid = tl.program_id(1)   # batch index

    base = bid * npts
    idx  = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < npts

    kx = tl.load(kx_ptr + base + idx, mask=mask, other=0.0)
    ky = tl.load(ky_ptr + base + idx, mask=mask, other=0.0)
    kz = -tl.load(kz_ptr + base + idx, mask=mask, other=0.0)

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

    ab_base = bid * norders

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

        a_re = tl.load(a_re_ptr + ab_base + n)
        a_im = tl.load(a_im_ptr + ab_base + n)
        b_re = tl.load(b_re_ptr + ab_base + n)
        b_im = tl.load(b_im_ptr + ab_base + n)

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
        tl.store(e1_re_ptr + base + idx, ec1_re, mask=mask)
        tl.store(e1_im_ptr + base + idx, ec1_im, mask=mask)
        tl.store(e2_re_ptr + base + idx, ec2_re, mask=mask)
        tl.store(e2_im_ptr + base + idx, ec2_im, mask=mask)
        tl.store(e3_re_ptr + base + idx, ec3_re, mask=mask)
        tl.store(e3_im_ptr + base + idx, ec3_im, mask=mask)
    else:
        tl.store(e1_re_ptr + base + idx, esr_re, mask=mask)
        tl.store(e1_im_ptr + base + idx, esr_im, mask=mask)
        tl.store(e2_re_ptr + base + idx, est_re, mask=mask)
        tl.store(e2_im_ptr + base + idx, est_im, mask=mask)
        tl.store(e3_re_ptr + base + idx, esp_re, mask=mask)
        tl.store(e3_im_ptr + base + idx, esp_im, mask=mask)


# automatically detect and return the best available device
def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            logger.info('Using GPU acceleration')
            return torch.device('cuda')
        except Exception as e:
            logger.warning(f'CUDA available but failed: {e}. Falling back to CPU.')
    return torch.device('cpu')

# compute scattered light field using pytorch, with optional GPU acceleration
class TorchLorenzMie(LorenzMie):

    method: str = 'torch'

    def __init__(self, *args, device: torch.device | None = None, **kwargs) -> None:
        self._device = device if device is not None else get_device()
        super().__init__(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

# overwriting the following functions to use torch tensors instead of numpy arrays

    def _allocate(self) -> None:
        '''Allocate torch tensors for field computation'''
        logger.debug('Allocating torch buffers')
        shape = self.coordinates.shape
        npts = shape[1]
        self._coords = torch.tensor(
            self.coordinates, dtype=torch.float32, device=self._device)
        self._field_t = torch.zeros(
            shape, dtype=torch.complex64, device=self._device)
        # Pre-allocated lorenzmie() working buffers — reused across calls to
        # avoid repeated GPU memory allocation in the sequential hot path
        self._lm_pi_nm1 = torch.zeros(npts, dtype=torch.float32, device=self._device)
        self._lm_pi_n   = torch.ones(npts, dtype=torch.float32, device=self._device)
        self._lm_swisc  = torch.empty(npts, dtype=torch.float32, device=self._device)
        self._lm_twisc  = torch.empty(npts, dtype=torch.float32, device=self._device)
        self._lm_tau_n  = torch.empty(npts, dtype=torch.float32, device=self._device)
        self._lm_Es     = torch.zeros(3, npts, dtype=torch.complex64, device=self._device)
        self._lm_Ec     = torch.zeros(3, npts, dtype=torch.complex64, device=self._device)
        self._batch_N   = 0
        self._tri_e_re  = torch.zeros(3, npts, dtype=torch.float32, device=self._device)
        self._tri_e_im  = torch.zeros(3, npts, dtype=torch.float32, device=self._device)

    def _allocate_batch(self, N: int) -> None:
        '''Allocate [N, npts] working buffers for _batch_lorenzmie'''
        npts = self._coords.shape[1]
        self._batch_N = N
        d_r, d_c, dev = torch.float32, torch.complex64, self._device
        self._blm_pi_nm1  = torch.zeros(N, npts, dtype=d_r, device=dev)
        self._blm_pi_n    = torch.ones( N, npts, dtype=d_r, device=dev)
        self._blm_swisc   = torch.empty(N, npts, dtype=d_r, device=dev)
        self._blm_twisc   = torch.empty(N, npts, dtype=d_r, device=dev)
        self._blm_tau_n   = torch.empty(N, npts, dtype=d_r, device=dev)
        self._blm_xi_nm2  = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_xi_nm1  = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_xi_n    = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_Dn      = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_pi_n_c  = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_tau_n_c = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_scratch = torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_scratch2= torch.empty(N, npts, dtype=d_c, device=dev)
        self._blm_Es      = torch.zeros(N, 3, npts, dtype=d_c, device=dev)
        self._blm_Ec      = torch.zeros(N, 3, npts, dtype=d_c, device=dev)
        self._blm_tri_e_re = torch.zeros(3, N, npts, dtype=d_r, device=dev)
        self._blm_tri_e_im = torch.zeros(3, N, npts, dtype=d_r, device=dev)

    def hologram(self, cartesian: bool = True, bohren: bool = True):
        field = self.field(cartesian=cartesian, bohren=bohren)
        field[0] += 1.0 + 0j
        return (field.real**2 + field.imag**2).sum(dim=0).cpu().numpy() # convert final output to numpy array 

    def field(self, cartesian: bool = True, bohren: bool = True) -> torch.Tensor:
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
            self._field_t += particle_field
            self._field_t *= phase

        return self._field_t

#    def _scattered_field(self, particle, ab, kdr, cartesian=True, bohren=True):
#        return self.lorenzmie(ab, kdr, cartesian=cartesian, bohren=bohren)

    def lorenzmie(self,
                  ab: torch.Tensor,
                  kdr: torch.Tensor,
                  cartesian: bool = True,
                  bohren: bool = True) -> torch.Tensor:
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

#    def lorenzmie(self,
#                  ab: torch.Tensor,
#                  kdr: torch.Tensor,
#                  cartesian: bool = True,
#                  bohren: bool = True) -> torch.Tensor:
#
#        dtype_r = torch.float32
#        dtype_c = torch.complex64
#
#        norders = ab.shape[0]
#        npts = kdr.shape[1]
#
#        kx = kdr[0]
#        ky = kdr[1]
#        kz = -kdr[2]
#
#        krho = torch.hypot(kx, ky)
#        kr = torch.hypot(krho, kz)
#        sinkr = torch.sin(kr)
#        coskr = torch.cos(kr)
#
#        phi = torch.atan2(ky, kx)
#        cos_phi = torch.cos(phi)
#        sin_phi = torch.sin(phi)
#        theta = torch.atan2(krho, kz)
#        cos_theta = torch.cos(theta)
#        sin_theta = torch.sin(theta)
#
#        sign_kz = torch.sign(kz).to(dtype_c)
#        factor = (1j * sign_kz) if bohren else (-1j * sign_kz)
#
#        sinkr_c = sinkr.to(dtype_c)
#        coskr_c = coskr.to(dtype_c)
#        xi_nm2 = coskr_c + factor * sinkr_c   # xi_{-1}(kr)
#        xi_nm1 = sinkr_c - factor * coskr_c   # xi_0(kr)
#
#        pi_nm1 = self._lm_pi_nm1
#        pi_n   = self._lm_pi_n
#        swisc  = self._lm_swisc
#        twisc  = self._lm_twisc
#        tau_n  = self._lm_tau_n
#        Es     = self._lm_Es
#        pi_nm1.zero_()
#        pi_n.fill_(1.)
#        Es.zero_()
#        kr_c = kr.to(dtype_c)
#
#        for n in range(1, norders):
#
#            torch.mul(pi_n, cos_theta, out=swisc)
#            torch.sub(swisc, pi_nm1, out=twisc)
#            torch.mul(twisc, float(n), out=tau_n)
#            torch.sub(pi_nm1, tau_n, out=tau_n)
#
#            xi_n = (2. * n - 1.) * (xi_nm1 / kr_c) - xi_nm2
#
#            Dn = n * (xi_n / kr_c) - xi_nm1
#
#            En = (1j ** n) * (2. * n + 1.) / (n * n + n)
#
#            pi_n_c = pi_n.to(dtype_c)
#            tau_n_c = tau_n.to(dtype_c)
#
#            Mo1n_1 = pi_n_c * xi_n
#            Mo1n_2 = tau_n_c * xi_n
#            Ne1n_0 = (n * n + n) * pi_n_c * xi_n
#            Ne1n_1 = tau_n_c * Dn
#            Ne1n_2 = pi_n_c * Dn
#
#            En_a = En * ab[n, 0]
#            En_b = En * ab[n, 1]
#
#            Es[0] += 1j * En_a * Ne1n_0
#            Es[1] += 1j * En_a * Ne1n_1 - En_b * Mo1n_1
#            Es[2] += 1j * En_a * Ne1n_2 - En_b * Mo1n_2
#
#            pi_nm1.copy_(pi_n)
#            torch.add(swisc, twisc, alpha=(1. + 1. / n), out=pi_n)
#            xi_nm2 = xi_nm1
#            xi_nm1 = xi_n
#
#        cos_phi_c = cos_phi.to(dtype_c)
#        sin_phi_c = sin_phi.to(dtype_c)
#        cos_theta_c = cos_theta.to(dtype_c)
#        sin_theta_c = sin_theta.to(dtype_c)
#
#        Es[0] *= cos_phi_c * sin_theta_c / (kr_c * kr_c)
#        Es[1] *= cos_phi_c / kr_c
#        Es[2] *= sin_phi_c / kr_c
#
#        if not cartesian:
#            return Es
#
#        Ec = self._lm_Ec
#        Ec[0] = (Es[0] * sin_theta_c * cos_phi_c
#                 + Es[1] * cos_theta_c * cos_phi_c
#                 - Es[2] * sin_phi_c)
#        Ec[1] = (Es[0] * sin_theta_c * sin_phi_c
#                 + Es[1] * cos_theta_c * sin_phi_c
#                 + Es[2] * cos_phi_c)
#        Ec[2] = Es[0] * cos_theta_c - Es[1] * sin_theta_c
#
#        return Ec

# BATCHING

    def batch_hologram(self,
                       particle_lists: list,
                       cartesian: bool = True,
                       bohren: bool = True):
        '''Generate B holograms simultaneously on the GPU.

        Arguments
        ---------
        particle_lists : a list of lists of the particles for each hologram.
            Lists can have different numbers of particles — shorter lists
            are padded with zero-ab dummy particles automatically.

        Keywords
        --------
        cartesian : bool
            If True (default), project field onto Cartesian coordinates.
        bohren : bool
            If True (default), use Bohren sign convention.

        Returns
        -------
        holograms : numpy.ndarray
            [B, npts] hologram intensities.
        '''
        field = self.batch_field(particle_lists, cartesian=cartesian, bohren=bohren)
        # field: [B, 3, npts]
        field[:, 0, :] += 1.0 + 0j
        return (field.real**2 + field.imag**2).sum(dim=1).cpu().numpy()
        # → [B, npts]

    def batch_field(self,
                    particle_lists: list,
                    cartesian: bool = True,
                    bohren: bool = True) -> torch.Tensor:
        '''Compute B scattered fields simultaneously on the GPU.

        All B*P particles are flattened into a single batched lorenzmie()
        call. Holograms with fewer than P particles are padded with dummy
        particles whose ab coefficients are zero (no scattering).

        Arguments
        ---------
        particle_lists : list of lists of Particle
            B lists of particles, one per hologram.

        Returns
        -------
        field : torch.Tensor
            [B, 3, npts] total scattered field for each hologram.
        '''
        k = float(self.instrument.wavenumber())
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength

        B = len(particle_lists)
        P = max(len(pl) for pl in particle_lists)
        npts = self._coords.shape[1]

        all_abs = [[particle.ab(n_m, wavelength) for particle in plist]
                   for plist in particle_lists]
        norders = max(ab.shape[0] for plist_abs in all_abs for ab in plist_abs)

        # Build [B, P, norders, 2] and [B, P, 3] up front, then slice per slot.
        # Dummy particles (padding) have ab=0 so they contribute nothing.
        ab_all = torch.zeros(
            B, P, norders, 2, dtype=torch.complex64, device=self._device)
        r_p_all = torch.zeros(
            B, P, 3, dtype=torch.float32, device=self._device)

        for b, plist in enumerate(particle_lists):
            for p, particle in enumerate(plist):
                ab_np = all_abs[b][p]
                n = ab_np.shape[0]
                ab_all[b, p, :n] = torch.tensor(
                    ab_np, dtype=torch.complex64, device=self._device)
                r_p_all[b, p] = torch.tensor(
                    particle.r_p + particle.r_0,
                    dtype=torch.float32, device=self._device)

        field = torch.zeros(B, 3, npts, dtype=torch.complex64, device=self._device)

        # Loop over particle slots rather than flattening all B*P at once.
        # Each iteration processes [B, npts] tensors instead of [B*P, npts],
        # keeping intermediate allocations P times smaller for better cache use.
        for p in range(P):
            ab_p  = ab_all[:, p, :, :]   # [B, norders, 2] — view, no copy
            r_p_p = r_p_all[:, p, :]     # [B, 3]          — view, no copy

            dr  = self._coords.unsqueeze(0) - r_p_p.unsqueeze(-1)  # [B, 3, npts]
            kdr = k * dr

            field += self._batch_lorenzmie(ab_p, kdr, cartesian=cartesian, bohren=bohren)
            phases = torch.exp(-1j * k * r_p_p[:, 2].to(torch.complex64))  # [B]
            field *= phases.reshape(B, 1, 1)

        return field

    def _batch_lorenzmie(self,
                         ab: torch.Tensor,
                         kdr: torch.Tensor,
                         cartesian: bool = True,
                         bohren: bool = True) -> torch.Tensor:
        '''
        Batched lorenzmie computation over N particles simultaneously via
        a 2-D Triton grid: axis-0 over pixel blocks, axis-1 over batch slots.
        The entire Mie series loop runs inside the kernel — no Python loop,
        no per-order kernel launches.

        Arguments
        ---------
        ab : torch.Tensor
            [N, norders, 2] Mie scattering coefficients for all N particles
        kdr : torch.Tensor
            [N, 3, npts] wavenumber-scaled displacements for all N particles

        Returns
        -------
        field : torch.Tensor
            [N, 3, npts] complex scattered field for each particle.
        '''
        N       = kdr.shape[0]
        norders = ab.shape[1]
        npts    = kdr.shape[2]

        if self._batch_N != N:
            self._allocate_batch(N)

        a = ab[:, :, 0].contiguous()   # [N, norders] complex64
        b = ab[:, :, 1].contiguous()   # [N, norders] complex64

        e_re = self._blm_tri_e_re      # [3, N, npts] float32
        e_im = self._blm_tri_e_im      # [3, N, npts] float32

        BLOCK = 512
        grid  = (triton.cdiv(npts, BLOCK), N)
        _batch_lorenzmie_kernel[grid](
            kdr[:, 0, :].contiguous(), kdr[:, 1, :].contiguous(), kdr[:, 2, :].contiguous(),
            a.real.contiguous(), a.imag.contiguous(),
            b.real.contiguous(), b.imag.contiguous(),
            norders,
            e_re[0], e_im[0],
            e_re[1], e_im[1],
            e_re[2], e_im[2],
            npts,
            bohren=bohren,
            cartesian=cartesian,
            BLOCK=BLOCK,
        )
        # [3, N, npts] → [N, 3, npts] to match the expected output shape
        return torch.complex(e_re, e_im).permute(1, 0, 2)

#    def _batch_lorenzmie(self,
#                         ab: torch.Tensor,
#                         kdr: torch.Tensor,
#                         cartesian: bool = True,
#                         bohren: bool = True) -> torch.Tensor:
#        '''
#        Batched lorenzmie computation over N particles simultaneously.
#
#        This is the same computation as lorenzmie() but with an extra
#        leading N dimension so the GPU processes all particles in parallel.
#
#        Arguments
#        ---------
#        ab : torch.Tensor
#            [N, norders, 2] Mie scattering coefficients for all N particles
#        kdr : torch.Tensor
#            [N, 3, npts] wavenumber-scaled displacements for all N particles
#
#        Returns
#        -------
#        field : torch.Tensor
#            [N, 3, npts] complex scattered field for each particle.
#        '''
#        dtype_c = torch.complex64
#
#        N = kdr.shape[0]
#        norders = ab.shape[1]
#
#        if self._batch_N != N:
#            self._allocate_batch(N)
#
#        # Geometry — allocated once per call, not in the Mie loop
#        kx = kdr[:, 0, :]
#        ky = kdr[:, 1, :]
#        kz = -kdr[:, 2, :]
#
#        krho      = torch.hypot(kx, ky)
#        kr        = torch.hypot(krho, kz)
#        sinkr     = torch.sin(kr)
#        coskr     = torch.cos(kr)
#        phi       = torch.atan2(ky, kx)
#        cos_phi   = torch.cos(phi)
#        sin_phi   = torch.sin(phi)
#        theta     = torch.atan2(krho, kz)
#        cos_theta = torch.cos(theta)
#        sin_theta = torch.sin(theta)
#        sign_kz   = torch.sign(kz).to(dtype_c)
#        factor    = 1j * sign_kz if bohren else -1j * sign_kz
#        sinkr_c   = sinkr.to(dtype_c)
#        coskr_c   = coskr.to(dtype_c)
#        kr_c      = kr.to(dtype_c)
#
#        # Bind pre-allocated buffers to local names
#        pi_nm1  = self._blm_pi_nm1
#        pi_n    = self._blm_pi_n
#        swisc   = self._blm_swisc
#        twisc   = self._blm_twisc
#        tau_n   = self._blm_tau_n
#        xi_nm2  = self._blm_xi_nm2
#        xi_nm1  = self._blm_xi_nm1
#        xi_n    = self._blm_xi_n
#        Dn      = self._blm_Dn
#        pi_n_c  = self._blm_pi_n_c
#        tau_n_c = self._blm_tau_n_c
#        scratch  = self._blm_scratch
#        scratch2 = self._blm_scratch2
#        Es       = self._blm_Es
#        Ec       = self._blm_Ec
#
#        # Initialise state
#        pi_nm1.zero_()
#        pi_n.fill_(1.)
#        Es.zero_()
#
#        torch.mul(factor, sinkr_c, out=scratch)
#        torch.add(coskr_c, scratch, out=xi_nm2)   # xi_{-1}
#        torch.mul(factor, coskr_c, out=scratch)
#        torch.sub(sinkr_c, scratch, out=xi_nm1)   # xi_0
#
#        for n in range(1, norders):
#
#            # pi / tau recurrence (float32, in-place)
#            torch.mul(pi_n, cos_theta, out=swisc)
#            torch.sub(swisc, pi_nm1, out=twisc)
#            torch.mul(twisc, float(n), out=tau_n)
#            torch.sub(pi_nm1, tau_n, out=tau_n)
#
#            # xi recurrence (complex64, in-place)
#            torch.div(xi_nm1, kr_c, out=xi_n)
#            xi_n.mul_(2.*n - 1.)
#            xi_n.sub_(xi_nm2)
#
#            # Dn (complex64, in-place)
#            torch.div(xi_n, kr_c, out=Dn)
#            Dn.mul_(float(n))
#            Dn.sub_(xi_nm1)
#
#            En = (1j ** n) * (2.*n + 1.) / (n*n + n)
#
#            # float → complex without allocation: write into the real view
#            pi_n_c.real.copy_(pi_n)
#            pi_n_c.imag.zero_()
#            tau_n_c.real.copy_(tau_n)
#            tau_n_c.imag.zero_()
#
#            En_a        = En * ab[:, n, 0].unsqueeze(-1)   # [N, 1]
#            En_b        = En * ab[:, n, 1].unsqueeze(-1)   # [N, 1]
#            coeff_a0    = ((n*n + n) * 1j) * En_a          # [N, 1]
#            coeff_1j_a  = 1j * En_a                        # [N, 1]
#
#            # Es[:, 0, :] += (n²+n)·1j·En_a · pi_n_c·xi_n
#            torch.mul(pi_n_c, xi_n, out=scratch)           # scratch  = pi_n_c * xi_n
#            torch.mul(scratch, coeff_a0, out=scratch2)     # scratch2 = coeff_a0 * scratch
#            Es[:, 0, :].add_(scratch2)
#
#            # Es[:, 1, :] += -En_b · pi_n_c·xi_n  (partial)
#            torch.mul(scratch, -En_b, out=scratch2)
#            Es[:, 1, :].add_(scratch2)
#
#            # Es[:, 2, :] += -En_b · tau_n_c·xi_n  (partial)
#            torch.mul(tau_n_c, xi_n, out=scratch)
#            torch.mul(scratch, -En_b, out=scratch2)
#            Es[:, 2, :].add_(scratch2)
#
#            # Es[:, 1, :] += 1j·En_a · tau_n_c·Dn  (completes Es[1])
#            torch.mul(tau_n_c, Dn, out=scratch)
#            torch.mul(scratch, coeff_1j_a, out=scratch2)
#            Es[:, 1, :].add_(scratch2)
#
#            # Es[:, 2, :] += 1j·En_a · pi_n_c·Dn  (completes Es[2])
#            torch.mul(pi_n_c, Dn, out=scratch)
#            torch.mul(scratch, coeff_1j_a, out=scratch2)
#            Es[:, 2, :].add_(scratch2)
#
#            # Advance pi recurrence: swap buffers instead of copying
#            pi_nm1, pi_n = pi_n, pi_nm1
#            torch.add(swisc, twisc, alpha=(1. + 1./n), out=pi_n)
#
#            # Advance xi recurrence: rotate three buffers, no data copy
#            xi_nm2, xi_nm1, xi_n = xi_nm1, xi_n, xi_nm2
#
#        cos_phi_c   = cos_phi.to(dtype_c)
#        sin_phi_c   = sin_phi.to(dtype_c)
#        cos_theta_c = cos_theta.to(dtype_c)
#        sin_theta_c = sin_theta.to(dtype_c)
#
#        Es[:, 0, :] *= cos_phi_c * sin_theta_c / (kr_c * kr_c)
#        Es[:, 1, :] *= cos_phi_c / kr_c
#        Es[:, 2, :] *= sin_phi_c / kr_c
#
#        if not cartesian:
#            return Es
#
#        Ec[:, 0, :] = (Es[:, 0, :] * sin_theta_c * cos_phi_c
#                       + Es[:, 1, :] * cos_theta_c * cos_phi_c
#                       - Es[:, 2, :] * sin_phi_c)
#        Ec[:, 1, :] = (Es[:, 0, :] * sin_theta_c * sin_phi_c
#                       + Es[:, 1, :] * cos_theta_c * sin_phi_c
#                       + Es[:, 2, :] * cos_phi_c)
#        Ec[:, 2, :] = Es[:, 0, :] * cos_theta_c - Es[:, 1, :] * sin_theta_c
#
#        return Ec

    @classmethod
    def batch_example(cls,
                      show: bool = True,
                      save: bool = False,
                      filename: str = None,
                      **kwargs) -> None:  # pragma: no cover
        '''Demonstrate batch_hologram() with four holograms'''
        import numpy as np
        from pylorenzmie.theory import Sphere, Instrument
        from time import perf_counter

        shape = (201, 201)
        coords = cls.meshgrid(shape)

        instrument = Instrument()
        instrument.magnification = 0.048
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340

        def sphere(x, y, z, a_p, n_p):
            p = Sphere()
            p.r_p = [x, y, z]
            p.a_p = a_p
            p.n_p = n_p
            return p

        # Defined holograms
        particle_lists = [
            [sphere(100, 100, 200, 0.5, 1.45)],
            [sphere(80,  120, 300, 1.0, 1.40)],
            [sphere(150, 150, 200, 0.5, 1.45), sphere(100, 10, 250, 1., 1.45)],
            [sphere(60,  60,  200, 0.5, 1.45), sphere(140, 100, 250, 1, 1.45)],
        ]

        model = cls(coordinates=coords, instrument=instrument, **kwargs)
        
        start = perf_counter() 
        batch_holos = model.batch_hologram(particle_lists)

        print(f'Time to calculate: {perf_counter()-start:.1e} s')
            
        start = perf_counter()
        batch_holos = model.batch_hologram(particle_lists)

        print(f'Second pass: {perf_counter()-start:.1e} s')

        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(batch_holos[i].reshape(shape), cmap='gray')
            ax.axis('off')
        fig.suptitle(f'{cls.__name__}.batch_hologram()')
        plt.tight_layout()

        if save:
            fname = filename or f'{cls.__name__}_batch_example.png'
            plt.savefig(fname)
            print(f'Saved to {fname}')
        if show:
            plt.show()

    @classmethod
    def speed_example(cls, **kwargs) -> None:  # pragma: no cover
        '''Compare time to compute 128 identical holograms across four methods:
        numpy sequential, cupy sequential, torch sequential, torch batch'''
        from pylorenzmie.theory import Sphere, Instrument
        from pylorenzmie.theory.LorenzMie import LorenzMie
        from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
        import cupy as cp
        from time import perf_counter

        N = 128
        shape = (201, 201)
        coords = cls.meshgrid(shape)

        instrument = Instrument()
        instrument.magnification = 0.048
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340

        def sphere(x, y, z, a_p, n_p):
            p = Sphere()
            p.r_p = [x, y, z]
            p.a_p = a_p
            p.n_p = n_p
            return p

        particles = [
            sphere(100, 100, 200, 0.5,  1.45),
            sphere(80,  120, 300, 1.0,  1.40),
            sphere(150,  50, 250, 0.75, 1.42),
            sphere(60,  160, 350, 1.2,  1.43),
        ]
        particle_lists = [particles for _ in range(N)]

        def gpu_sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ---- 1. numpy sequential ----
        numpy_model = LorenzMie(coordinates=coords, instrument=instrument)
        numpy_model.particle = particle_lists[0]
        numpy_model.hologram()  # warm-up
        t0 = perf_counter()
        for plist in particle_lists:
            numpy_model.particle = plist
            numpy_model.hologram()
        numpy_time = perf_counter() - t0

        # ---- 2. cupy sequential ----
        cupy_model = cupyLorenzMie(coordinates=coords, instrument=instrument)
        cupy_model.particle = particle_lists[0]
        cupy_model.hologram()  # warm-up
        cp.cuda.Device().synchronize()
        t0 = perf_counter()
        for plist in particle_lists:
            cupy_model.particle = plist
            cupy_model.hologram()
        cp.cuda.Device().synchronize()
        cupy_time = perf_counter() - t0

        # ---- 3. torch sequential ----
        torch_model = cls(coordinates=coords, instrument=instrument, **kwargs)
        torch_model.particle = particle_lists[0]
        torch_model.hologram()  # warm-up
        gpu_sync()
        t0 = perf_counter()
        for plist in particle_lists:
            torch_model.particle = plist
            torch_model.hologram()
        gpu_sync()
        torch_seq_time = perf_counter() - t0

        # ---- 4. torch batch ----
        torch_model.batch_hologram(particle_lists)  # warm-up
        gpu_sync()
        t0 = perf_counter()
        torch_model.batch_hologram(particle_lists)
        gpu_sync()
        torch_batch_time = perf_counter() - t0

        print(f'\n{N} holograms ({shape[0]}x{shape[1]}), device: {torch_model.device}')
        print(f'{"Method":<25} {"Time (s)":>10}  {"vs numpy":>10}  {"vs prev":>10}')
        print('-' * 60)
        results = [
            ('numpy sequential',  numpy_time),
            ('cupy sequential',   cupy_time),
            ('torch sequential',  torch_seq_time),
            ('torch batch',       torch_batch_time),
        ]
        prev = numpy_time
        for name, t in results:
            print(f'{name:<25} {t:>10.3e}  {numpy_time/t:>9.1f}x  {prev/t:>9.1f}x')
            prev = t


if __name__ == '__main__':  # pragma: no cover
    TorchLorenzMie.batch_example(save=True)
    TorchLorenzMie.speed_example()

