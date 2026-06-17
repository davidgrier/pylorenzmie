'''GPU-accelerated batched Lorenz-Mie scattering using PyTorch and Triton.

This module extends :class:`~pylorenzmie.theory.torchLorenzMie.TorchLorenzMie`
with a batched computation path that generates B holograms in a single Triton
kernel launch, which is substantially faster than calling
:meth:`~TorchLorenzMie.hologram` B times in a loop.

Key components
--------------
_batch_lorenzmie_kernel
    Triton kernel — 2-D grid variant that handles N particles simultaneously.
TorchLorenzMieBatch
    Subclass of :class:`TorchLorenzMie` that adds :meth:`batch_hologram` and
    :meth:`batch_field` for generating many holograms in a single GPU call.
'''

from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tl_libdevice
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
    '''Triton kernel: batched Lorenz-Mie scattered field for N particles.

    Extends :func:`_lorenzmie_kernel` to a 2-D launch grid:
    axis-0 iterates over pixel blocks (``pid``), axis-1 over batch slots
    (``bid``).  Each program instance handles one particle's pixels
    independently, so all N particles run in parallel.

    Parameters
    ----------
    kx_ptr, ky_ptr, kz_ptr : pointer to float32[N*npts]
        Wavenumber-scaled displacements, stored batch-major (row ``b`` starts
        at offset ``b * npts``).
    a_re_ptr, a_im_ptr : pointer to float32[N*norders]
        Real/imaginary Mie ``a_n`` coefficients, stored batch-major.
    b_re_ptr, b_im_ptr : pointer to float32[N*norders]
        Real/imaginary Mie ``b_n`` coefficients, stored batch-major.
    norders : tl.constexpr int
        Number of Mie orders (compile-time constant shared across the batch).
    e1_re_ptr … e3_im_ptr : pointer to float32[N*npts]
        Output field buffers, stored batch-major.
    npts : int
        Number of pixels per particle.
    bohren : tl.constexpr bool
        Sign convention (see :func:`_lorenzmie_kernel`).
    cartesian : tl.constexpr bool
        Coordinate system for the output field.
    BLOCK : tl.constexpr int
        Pixel tile size per program instance.
    '''
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


class TorchLorenzMieBatch(TorchLorenzMie):
    '''Batched GPU-accelerated Lorenz-Mie hologram calculator.

    Extends :class:`TorchLorenzMie` with a batched computation path that
    generates B holograms in a single Triton kernel launch.  All other
    behaviour (single-hologram API, device selection, coordinate handling)
    is inherited unchanged.

    Additional methods
    ------------------
    batch_hologram(particle_lists)
        Generate B holograms simultaneously and return ``[B, npts]`` intensities.
    batch_field(particle_lists)
        Compute B scattered fields and return a ``[B, 3, npts]`` complex tensor.

    Examples
    --------
    ::

        from pylorenzmie.theory import Sphere, Instrument
        from pylorenzmie.theory.torchLorenzMieBatch import TorchLorenzMieBatch

        instrument = Instrument()
        coords = TorchLorenzMieBatch.meshgrid((201, 201))
        model = TorchLorenzMieBatch(coordinates=coords, instrument=instrument)
        holograms = model.batch_hologram(particle_lists)  # shape (B, 201*201)
    '''

    def _allocate(self) -> None:
        super()._allocate()
        self._batch_N = 0

    def _allocate_batch(self, N: int) -> None:
        '''Allocate [N, npts] output buffers for _batch_lorenzmie'''
        npts = self._coords.shape[1]
        self._batch_N      = N
        d_r, dev           = torch.float32, self._device
        self._blm_tri_e_re = torch.zeros(3, N, npts, dtype=d_r, device=dev)
        self._blm_tri_e_im = torch.zeros(3, N, npts, dtype=d_r, device=dev)

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

#---------------------------------------------------------------------------
#TESTS

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

#------------------------------------------------------------------------------
    @classmethod
    def speed_example(cls, **kwargs) -> None:  # pragma: no cover

        '''
        Compare time to compute 128 identical holograms across four methods:
        numpy sequential, cupy sequential, torch sequential, torch batch
        '''
        from pylorenzmie.theory import Sphere, Instrument
        from pylorenzmie.theory.LorenzMie import LorenzMie
        from pylorenzmie.theory.cupyLorenzMie import cupyLorenzMie
        import cupy as cp
        from time import perf_counter

        '''
        Setup: Creates a 201x201 coordinate grid. Defines a list of 4
        particles as particles. Creates a list of 128 of 'particles' i.e
        128 identical holograms, each with 4 particles (multiple particles
        to properly test batching time speedup)
        '''
        N = 128
        shape = (1024, 1024)
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

        '''
        Sync timer to gpu time for accurate counter results.
        '''
        def gpu_sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        '''
        NumPy Sequential: Create a plain lorenzmie model, then
        loop over the 128 holograms one at a time. Baseline model.
        '''
        numpy_model = LorenzMie(coordinates=coords, instrument=instrument)
        numpy_model.particle = particle_lists[0]
        numpy_model.hologram()  # warm-up
        t0 = perf_counter()
        for plist in particle_lists:
            numpy_model.particle = plist
            numpy_model.hologram()
        numpy_time = perf_counter() - t0

        '''
        CuPy Sequential: same as above but using cupyLorenzMie.
        '''
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

        '''
        Torch sequential: same as above but using torchLorenzmie.
        '''
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

        '''
        Torch batch: now uses batch capabilities. Processes 128 holograms
        in batches of size 'batch_size' rather than serially.
        '''
        batch_size = 1
        batch_list = [particle_lists[i:i+batch_size] for i in range(0, len(particle_lists), batch_size)]

        for batch in batch_list:  # warm-up
            torch_model.batch_hologram(batch)

        gpu_sync()
        t0 = perf_counter()
        results = []
        for batch in batch_list:
            results.append(torch_model.batch_hologram(batch))
        gpu_sync()
        torch_batch_time = perf_counter() - t0

        '''
        Print table with times
        '''
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
    TorchLorenzMieBatch.speed_example()
