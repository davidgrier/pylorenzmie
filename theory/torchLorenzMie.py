'''Compute scattered light field on a GPU with PyTorch.

Inherits LorenzMie numpy base class.  Two GPU back ends share one class:

* CUDA GPUs run a Triton kernel (``_tritonkernel``);
* Apple GPUs run a Metal kernel through the MPS back end of PyTorch.

Both work in single precision.
'''

from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.lib.lmtypes import Image
import cmath
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# Metal kernel: the same pointwise algorithm as numbaLorenzMie, one GPU
# thread per pixel.  The point-independent factors of the partial-wave sum
# arrive precomputed in one array of eight floats per order:
# Re/Im of an0 = an*(n*n+n), an, bn, then c2 = 2n-1 and c3 = 1+1/n.
# The pixel displacement is formed in the kernel from the coordinates and
# the particle position, and the propagation phase exp(-i k z_p) is applied
# on the way out.
_METAL_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

kernel void lorenzmie(device const float* coords  [[buffer(0)]],  // (3, npts)
                      device const float* consts  [[buffer(1)]],  // (norders, 8)
                      device float2*      out     [[buffer(2)]],  // (3, npts)
                      constant uint&      npts    [[buffer(3)]],
                      constant uint&      norders [[buffer(4)]],
                      constant uint&      cart    [[buffer(5)]],
                      constant uint&      bohren  [[buffer(6)]],
                      constant float&     k       [[buffer(7)]],
                      constant float&     xp      [[buffer(8)]],
                      constant float&     yp      [[buffer(9)]],
                      constant float&     zp      [[buffer(10)]],
                      constant float&     phre    [[buffer(11)]],
                      constant float&     phim    [[buffer(12)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= npts) return;
    float kx = k * (coords[i] - xp);
    float ky = k * (coords[npts + i] - yp);
    float kz = -(k * (coords[2*npts + i] - zp));
    float krho = sqrt(kx*kx + ky*ky);
    float kr = sqrt(krho*krho + kz*kz);
    float inv_kr = 1.0f / kr;
    float sinkr = precise::sin(kr);
    float coskr = precise::cos(kr);
    float cost = kz * inv_kr;
    float sint = krho * inv_kr;
    float cosp = 1.0f, sinp = 0.0f;
    if (krho > 0.0f) { cosp = kx / krho; sinp = ky / krho; }

    // sign of Im(factor) selects h_n^(1) (kz > 0) or h_n^(2) (kz < 0)
    float s = (kz > 0.0f) ? 1.0f : ((kz < 0.0f) ? -1.0f : 0.0f);
    if (bohren == 0) s = -s;
    float2 xm2 = float2(coskr, s * sinkr);       // xi_{-1}(kr)
    float2 xm1 = float2(sinkr, -s * coskr);      // xi_0(kr)
    float pim1 = 0.0f, pin = 1.0f;
    float2 Er = float2(0.0f), Et = float2(0.0f), Ep = float2(0.0f);
    for (uint n = 1; n < norders; ++n) {
        device const float* c = consts + 8*n;
        float2 an0 = float2(c[0], c[1]);
        float2 an  = float2(c[2], c[3]);
        float2 bn  = float2(c[4], c[5]);
        float sw = pin * cost;
        float tw = sw - pim1;
        float tau = pim1 - float(n) * tw;
        float2 xn = c[6] * (xm1 * inv_kr) - xm2;
        float2 Dn = float(n) * (xn * inv_kr) - xm1;
        float2 Mt = pin * xn;
        float2 Mp = tau * xn;
        float2 Nt = tau * Dn;
        float2 Np = pin * Dn;
        Er += cmul(an0, Mt);
        Et += cmul(an, Nt) - cmul(bn, Mt);
        Ep += cmul(an, Np) - cmul(bn, Mp);
        pim1 = pin;
        pin = sw + c[7] * tw;
        xm2 = xm1;
        xm1 = xn;
    }
    Er *= cosp * sint * inv_kr * inv_kr;
    Et *= cosp * inv_kr;
    Ep *= sinp * inv_kr;
    float2 ph = float2(phre, phim);
    if (cart != 0) {
        out[i]          = cmul(Er*sint*cosp + Et*cost*cosp - Ep*sinp, ph);
        out[npts + i]   = cmul(Er*sint*sinp + Et*cost*sinp + Ep*cosp, ph);
        out[2*npts + i] = cmul(Er*cost - Et*sint, ph);
    } else {
        out[i] = cmul(Er, ph);
        out[npts + i] = cmul(Et, ph);
        out[2*npts + i] = cmul(Ep, ph);
    }
}
'''

_metal_library = None


def _metal():
    '''The compiled Metal library, built on first use.'''
    global _metal_library
    if _metal_library is None:
        _metal_library = torch.mps.compile_shader(_METAL_SOURCE)
    return _metal_library


def _partial_wave_table(ab: np.ndarray) -> np.ndarray:
    '''Point-independent factors of the partial-wave sum, shape (norders, 8).

    Folds the Mie coefficients together with the prefactors
    ``i**n (2n + 1) / (n (n + 1))`` of Bohren and Huffman (4.45), exactly as
    ``numbaLorenzMie`` does, and returns them as single-precision
    (Re, Im) pairs of ``an0``, ``an``, ``bn`` followed by ``c2`` and ``c3``.
    '''
    norders = ab.shape[0]
    n = np.arange(norders, dtype=np.float64)
    nn = np.maximum(n, 1.)
    En = (1.j ** np.arange(norders)) * (2. * n + 1.) / (nn * nn + nn)
    an = 1.j * En * ab[:, 0]
    bn = En * ab[:, 1]
    an0 = an * (n * n + n)
    table = np.stack([an0.real, an0.imag, an.real, an.imag,
                      bn.real, bn.imag, 2. * n - 1., 1. + 1. / nn], axis=1)
    return np.ascontiguousarray(table, dtype=np.float32)


def get_device(prefer: str | None = None) -> torch.device:
    '''Return a working GPU device, or raise RuntimeError.

    Parameters
    ----------
    prefer : str, optional
        ``'cuda'`` or ``'mps'`` to require that device.  By default a
        CUDA GPU is used if there is one, otherwise an Apple GPU.

    Returns
    -------
    device : torch.device
        A validated ``cuda`` or ``mps`` device.

    Raises
    ------
    RuntimeError
        If no suitable GPU is available or it cannot be initialised.
    '''
    if prefer not in (None, 'cuda', 'mps'):
        raise ValueError(f"prefer must be 'cuda' or 'mps', not {prefer!r}")
    if prefer in (None, 'cuda') and torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            logger.info('Using CUDA acceleration')
            return torch.device('cuda')
        except Exception as e:
            raise RuntimeError(
                f'CUDA device found but failed to initialise: {e}') from e
    if prefer in (None, 'mps') and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps')
            logger.info('Using Metal (MPS) acceleration')
            return torch.device('mps')
        except Exception as e:
            raise RuntimeError(
                f'Metal device found but failed to initialise: {e}') from e
    raise RuntimeError(
        'TorchLorenzMie requires a CUDA or Apple (MPS) GPU; none found. '
        'Use LorenzMie (NumPy) or numbaLorenzMie for CPU.')


class TorchLorenzMie(LorenzMie):
    '''GPU-accelerated Lorenz-Mie hologram calculator.

    A single kernel runs the full Mie partial-wave sum on the GPU: a
    Triton kernel on a CUDA GPU, a Metal kernel on an Apple GPU (through
    PyTorch's MPS back end).  All other behavior (coordinate handling,
    multi-particle accumulation, properties interface) is inherited from
    :class:`LorenzMie`.

    Parameters
    ----------
    device : torch.device or str, optional
        ``'cuda'`` or ``'mps'``.  Default: a CUDA GPU if there is one,
        otherwise an Apple GPU (see :func:`get_device`).

    Attributes
    ----------
    device : torch.device
        Compute device in use (``cuda`` or ``mps``).

    Notes
    -----
    The calculation is done in single precision, because Apple GPUs have
    no double precision.  Compared with the double-precision NumPy
    calculator the hologram differs by about 1e-5 for a sphere of 0.5 um
    radius and by up to about 1e-3 for radii of a few um, which is below
    typical camera noise but not negligible for the largest spheres.
    The returned hologram is a float64 NumPy array.
    '''

    method: str = 'torch numpy'

    def __init__(self,
                 *args,
                 device: torch.device | str | None = None,
                 **kwargs) -> None:
        if device is None:
            device = get_device()
        elif torch.device(device).type in ('cuda', 'mps'):
            device = get_device(torch.device(device).type)
        else:
            raise ValueError(f'Unsupported device {device!r}: '
                             "use 'cuda' or 'mps'")
        self._device = device
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
            self.coordinates, dtype=torch.float32,
            device=self._device).contiguous()
        if self._device.type == 'mps':
            self._metal_out = torch.zeros(
                shape, dtype=torch.complex64, device=self._device)
            self._metal_table = None      # device copy of the Mie table
            self._host_tables = []        # see _scattered_field_metal
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
        # float64 only on the host: Apple GPUs have no double precision
        result = (intensity + 2.*field[0].real + 1.).cpu().double().numpy()
        self._release_host_tables()           # .cpu() has synchronized
        return result

    def scattered_field(self,
                        particle: 'Particle',
                        cartesian: bool = True,
                        bohren: bool = True) -> torch.Tensor:
        '''Electric field scattered by a single particle.

        Parameters
        ----------
        particle : Particle
            The scattering particle.
        cartesian : bool
            If True (default), project the field onto Cartesian
            coordinates.
        bohren : bool
            If True (default), use +z along the propagation direction.

        Returns
        -------
        field : torch.Tensor, shape (3, npts), dtype complex64
            Complex scattered field including the propagation phase
            ``exp(-i k z_p)``.
        '''
        if self._device.type == 'mps':
            return self._scattered_field_metal(particle, cartesian, bohren)
        k = float(self.instrument.wavenumber())
        r_p = torch.tensor(
            particle.r_p + particle.r_0,
            dtype=torch.float32,
            device=self._device)
        ab = torch.tensor(
            particle.ab(self.instrument.n_m, self.instrument.wavelength),
            dtype=torch.complex64,
            device=self._device)
        kdr = k * (self._coords - r_p[:, None])
        phase = torch.exp(torch.tensor(
            -1j * k * float(r_p[2]),
            dtype=torch.complex64,
            device=self._device))
        return self.lorenzmie(ab, kdr, cartesian=cartesian, bohren=bohren) * phase

    def _release_host_tables(self) -> None:
        '''Free the host arrays of queued copies once the GPU is done.'''
        if self._device.type == 'mps' and self._host_tables:
            torch.mps.synchronize()
            self._host_tables.clear()

    def _scattered_field_metal(self,
                               particle: 'Particle',
                               cartesian: bool,
                               bohren: bool) -> torch.Tensor:
        '''Scattered field of one particle from the Metal kernel.

        The displacement of each pixel and the propagation phase are
        formed inside the kernel, so one small table of partial-wave
        factors is the only data sent to the GPU per particle.
        '''
        k = float(self.instrument.wavenumber())
        x, y, z = (float(c) for c in particle.r_p + particle.r_0)
        ab = particle.ab(self.instrument.n_m, self.instrument.wavelength)
        norders = ab.shape[0]
        # The table goes to a preallocated device buffer with a
        # non-blocking copy, which takes about a tenth of the time of a
        # blocking one.  The GPU reads the host array after the copy is
        # queued, so it is kept alive until the next synchronization
        # (end of hologram(), or start of field()).  Copies and kernels
        # run in order, so particles can share the device buffer.
        host = _partial_wave_table(ab)
        self._host_tables.append(host)
        table = self._metal_table
        if table is None or table.shape[0] < norders:
            table = torch.zeros(max(norders, 128), 8, dtype=torch.float32,
                                device=self._device)
            self._metal_table = table
        table[:norders].copy_(torch.from_numpy(host), non_blocking=True)
        phase = cmath.exp(-1j * k * z)
        out = self._metal_out
        npts = self._coords.shape[1]
        _metal().lorenzmie(
            self._coords, table, torch.view_as_real(out),
            npts, norders, int(cartesian), int(bohren),
            k, x, y, z, phase.real, phase.imag,
            threads=npts)
        return out

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
        self._release_host_tables()
        self._field_t.zero_()
        for particle in self.particle:
            self._field_t += self.scattered_field(
                particle, cartesian=cartesian, bohren=bohren)
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
        if self._device.type != 'cuda':
            raise NotImplementedError(
                'lorenzmie() takes precomputed displacements and is '
                'implemented for CUDA only; on Apple GPUs use '
                'scattered_field(), hologram() or field()')
        import triton
        from pylorenzmie.theory._tritonkernel import _lorenzmie_kernel

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
