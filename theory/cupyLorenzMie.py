from pylorenzmie.theory.LorenzMie import LorenzMie
import cupy as cp
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class cupyLorenzMie(LorenzMie):
    '''
    Compute scattered light field with CUDA acceleration.

    ...

    Inherits
    --------
    pylorenzmie.theory.LorenzMie

    Properties
    ----------
    double_precision : bool
        If True, perform GPU calculations in double precision.
        Default: True

    Methods
    -------
    field(cartesian=True, bohren=True, gpu=False)
        Returns the complex-valued field at each of the coordinates.

        gpu : bool
            If True, return gpu variable for field.
            Default [False]: return cpu field

    '''

    method: str = 'cupy numpy'

    def __init__(self,
                 *args,
                 double_precision: bool = True,
                 **kwargs) -> None:
        self.double_precision = double_precision
        super().__init__(*args, **kwargs)

    @property
    def double_precision(self) -> bool:
        return self._double_precision

    @double_precision.setter
    def double_precision(self, double_precision: bool) -> None:
        if double_precision:
            try:
                self.lorenzmie = self.culorenzmie()
                self.dtype = cp.float64
                self.ctype = cp.complex128
            except cp.cuda.runtime.CUDARuntimeError:
                logger.warning('GPU not capable of double precision. '
                               'Falling back to single precision.')
                double_precision = False
        if not double_precision:
            self.lorenzmie = self.culorenzmief()
            self.dtype = cp.float32
            self.ctype = cp.complex64
        self._double_precision = double_precision

    def _allocate(self) -> None:
        '''Allocate buffers for calculation'''
        shape = self.coordinates.shape
        self._field = cp.empty(shape, dtype=self.ctype)
        self.buffer = cp.empty(shape, dtype=self.ctype)
        self.coords = cp.asarray(self.coordinates, self.dtype)
        self.threadsperblock = 32
        self.blockspergrid = ((shape[1] + (self.threadsperblock - 1)) //
                              self.threadsperblock)

    def hologram(self,
                 cartesian: bool = True,
                 bohren: bool = True,
                 device: bool = False) -> cp.ndarray:
        '''Returns the hologram of the particle

        Returns
        -------
        hologram : cp.ndarray
            The hologram of the particle on the GPU

        Keywords
        --------
        cartesian : bool
            If True, compute the field in cartesian coordinates.
            Default: True
        bohren : bool
            If True, use Bohren's convention for the field.
            Default: True
        '''
        field = self.field(cartesian=cartesian, bohren=bohren, device=True)
        field[0, :] += 1.
        hologram = cp.sum(field.real**2 + field.imag**2, axis=0)
        return hologram if device else hologram.get()

    def field(self,
              cartesian: bool = True,
              bohren: bool = True,
              device: bool = False) -> cp.ndarray:
        '''Returns the field scattered by a particle'''
        k = self.dtype(self.instrument.wavenumber())
        n_m = self.instrument.n_m
        wavelength = self.instrument.wavelength
        self._field.fill(0.+0.j)
        for particle in self.particle:
            ab = particle.ab(n_m, wavelength)
            a = cp.asarray(ab[:, 0], dtype=self.ctype)
            b = cp.asarray(ab[:, 1], dtype=self.ctype)
            r_p = (particle.r_p + particle.r_0).astype(self.dtype)
            self.lorenzmie((self.blockspergrid,), (self.threadsperblock,),
                           (*self.coords, self.coords.shape[1],
                            a, b, ab.shape[0],
                            *r_p, k, cartesian, bohren,
                            *self.buffer))
            self._field += self.buffer
        return self._field if device else self._field.get()

    def culorenzmief(self) -> cp.RawKernel:
        '''Return CUDA kernel for single-precision field computation'''
        return cp.RawKernel(self._cudalorenzmie, 'lorenzmie')

    def culorenzmie(self) -> cp.RawKernel:
        '''Return CUDA kernel for double-precision field computation'''
        change = {'__sincosf(': 'sincos(',
                  'f(': '(',
                  '.f': '.',
                  'float': 'double',
                  'Float': 'Double'}
        code = self._cudalorenzmie
        for before, after in change.items():
            code = code.replace(before, after)
        return cp.RawKernel(code, 'lorenzmie')

    _cudalorenzmie = r'''
# include <cuComplex.h>

// avoid the overhead of cuC* macros.
static __device__ __inline__ cuFloatComplex cmul(const cuFloatComplex &a,
                                                 const cuFloatComplex &b) {
    return make_cuFloatComplex(a.x*b.x - a.y*b.y,
                               a.x*b.y + a.y*b.x);
}

static __device__ __inline__ cuFloatComplex cadd(const cuFloatComplex &a,
                                                 const cuFloatComplex &b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

static __device__ __inline__ cuFloatComplex csub(const cuFloatComplex &a,
                                                 const cuFloatComplex &b) {
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

static __device__ __inline__ cuFloatComplex cdiv(const cuFloatComplex &a,
                                                 const cuFloatComplex &b) {
    // a/b = (a * conj(b)) / |b|^2
    float denom = b.x*b.x + b.y*b.y;
    return make_cuFloatComplex((a.x*b.x + a.y*b.y) / denom,
                               (a.y*b.x - a.x*b.y) / denom);
}

// kernel parameters marked __restrict__/const and read with __ldg where
// appropriate to help the optimizer and use the read‑only cache.
extern "C" __global__
void lorenzmie(const float * __restrict__ x,
               const float * __restrict__ y,
               const float * __restrict__ z,
               int length,
               const cuFloatComplex * __restrict__ a,
               const cuFloatComplex * __restrict__ b,
               int norders,
               float x_p, float y_p, float z_p, float k,
               bool bohren, bool cartesian,
               cuFloatComplex * __restrict__ e1,
               cuFloatComplex * __restrict__ e2,
               cuFloatComplex * __restrict__ e3) {

    // constant values reused by every thread
    const cuFloatComplex i = make_cuFloatComplex(0.f, 1.f);
    const cuFloatComplex one = make_cuFloatComplex(1.f, 0.f);
    const cuFloatComplex two = make_cuFloatComplex(2.f, 0.f);
    const cuFloatComplex phase = make_cuFloatComplex(cosf(k*z_p), -sinf(k*z_p));

    // sign factor for 'bohren' does not depend on the particle index,
    // compute it once per kernel invocation.
    const cuFloatComplex bohrenSign = bohren ? make_cuFloatComplex(0.f, 1.f)
                                             : make_cuFloatComplex(0.f, -1.f);

    // thread index and stride for strided loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // iterate over particles with a simple strided loop
    for (; idx < length; idx += stride) {
        // read coordinate values through __ldg so they may be cached
        float kx = k * (__ldg(&x[idx]) - x_p);
        float ky = k * (__ldg(&y[idx]) - y_p);
        float kz = -k * (__ldg(&z[idx]) - z_p);

        float krho = hypotf(kx, ky);
        float kr = hypotf(krho, kz);

        float phi = atan2f(ky, kx);
        float theta = atan2f(krho, kz);
        float sinphi, cosphi, sintheta, costheta, sinkr, coskr;
        __sincosf(phi, &sinphi, &cosphi);
        __sincosf(theta, &sintheta, &costheta);
        __sincosf(kr, &sinkr, &coskr);

        cuFloatComplex sinp = make_cuFloatComplex(sinphi, 0.f);
        cuFloatComplex cosp = make_cuFloatComplex(cosphi, 0.f);
        cuFloatComplex sint = make_cuFloatComplex(sintheta, 0.f);
        cuFloatComplex cost = make_cuFloatComplex(costheta, 0.f);
        cuFloatComplex sinkrc = make_cuFloatComplex(sinkr, 0.f);
        cuFloatComplex coskrc = make_cuFloatComplex(coskr, 0.f);
        cuFloatComplex krc = make_cuFloatComplex(kr, 0.f);
        cuFloatComplex inv_kr = make_cuFloatComplex(1.f/kr, 0.f);

        // sign depending only on kz and bohren flag
        cuFloatComplex factor = make_cuFloatComplex((kz > 0) - (kz < 0), 0.f);
        factor = cmul(factor, bohrenSign);

        cuFloatComplex xi_nm2 = cadd(coskrc, cmul(factor, sinkrc));
        cuFloatComplex xi_nm1 = csub(sinkrc, cmul(factor, coskrc));
        cuFloatComplex xi_n;

        cuFloatComplex pi_nm1 = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex pi_n   = make_cuFloatComplex(1.f,0.f);

        cuFloatComplex mo1nr = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex mo1nt = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex mo1np = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex ne1nr = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex ne1nt = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex ne1np = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex esr    = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex est    = make_cuFloatComplex(0.f,0.f);
        cuFloatComplex esp    = make_cuFloatComplex(0.f,0.f);

        cuFloatComplex imagFactor = one;      // (-i)^j accumulator

        #pragma unroll 8
        for (int j = 1; j < norders; ++j) {
            cuFloatComplex n = make_cuFloatComplex((float)j, 0.f);

            cuFloatComplex swisc = cmul(pi_n, cost);
            cuFloatComplex twisc = csub(swisc, pi_nm1);
            cuFloatComplex tau_n = csub(pi_nm1, cmul(n, twisc));

            xi_n = csub(cmul(csub(cmul(two, n), one),
                              cdiv(xi_nm1, krc)), xi_nm2);

            cuFloatComplex dn = csub(cdiv(cmul(n, xi_n), krc), xi_nm1);

            mo1nt = cmul(pi_n, xi_n);
            mo1np = cmul(tau_n, xi_n);

            ne1nr = cmul(cmul(cmul(n, cadd(n, one)), pi_n), xi_n);
            ne1nt = cmul(tau_n, dn);
            ne1np = cmul(pi_n, dn);

            imagFactor = cmul(imagFactor, i);
            cuFloatComplex en = cdiv(cdiv(cmul(imagFactor,
                                  cadd(cmul(two, n), one)), n),
                                     cadd(n, one));

            cuFloatComplex aj = a[j];
            cuFloatComplex bj = b[j];

            esr = cadd(esr, cmul(cmul(cmul(i, en), aj), ne1nr));
            est = cadd(est, cmul(cmul(cmul(i, en), aj), ne1nt));
            esp = cadd(esp, cmul(cmul(cmul(i, en), aj), ne1np));
            esr = csub(esr, cmul(cmul(en, bj), mo1nr));
            est = csub(est, cmul(cmul(en, bj), mo1nt));
            esp = csub(esp, cmul(cmul(en, bj), mo1np));

            // update recursion variables
            pi_nm1 = pi_n;
            pi_n   = cadd(swisc,
                         cmul(cdiv(cadd(n, one), n), twisc));

            xi_nm2 = xi_nm1;
            xi_nm1 = xi_n;
        }

        esr = cmul(esr, cmul(cosp, inv_kr));
        esr = cmul(esr, cmul(sint, inv_kr));
        est = cmul(est, cmul(cosp, inv_kr));
        esp = cmul(esp, cmul(sinp, inv_kr));

        if (cartesian) {
            cuFloatComplex ecx = csub(cadd(cmul(esr, cmul(sint, cosp)),
                                          cmul(est, cmul(cost, cosp))),
                                      cmul(esp, sinp));
            cuFloatComplex ecy = cadd(cmul(esr, cmul(sint, sinp)),
                                      cadd(cmul(est, cmul(cost, sinp)),
                                           cmul(esp, cosp)));
            cuFloatComplex ecz = csub(cmul(esr, cost), cmul(est, sint));

            e1[idx] = cmul(ecx, phase);
            e2[idx] = cmul(ecy, phase);
            e3[idx] = cmul(ecz, phase);
        } else {
            e1[idx] = cmul(esr, phase);
            e2[idx] = cmul(est, phase);
            e3[idx] = cmul(esp, phase);
        }
    }
}
'''


if __name__ == '__main__':
    cupyLorenzMie.example(double_precision=True)
