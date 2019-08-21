import cupy as cp

cufield = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void field(float *coordsx, float *coordsy, float *coordsz,
           float x_p, float y_p, float z_p, float k,
           cuFloatComplex phase,
           float ar [], float ai [],
           float br [], float bi [],
           int norders, int length,
           bool bohren, bool cartesian,
           cuFloatComplex *e1, cuFloatComplex *e2, cuFloatComplex *e3) {
        
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length;          idx += blockDim.x * gridDim.x) {

        float kx, ky, kz, krho, kr, phi, theta;
        float cosphi, costheta, coskr, sinphi, sintheta, sinkr;

        kx = k * (coordsx[idx] - x_p);
        ky = k * (coordsy[idx] - y_p);
        kz = k * (coordsz[idx] - z_p);

        kz *= -1;

        krho = sqrt(kx*kx + ky*ky);
        kr = sqrt(krho*krho + kz*kz);

        phi = atan2(ky, kx);
        theta = atan2(krho, kz);
        sincos(phi, &sinphi, &cosphi);
        sincos(theta, &sintheta, &costheta);
        sincos(kr, &sinkr, &coskr);

        cuFloatComplex i = make_cuFloatComplex(0.0, 1.0);
        cuFloatComplex factor, xi_nm2, xi_nm1;
        cuFloatComplex mo1nr, mo1nt, mo1np, ne1nr, ne1nt, ne1np;
        cuFloatComplex esr, est, esp;

        if (kz > 0.) {
            factor = i;
        }
        else if (kz < 0.) {
            factor = make_cuFloatComplex(0., -1.);
        }
        else {
            factor = make_cuFloatComplex(0., 0.);
        }

        if (bohren == false) {
            factor = cuCmulf(factor, make_cuFloatComplex(-1.,  0.));
        }

        xi_nm2 = cuCaddf(make_cuFloatComplex(coskr, 0.),
                 cuCmulf(factor, make_cuFloatComplex(sinkr, 0.)));
        xi_nm1 = cuCsubf(make_cuFloatComplex(sinkr, 0.),
                 cuCmulf(factor, make_cuFloatComplex(coskr, 0.)));

        cuFloatComplex pi_nm1 = make_cuFloatComplex(0.0, 0.0);
        cuFloatComplex pi_n = make_cuFloatComplex(1.0, 0.0);

        mo1nr = make_cuFloatComplex(0.0, 0.0);
        mo1nt = make_cuFloatComplex(0.0, 0.0);
        mo1np = make_cuFloatComplex(0.0, 0.0);
        ne1nr = make_cuFloatComplex(0.0, 0.0);
        ne1nt = make_cuFloatComplex(0.0, 0.0);
        ne1np = make_cuFloatComplex(0.0, 0.0);

        esr = make_cuFloatComplex(0.0, 0.0);
        est = make_cuFloatComplex(0.0, 0.0);
        esp = make_cuFloatComplex(0.0, 0.0);

        cuFloatComplex swisc, twisc, tau_n, xi_n, dn;

        cuFloatComplex cost, sint, cosp, sinp, krc;
        cost = make_cuFloatComplex(costheta, 0.);
        cosp = make_cuFloatComplex(cosphi, 0.);
        sint = make_cuFloatComplex(sintheta, 0.);
        sinp = make_cuFloatComplex(sinphi, 0.);
        krc  = make_cuFloatComplex(kr, 0.);

        cuFloatComplex one, two, n, fac, en, a, b;
        one = make_cuFloatComplex(1., 0.);
        two = make_cuFloatComplex(2., 0.);
        int mod;

        for (int j = 1; j < norders; j++) {
            n = make_cuFloatComplex(float(j), 0.);

            swisc = cuCmulf(pi_n, cost);
            twisc = cuCsubf(swisc, pi_nm1);
            tau_n = cuCsubf(pi_nm1, cuCmulf(n, twisc));

            xi_n = cuCsubf(cuCmulf(cuCsubf(cuCmulf(two, n), one),
                           cuCdivf(xi_nm1, krc)), xi_nm2);

            dn = cuCsubf(cuCdivf(cuCmulf(n, xi_n), krc), xi_nm1);

            mo1nt = cuCmulf(pi_n, xi_n);
            mo1np = cuCmulf(tau_n, xi_n);

            ne1nr = cuCmulf(cuCmulf(cuCmulf(n, cuCaddf(n, one)), 
                            pi_n), xi_n);
            ne1nt = cuCmulf(tau_n, dn);
            ne1np = cuCmulf(pi_n, dn);

            mod = j % 4;
            if (mod == 1) {fac = i;}
            else if (mod == 2) {fac = make_cuFloatComplex(-1., 0.);}
            else if (mod == 3) {fac = make_cuFloatComplex(0., -1.);}
            else {fac = one;}

            en = cuCdivf(cuCdivf(cuCmulf(fac,
                    cuCaddf(cuCmulf(two, n), one)), n), cuCaddf(n, one));

            a = make_cuFloatComplex(ar[j], ai[j]);
            b = make_cuFloatComplex(br[j], bi[j]);

            esr = cuCaddf(esr, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1nr));
            est = cuCaddf(est, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1nt));
            esp = cuCaddf(esp, cuCmulf(cuCmulf(cuCmulf(i, en), a), ne1np));
            esr = cuCsubf(esr, cuCmulf(cuCmulf(en, b), mo1nr));
            est = cuCsubf(est, cuCmulf(cuCmulf(en, b), mo1nt));
            esp = cuCsubf(esp, cuCmulf(cuCmulf(en, b), mo1np));

            pi_nm1 = pi_n;
            pi_n = cuCaddf(swisc,
                           cuCmulf(cuCdivf(cuCaddf(n, one), n), twisc));

            xi_nm2 = xi_nm1;
            xi_nm1 = xi_n;
        }

    

        cuFloatComplex radialfactor = make_cuFloatComplex(1. / kr,
                                                          0.);
        cuFloatComplex radialfactorsq = make_cuFloatComplex(1. / (kr*kr),
                                                            0.);
        esr = cuCmulf(esr, cuCmulf(cuCmulf(cosp, sint), radialfactorsq));
        est = cuCmulf(est, cuCmulf(cosp, radialfactor));
        esp = cuCmulf(esp, cuCmulf(sinp, radialfactor));

        if (cartesian == true) {
            cuFloatComplex ecx, ecy, ecz;
            ecx = cuCmulf(esr, cuCmulf(sint, cosp));
            ecx = cuCaddf(ecx, cuCmulf(est, cuCmulf(cost, cosp)));
            ecx = cuCsubf(ecx, cuCmulf(esp, sinp));

            ecy = cuCmulf(esr, cuCmulf(sint, sinp));
            ecy = cuCaddf(ecy, cuCmulf(est, cuCmulf(cost, sinp)));
            ecy = cuCaddf(ecy, cuCmulf(esp, cosp));

            ecz = cuCsubf(cuCmulf(esr, cost), cuCmulf(est, sint));

            e1[idx] = ecx;
            e2[idx] = ecy;
            e3[idx] = ecz;
        }
        else {
            e1[idx] = esr;
            e2[idx] = est;
            e3[idx] = esp;
        }
        e1[idx] = cuCmulf(e1[idx], phase);
        e2[idx] = cuCmulf(e2[idx], phase);
        e3[idx] = cuCmulf(e3[idx], phase);
    }
}
''', 'field')

cuhologram = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void hologram(cuFloatComplex *Ex,
              cuFloatComplex *Ey,
              cuFloatComplex *Ez,
              float alpha,
              int n,
              float *hologram) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n; 
         idx += blockDim.x * gridDim.x) {
        cuFloatComplex ex = Ex[idx];
        cuFloatComplex ey = Ey[idx];
        cuFloatComplex ez = Ez[idx];

        ex = cuCaddf(ex, make_cuFloatComplex(1., 0.));

        ex = cuCmulf(ex, make_cuFloatComplex(alpha, 0.));
        ey = cuCmulf(ey, make_cuFloatComplex(alpha, 0.));
        ez = cuCmulf(ez, make_cuFloatComplex(alpha, 0.));

        cuFloatComplex ix = cuCmulf(ex, cuConjf(ex));
        cuFloatComplex iy = cuCmulf(ey, cuConjf(ey));
        cuFloatComplex iz = cuCmulf(ez, cuConjf(ez));

        hologram[idx] = cuCrealf(cuCaddf(ix, cuCaddf(iy, iz)));
    }
}
''', 'hologram')


curesiduals = cp.ElementwiseKernel(
    'float32 holo, float32 data, float32 noise',
    'float32 residuals',
    'residuals = (holo - data) / noise',
    'cu_residuals')

cuchisqr = cp.ReductionKernel(
    'float32 holo, float32 data, float32 noise',
    'float32 chisqr',
    '((holo - data) / noise) * ((holo - data) / noise)',
    'a + b',
    'chisqr = a',
    '0',
    'cu_chisqr')
