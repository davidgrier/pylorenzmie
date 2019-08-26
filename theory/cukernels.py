import cupy as cp

cufield = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void field(double *coordsx, double *coordsy, double *coordsz,
           double x_p, double y_p, double z_p, double k,
           cuDoubleComplex phase,
           double ar [], double ai [],
           double br [], double bi [],
           int norders, int length,
           bool bohren, bool cartesian,
           cuDoubleComplex *e1, cuDoubleComplex *e2, cuDoubleComplex *e3) {
        
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length;          idx += blockDim.x * gridDim.x) {

        double kx, ky, kz, krho, kr, phi, theta;
        double cosphi, costheta, coskr, sinphi, sintheta, sinkr;

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

        cuDoubleComplex i = make_cuDoubleComplex(0.0, 1.0);
        cuDoubleComplex factor, xi_nm2, xi_nm1;
        cuDoubleComplex mo1nr, mo1nt, mo1np, ne1nr, ne1nt, ne1np;
        cuDoubleComplex esr, est, esp;

        if (kz > 0.) {
            factor = i;
        }
        else if (kz < 0.) {
            factor = make_cuDoubleComplex(0., -1.);
        }
        else {
            factor = make_cuDoubleComplex(0., 0.);
        }

        if (bohren == false) {
            factor = cuCmul(factor, make_cuDoubleComplex(-1.,  0.));
        }

        xi_nm2 = cuCadd(make_cuDoubleComplex(coskr, 0.),
                 cuCmul(factor, make_cuDoubleComplex(sinkr, 0.)));
        xi_nm1 = cuCsub(make_cuDoubleComplex(sinkr, 0.),
                 cuCmul(factor, make_cuDoubleComplex(coskr, 0.)));

        cuDoubleComplex pi_nm1 = make_cuDoubleComplex(0.0, 0.0);
        cuDoubleComplex pi_n = make_cuDoubleComplex(1.0, 0.0);

        mo1nr = make_cuDoubleComplex(0.0, 0.0);
        mo1nt = make_cuDoubleComplex(0.0, 0.0);
        mo1np = make_cuDoubleComplex(0.0, 0.0);
        ne1nr = make_cuDoubleComplex(0.0, 0.0);
        ne1nt = make_cuDoubleComplex(0.0, 0.0);
        ne1np = make_cuDoubleComplex(0.0, 0.0);

        esr = make_cuDoubleComplex(0.0, 0.0);
        est = make_cuDoubleComplex(0.0, 0.0);
        esp = make_cuDoubleComplex(0.0, 0.0);

        cuDoubleComplex swisc, twisc, tau_n, xi_n, dn;

        cuDoubleComplex cost, sint, cosp, sinp, krc;
        cost = make_cuDoubleComplex(costheta, 0.);
        cosp = make_cuDoubleComplex(cosphi, 0.);
        sint = make_cuDoubleComplex(sintheta, 0.);
        sinp = make_cuDoubleComplex(sinphi, 0.);
        krc  = make_cuDoubleComplex(kr, 0.);

        cuDoubleComplex one, two, n, fac, en, a, b;
        one = make_cuDoubleComplex(1., 0.);
        two = make_cuDoubleComplex(2., 0.);
        int mod;

        for (int j = 1; j < norders; j++) {
            n = make_cuDoubleComplex(double(j), 0.);

            swisc = cuCmul(pi_n, cost);
            twisc = cuCsub(swisc, pi_nm1);
            tau_n = cuCsub(pi_nm1, cuCmul(n, twisc));

            xi_n = cuCsub(cuCmul(cuCsub(cuCmul(two, n), one),
                           cuCdiv(xi_nm1, krc)), xi_nm2);

            dn = cuCsub(cuCdiv(cuCmul(n, xi_n), krc), xi_nm1);

            mo1nt = cuCmul(pi_n, xi_n);
            mo1np = cuCmul(tau_n, xi_n);

            ne1nr = cuCmul(cuCmul(cuCmul(n, cuCadd(n, one)), 
                            pi_n), xi_n);
            ne1nt = cuCmul(tau_n, dn);
            ne1np = cuCmul(pi_n, dn);

            mod = j % 4;
            if (mod == 1) {fac = i;}
            else if (mod == 2) {fac = make_cuDoubleComplex(-1., 0.);}
            else if (mod == 3) {fac = make_cuDoubleComplex(0., -1.);}
            else {fac = one;}

            en = cuCdiv(cuCdiv(cuCmul(fac,
                    cuCadd(cuCmul(two, n), one)), n), cuCadd(n, one));

            a = make_cuDoubleComplex(ar[j], ai[j]);
            b = make_cuDoubleComplex(br[j], bi[j]);

            esr = cuCadd(esr, cuCmul(cuCmul(cuCmul(i, en), a), ne1nr));
            est = cuCadd(est, cuCmul(cuCmul(cuCmul(i, en), a), ne1nt));
            esp = cuCadd(esp, cuCmul(cuCmul(cuCmul(i, en), a), ne1np));
            esr = cuCsub(esr, cuCmul(cuCmul(en, b), mo1nr));
            est = cuCsub(est, cuCmul(cuCmul(en, b), mo1nt));
            esp = cuCsub(esp, cuCmul(cuCmul(en, b), mo1np));

            pi_nm1 = pi_n;
            pi_n = cuCadd(swisc,
                           cuCmul(cuCdiv(cuCadd(n, one), n), twisc));

            xi_nm2 = xi_nm1;
            xi_nm1 = xi_n;
        }

    

        cuDoubleComplex radialfactor = make_cuDoubleComplex(1. / kr,
                                                          0.);
        cuDoubleComplex radialfactorsq = make_cuDoubleComplex(1. / (kr*kr),
                                                            0.);
        esr = cuCmul(esr, cuCmul(cuCmul(cosp, sint), radialfactorsq));
        est = cuCmul(est, cuCmul(cosp, radialfactor));
        esp = cuCmul(esp, cuCmul(sinp, radialfactor));

        if (cartesian == true) {
            cuDoubleComplex ecx, ecy, ecz;
            ecx = cuCmul(esr, cuCmul(sint, cosp));
            ecx = cuCadd(ecx, cuCmul(est, cuCmul(cost, cosp)));
            ecx = cuCsub(ecx, cuCmul(esp, sinp));

            ecy = cuCmul(esr, cuCmul(sint, sinp));
            ecy = cuCadd(ecy, cuCmul(est, cuCmul(cost, sinp)));
            ecy = cuCadd(ecy, cuCmul(esp, cosp));

            ecz = cuCsub(cuCmul(esr, cost), cuCmul(est, sint));

            e1[idx] = ecx;
            e2[idx] = ecy;
            e3[idx] = ecz;
        }
        else {
            e1[idx] = esr;
            e2[idx] = est;
            e3[idx] = esp;
        }
        e1[idx] = cuCmul(e1[idx], phase);
        e2[idx] = cuCmul(e2[idx], phase);
        e3[idx] = cuCmul(e3[idx], phase);
    }
}
''', 'field')

cuhologram = cp.RawKernel(r'''
#include <cuComplex.h>

extern "C" __global__
void hologram(cuDoubleComplex *Ex,
              cuDoubleComplex *Ey,
              cuDoubleComplex *Ez,
              double alpha,
              int n,
              double *hologram) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < n; 
         idx += blockDim.x * gridDim.x) {
        cuDoubleComplex ex = Ex[idx];
        cuDoubleComplex ey = Ey[idx];
        cuDoubleComplex ez = Ez[idx];

        ex = cuCadd(ex, make_cuDoubleComplex(1., 0.));

        ex = cuCmul(ex, make_cuDoubleComplex(alpha, 0.));
        ey = cuCmul(ey, make_cuDoubleComplex(alpha, 0.));
        ez = cuCmul(ez, make_cuDoubleComplex(alpha, 0.));

        cuDoubleComplex ix = cuCmul(ex, cuConj(ex));
        cuDoubleComplex iy = cuCmul(ey, cuConj(ey));
        cuDoubleComplex iz = cuCmul(ez, cuConj(ez));

        hologram[idx] = cuCreal(cuCadd(ix, cuCadd(iy, iz)));
    }
}
''', 'hologram')


curesiduals = cp.ElementwiseKernel(
    'float64 holo, float64 data, float64 noise',
    'float64 residuals',
    'residuals = (holo - data) / noise',
    'curesiduals')

cuchisqr = cp.ReductionKernel(
    'float64 holo, float64 data, float64 noise',
    'float64 chisqr',
    '((holo - data) / noise) * ((holo - data) / noise)',
    'a + b',
    'chisqr = a',
    '0',
    'cuchisqr')
