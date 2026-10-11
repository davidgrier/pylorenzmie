'''Triton kernel for the CUDA, ROCm and Intel GPU implementations of
TorchLorenzMie.

Kept in a module of its own so that ``torchLorenzMie`` imports without
Triton, which is not available on every platform (macOS has none).  Only
portable Triton functions are used, so the kernel does not depend on the
CUDA-specific libdevice.
'''

import triton
import triton.language as tl


# Triton kernel: computes the Lorenz-Mie scattered field for a
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
    # direction cosines from ratios, as in the Numba and Metal kernels; on
    # the axis (krho = 0) the azimuth is undefined and any value will do.
    # This avoids atan2, which exists only in the CUDA-specific libdevice.
    cos_theta = kz / kr
    sin_theta = krho / kr
    off_axis  = krho > 0.0
    cos_phi   = tl.where(off_axis, kx / krho, 1.0)
    sin_phi   = tl.where(off_axis, ky / krho, 0.0)
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
