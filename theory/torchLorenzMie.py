from pylorenzmie.theory.LorenzMie import LorenzMie
import torch
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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
        self._coords = torch.tensor(
            self.coordinates, dtype=torch.float32, device=self._device)
        self._field_t = torch.zeros(
            shape, dtype=torch.complex64, device=self._device)

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

            particle_field = self._scattered_field(
                particle, ab, kdr, cartesian=cartesian, bohren=bohren)

            phase = torch.exp(torch.tensor(
                -1j * k * float(r_p[2]),
                dtype=torch.complex64,
                device=self._device))
            particle_field *= phase
            self._field_t += particle_field

        return self._field_t

    def _scattered_field(self, particle, ab, kdr, cartesian=True, bohren=True):
        return self.lorenzmie(ab, kdr, cartesian=cartesian, bohren=bohren)

    def lorenzmie(self,
                  ab: torch.Tensor,
                  kdr: torch.Tensor,
                  cartesian: bool = True,
                  bohren: bool = True) -> torch.Tensor:
        
        dtype_r = torch.float32
        dtype_c = torch.complex64

        norders = ab.shape[0]
        npts = kdr.shape[1]

        kx = kdr[0]
        ky = kdr[1]
        kz = -kdr[2] 

        krho = torch.hypot(kx, ky)
        kr = torch.hypot(krho, kz)
        sinkr = torch.sin(kr)
        coskr = torch.cos(kr)

        phi = torch.atan2(ky, kx)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        theta = torch.atan2(krho, kz)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        sign_kz = torch.sign(kz).to(dtype_c)
        factor = (1j * sign_kz) if bohren else (-1j * sign_kz)

        sinkr_c = sinkr.to(dtype_c)
        coskr_c = coskr.to(dtype_c)
        xi_nm2 = coskr_c + factor * sinkr_c   # xi_{-1}(kr)
        xi_nm1 = sinkr_c - factor * coskr_c   # xi_0(kr)

        pi_nm1 = torch.zeros(npts, dtype=dtype_r, device=self._device)
        pi_n = torch.ones(npts, dtype=dtype_r, device=self._device)

        Es = torch.zeros(3, npts, dtype=dtype_c, device=self._device)
        kr_c = kr.to(dtype_c)

        for n in range(1, norders):

            swisc = pi_n * cos_theta
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc

            xi_n = (2. * n - 1.) * (xi_nm1 / kr_c) - xi_nm2

            Dn = n * (xi_n / kr_c) - xi_nm1

            En = (1j ** n) * (2. * n + 1.) / (n * n + n)

            pi_n_c = pi_n.to(dtype_c)
            tau_n_c = tau_n.to(dtype_c)

            Mo1n_1 = pi_n_c * xi_n
            Mo1n_2 = tau_n_c * xi_n
            Ne1n_0 = (n * n + n) * pi_n_c * xi_n
            Ne1n_1 = tau_n_c * Dn
            Ne1n_2 = pi_n_c * Dn

            En_a = En * ab[n, 0]
            En_b = En * ab[n, 1]

            Es[0] += 1j * En_a * Ne1n_0
            Es[1] += 1j * En_a * Ne1n_1 - En_b * Mo1n_1
            Es[2] += 1j * En_a * Ne1n_2 - En_b * Mo1n_2

            pi_nm1 = pi_n
            pi_n = swisc + (1. + 1. / n) * twisc
            xi_nm2 = xi_nm1
            xi_nm1 = xi_n

        cos_phi_c = cos_phi.to(dtype_c)
        sin_phi_c = sin_phi.to(dtype_c)
        cos_theta_c = cos_theta.to(dtype_c)
        sin_theta_c = sin_theta.to(dtype_c)

        Es[0] *= cos_phi_c * sin_theta_c / (kr_c * kr_c)
        Es[1] *= cos_phi_c / kr_c
        Es[2] *= sin_phi_c / kr_c

        if not cartesian:
            return Es

        Ec = torch.zeros(3, npts, dtype=dtype_c, device=self._device)
        Ec[0] = (Es[0] * sin_theta_c * cos_phi_c
                 + Es[1] * cos_theta_c * cos_phi_c
                 - Es[2] * sin_phi_c)
        Ec[1] = (Es[0] * sin_theta_c * sin_phi_c
                 + Es[1] * cos_theta_c * sin_phi_c
                 + Es[2] * cos_phi_c)
        Ec[2] = Es[0] * cos_theta_c - Es[1] * sin_theta_c

        return Ec

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
        field = self.batch_field(
            particle_lists, cartesian=cartesian, bohren=bohren)
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
        # P = max particles across all holograms (shorter lists padded to this)
        P = max(len(pl) for pl in particle_lists)
        npts = self._coords.shape[1]
        
        # Do the same for norders to pad norders 
        all_abs = [particle.ab(n_m, wavelength)
                   for plist in particle_lists
                   for particle in plist]
        norders = max(ab.shape[0] for ab in all_abs)

        # Build padded ab [B, P, norders, 2] and r_p [B, P, 3] tensors.
        # Dummy particles have ab=0 so they contribute nothing to the field.
        ab_batch = torch.zeros(
            B, P, norders, 2, dtype=torch.complex64, device=self._device)
        r_p_batch = torch.zeros(
            B, P, 3, dtype=torch.float32, device=self._device)

        for b, plist in enumerate(particle_lists):
            for p, particle in enumerate(plist):
                ab_np = particle.ab(n_m, wavelength)
                n = ab_np.shape[0]
                ab_batch[b, p, :n, :] = torch.tensor(
                    ab_np, dtype=torch.complex64, device=self._device)
                r_p_batch[b, p] = torch.tensor(
                    particle.r_p + particle.r_0,
                    dtype=torch.float32, device=self._device)

        # Flatten B*P particles into N so the GPU computes all at once
        N = B * P
        ab_flat = ab_batch.reshape(N, norders, 2)   # [N, norders, 2]
        r_p_flat = r_p_batch.reshape(N, 3)           # [N, 3]

        # Compute kdr for all N particles simultaneously:
        # self._coords: [3, npts] → unsqueeze → [1, 3, npts]
        # r_p_flat:     [N, 3]   → unsqueeze → [N, 3, 1]
        dr = self._coords.unsqueeze(0) - r_p_flat.unsqueeze(-1)
        kdr = k * dr  # [N, 3, npts]

        # Compute all N fields at once
        # Added a mask to set all dummy particle fields to 0
        fields = self._batch_lorenzmie(
            ab_flat, kdr, cartesian=cartesian, bohren=bohren)
        real = (r_p_flat[:, 2] != 0).reshape(N, 1, 1)
        fields = torch.where(real, fields, torch.zeros_like(fields))
        phases = torch.exp(
            -1j * k * r_p_flat[:, 2].to(torch.complex64)
        ).reshape(N, 1, 1)
        fields *= phases

        # Reshape back into holograms, sum particles within each hologram
        fields = fields.reshape(B, P, 3, npts)  # [B, P, 3, npts]
        return fields.sum(dim=1)                 # [B, 3, npts]

    def _batch_lorenzmie(self,
                         ab: torch.Tensor,
                         kdr: torch.Tensor,
                         cartesian: bool = True,
                         bohren: bool = True) -> torch.Tensor:
        '''
        Batched lorenzmie computation over N particles simultaneously.

        This is the same computation as lorenzmie() but with an extra
        leading N dimension so the GPU processes all particles in parallel.

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
        dtype_r = torch.float32
        dtype_c = torch.complex64

        N = kdr.shape[0]
        norders = ab.shape[1]
        npts = kdr.shape[2]

        # GEOMETRY same as lorenzmie() but every tensor is [N, npts]
        kx = kdr[:, 0, :]    # [N, npts]
        ky = kdr[:, 1, :]    # [N, npts]
        kz = -kdr[:, 2, :]   # [N, npts]

        krho = torch.hypot(kx, ky)
        kr = torch.hypot(krho, kz)
        sinkr = torch.sin(kr)
        coskr = torch.cos(kr)

        phi = torch.atan2(ky, kx)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        theta = torch.atan2(krho, kz)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # SPECIAL FUNCTIONS
        sign_kz = torch.sign(kz).to(dtype_c)
        factor = (1j * sign_kz) if bohren else (-1j * sign_kz)

        sinkr_c = sinkr.to(dtype_c)
        coskr_c = coskr.to(dtype_c)
        xi_nm2 = coskr_c + factor * sinkr_c
        xi_nm1 = sinkr_c - factor * coskr_c

        pi_nm1 = torch.zeros(N, npts, dtype=dtype_r, device=self._device)
        pi_n = torch.ones(N, npts, dtype=dtype_r, device=self._device)

        Es = torch.zeros(N, 3, npts, dtype=dtype_c, device=self._device)
        kr_c = kr.to(dtype_c)

        for n in range(1, norders):

            swisc = pi_n * cos_theta         # [N, npts]
            twisc = swisc - pi_nm1
            tau_n = pi_nm1 - n * twisc

            xi_n = (2. * n - 1.) * (xi_nm1 / kr_c) - xi_nm2
            Dn = n * (xi_n / kr_c) - xi_nm1
            En = (1j ** n) * (2. * n + 1.) / (n * n + n)  # scalar

            pi_n_c = pi_n.to(dtype_c)
            tau_n_c = tau_n.to(dtype_c)

            Mo1n_1 = pi_n_c * xi_n
            Mo1n_2 = tau_n_c * xi_n
            Ne1n_0 = (n * n + n) * pi_n_c * xi_n
            Ne1n_1 = tau_n_c * Dn
            Ne1n_2 = pi_n_c * Dn

            En_a = En * ab[:, n, 0].unsqueeze(-1)   # [N, 1]
            En_b = En * ab[:, n, 1].unsqueeze(-1)   # [N, 1]

            Es[:, 0, :] += 1j * En_a * Ne1n_0
            Es[:, 1, :] += 1j * En_a * Ne1n_1 - En_b * Mo1n_1
            Es[:, 2, :] += 1j * En_a * Ne1n_2 - En_b * Mo1n_2

            pi_nm1 = pi_n
            pi_n = swisc + (1. + 1. / n) * twisc
            xi_nm2 = xi_nm1
            xi_nm1 = xi_n

        cos_phi_c = cos_phi.to(dtype_c)
        sin_phi_c = sin_phi.to(dtype_c)
        cos_theta_c = cos_theta.to(dtype_c)
        sin_theta_c = sin_theta.to(dtype_c)

        Es[:, 0, :] *= cos_phi_c * sin_theta_c / (kr_c * kr_c)
        Es[:, 1, :] *= cos_phi_c / kr_c
        Es[:, 2, :] *= sin_phi_c / kr_c

        if not cartesian:
            return Es

        Ec = torch.zeros(N, 3, npts, dtype=dtype_c, device=self._device)
        Ec[:, 0, :] = (Es[:, 0, :] * sin_theta_c * cos_phi_c
                       + Es[:, 1, :] * cos_theta_c * cos_phi_c
                       - Es[:, 2, :] * sin_phi_c)
        Ec[:, 1, :] = (Es[:, 0, :] * sin_theta_c * sin_phi_c
                       + Es[:, 1, :] * cos_theta_c * sin_phi_c
                       + Es[:, 2, :] * cos_phi_c)
        Ec[:, 2, :] = Es[:, 0, :] * cos_theta_c - Es[:, 1, :] * sin_theta_c

        return Ec

    @classmethod
    def batch_example(cls,
                      batch_size: int = 128,
                      max_particles: int = 3,
                      show: bool = True,
                      save: bool = False,
                      filename: str = None,
                      **kwargs) -> None:  # pragma: no cover
        '''Test batch_hologram() and verify it matches hologram().
 
        Generates batch_size holograms with random particles using both
        the single and batched methods, prints timing and max difference,
        and plots the first 8 holograms from each method side by side.
 
        Keywords
        --------
        batch_size : int
            Number of holograms to generate.
        max_particles : int
            Maximum particles per hologram.
        '''
        import numpy as np
        from pylorenzmie.theory import Sphere, Instrument
        from time import perf_counter
 
        shape = (201, 201)
        c = cls.meshgrid(shape)
 
        instrument = Instrument()
        instrument.magnification = 0.048
        instrument.numerical_aperture = 1.45
        instrument.wavelength = 0.447
        instrument.n_m = 1.340
 
        # Generate random particles
        def random_particle():
            p = Sphere()
            p.r_p = [np.random.uniform(20, 180),   # x
                     np.random.uniform(20, 180),   # y
                     np.random.uniform(100, 400)]  # z
            p.a_p = np.random.uniform(0.3, 1.5)
            p.n_p = np.random.uniform(1.38, 1.60)
            return p
 
        # Each hologram gets between 1 and max_particles random particles
        particle_lists = [
            [random_particle()
             for _ in range(np.random.randint(1, max_particles + 1))]
            for _ in range(batch_size)
        ]
 
        model = cls(coordinates=c, instrument=instrument, **kwargs)
        
        # Compute with batch_hologram()
        start = perf_counter()
        batch_holos = model.batch_hologram(particle_lists)
        batch_time = perf_counter() - start
        print(f'Batch time  ({batch_size} holograms): {batch_time:.1e} s')

        # Using regular hologram method
        start = perf_counter()
        single_holos = []
        for plist in particle_lists:
            model.particle = plist
            single_holos.append(model.hologram())
        single_time = perf_counter() - start
        print(f'Single time ({batch_size} holograms): {single_time:.1e} s')
        print(f'Speedup: {single_time / batch_time:.1f}x')

        diff = np.abs(batch_holos - np.stack(single_holos))
        print(f'Max difference: {diff.max():.2e}')
        print(f'Mean difference: {diff.mean():.2e}')
        
        # Plot first 8 holograms from each method
        n_plot = min(batch_size, 8)
        fig, axes = plt.subplots(2, n_plot, figsize=(2 * n_plot, 4))
        fig.suptitle(f'{cls.__name__} batch_example (B={batch_size})')
        for i in range(n_plot):
            axes[0, i].imshow(batch_holos[i].reshape(shape), cmap='gray')
            axes[0, i].set_title(f'{i+1}', fontsize=8)
            axes[0, i].axis('off')
            axes[1, i].imshow(single_holos[i].reshape(shape), cmap='gray')
            axes[1, i].set_title(f'{i+1}', fontsize=8)
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel('batch')
        axes[1, 0].set_ylabel('single')
        plt.tight_layout()
 
        if save:
            fname = filename or f'{cls.__name__}_batch_example.png'
            plt.savefig(fname)
            print(f'Saved to {fname}')
        if show:
            plt.show()
 

if __name__ == '__main__':  # pragma: no cover
    TorchLorenzMie.batch_example()

