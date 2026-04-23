from pylorenzmie.theory.LorenzMie import LorenzMie
from pylorenzmie.theory.torchLorenzMie import TorchLorenzMie
from Aberrated import Aberrated
import torch

# Make aberrated version of TorchLorenzMie
AberratedTorchBase = Aberrated(TorchLorenzMie)

class AberratedTorchLorenzMie(AberratedTorchBase):

# Overriding for torch 
    def _scattered_field(self, particle, ab, kdr, cartesian=True, bohren=True):
        field = TorchLorenzMie._scattered_field(self, particle, ab, kdr, cartesian=cartesian, bohren=bohren)
        r_p = particle.r_p + particle.r_0
        aberration = self._aberration(r_p)
        return field * torch.tensor(aberration, dtype=torch.complex64, device=self._device)

# Adding batching functionalities
    def _aberration_batched(self, r_p_flat: torch.Tensor) -> torch.Tensor:
        NA  = self.instrument.numerical_aperture
        n_m = self.instrument.n_m

        px = self._coords[0]
        py = self._coords[1]

        x_p = r_p_flat[:, 0].unsqueeze(1)
        y_p = r_p_flat[:, 1].unsqueeze(1)
        z_p = r_p_flat[:, 2]

        omega = (2.0 * NA / n_m) * z_p
        omega = omega.unsqueeze(1)
        omega = torch.where(omega == 0, torch.ones_like(omega), omega)
        
        x = (px.unsqueeze(0) - x_p) / omega
        y = (py.unsqueeze(0) - y_p) / omega

        rhosq = x * x + y * y
        phase = (6.0 * rhosq * (rhosq - 1.0) + 1.0) * (-self.spherical)

        return torch.exp(1j * phase.to(torch.complex64))

    def batch_field(self,
                    particle_lists: list,
                    cartesian: bool = True,
                    bohren: bool = True) -> torch.Tensor:
        k          = float(self.instrument.wavenumber())
        n_m        = self.instrument.n_m
        wavelength = self.instrument.wavelength

        B    = len(particle_lists)
        P    = max(len(pl) for pl in particle_lists)
        npts = self._coords.shape[1]

        all_abs = [particle.ab(n_m, wavelength)
                   for plist in particle_lists
                   for particle in plist]
        norders = max(ab.shape[0] for ab in all_abs)

        ab_batch  = torch.zeros(B, P, norders, 2,
                                dtype=torch.complex64, device=self._device)
        r_p_batch = torch.zeros(B, P, 3,
                                dtype=torch.float32, device=self._device)

        for b, plist in enumerate(particle_lists):
            for p, particle in enumerate(plist):
                ab_np = particle.ab(n_m, wavelength)
                n = ab_np.shape[0]
                ab_batch[b, p, :n, :] = torch.tensor(
                    ab_np, dtype=torch.complex64, device=self._device)
                r_p_batch[b, p] = torch.tensor(
                    particle.r_p + particle.r_0,
                    dtype=torch.float32, device=self._device)

        N        = B * P
        ab_flat  = ab_batch.reshape(N, norders, 2)
        r_p_flat = r_p_batch.reshape(N, 3)

        dr  = self._coords.unsqueeze(0) - r_p_flat.unsqueeze(-1)
        kdr = k * dr

        fields = self._batch_lorenzmie(ab_flat, kdr, cartesian=cartesian, bohren=bohren)
        real = (r_p_flat[:, 2] != 0).reshape(N, 1, 1)
        fields = torch.where(real, fields, torch.zeros_like(fields))
        print('after lorenzmie:', fields.isnan().any())

        if self.spherical != 0.0:
            aberration = self._aberration_batched(r_p_flat)
            print('aberration:', aberration.isnan().any())
            real = (r_p_flat[:, 2] != 0).unsqueeze(1)
            aberration = torch.where(real, aberration, torch.ones_like(aberration))
            print('aberration after mask:', aberration.isnan().any())
            fields *= aberration.unsqueeze(1)
            print('after applying aberration:', fields.isnan().any())

        phases = torch.exp(-1j * k * r_p_flat[:, 2].to(torch.complex64)).reshape(N, 1, 1)
        fields *= phases
        print('after phases:', fields.isnan().any())
        fields = fields.reshape(B, P, 3, npts)
        return fields.sum(dim=1)


if __name__ == '__main__':
    AberratedTorchLorenzMie.batch_example(spherical=0, save = True)
