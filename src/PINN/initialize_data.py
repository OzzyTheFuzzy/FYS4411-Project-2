#initialize_data.py
import torch
import numpy as np

def distance_and_distance_vec(r,eps=0.0):
    """
    Calculate pairwise displacement vectors and distances.

    Parameters:
        r: shape (batch_size, N, dim)

    Returns:
        r_ij_abs: shape (batch_size, N, N)
        r_ij:     shape (batch_size, N, N, dim)
    """
    # Pairwise displacement vectors
    r_ij = r[:, :, None, :] - r[:, None, :, :]   # (B, N, N, dim)

    # Pairwise distances
    r_ij_abs = torch.linalg.norm(r_ij, dim=-1)   # (B, N, N)
    
    return r_ij_abs, r_ij



class InitializeData:

    """
    Class for initialising data for PINN training
    Parameters:
    - N (int): Number of particles
    - dim (int): Dimensionality of the particles
    - device (torch.device): Device to use for computation
    - dtype (torch.dtype): Data type for tensors
    - hard_core_radius (float): Radius of the hard core interaction
    - omega_z (float): Frequency of the harmonic trap in the z-direction

    """

    def __init__(self, N, dim, device=None, dtype=torch.float32, hard_core_radius=0.0, omega_z=1.0):
        self.N = N
        self.dim = dim
        self.a = hard_core_radius
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.omega_z = omega_z if omega_z is not None else 1.0
        self.sigma_x = 1 / np.sqrt(2) 
        self.sigma_y = 1/ np.sqrt(2)
        self.sigma_z = 1 / np.sqrt(2 * omega_z)

    def initialize_pde(self, batch_size, width, seed):

        """ Generates collocation points for training the PDE loss.
       
        Parameters:
            - batch_size (int): Number of collocation points in each batch.
            - width    (float): Width of the Gaussian distribution where you sample positions from
            - seed     (int): Random seed.

        Returns:
            - torch.Tensor(batch_size, N, dim): Collocation points.
        """

        particle_positions = self.particle_positions(batch_size, width, seed)

        return particle_positions

        # Start positions near origin (Gaussian). Shape: (N, dim) 

    def particle_positions(self, batch_size, width=1.0, seed=17, max_tries=1000):
        """
        Sample many configurations from a Gaussian.

        Output shape: (batch_size, N, dim)
        """
        # seed for reproducibility
        g = torch.Generator(device=self.device).manual_seed(seed)

        sigmas = torch.ones(self.dim, device=self.device, dtype=self.dtype)

        if self.dim >= 1:
            sigmas[0] = self.sigma_x
        if self.dim >= 2:
            sigmas[1] = self.sigma_y
        if self.dim >= 3:
            sigmas[2] = self.sigma_z

        sigmas = width * sigmas.view(1, 1, self.dim)
        total_needed = batch_size
        n1 = total_needed//2
        n2 = total_needed//2

        # No hard core 
        if self.a == 0.0:

            candidates_1 = torch.randn(
                n1,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            candidates_2 = 1.5 * torch.randn(
                n2,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            # concatenate and shuffle candidates
            candidates = torch.cat([candidates_1, candidates_2], dim=0)
            # scramble the particle positions to avoid any ordering bias
            perm = torch.randperm(
                candidates.shape[0],
                generator=g,
                device=self.device,
            )

            return candidates[perm] 

        # Hard-core case: rejection sampling 
        accepted = []
        tries = 0
        n1 = 49 * total_needed // 100 
        n2 = 49 * total_needed // 100
        n3 = total_needed - n1 - n2  # takes the remainder to ensure n1+n2+n3 = batch_size
        if self.a>0.0:
            particle_pos_fn = self.particle_pos2(batch_size, seed, max_tries
                                                 )
            return particle_pos_fn

        while total_needed > 0 and tries < max_tries:
            tries += 1

            candidates_1 = torch.randn(
                n1,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            candidates_2 = 1.5 * torch.randn(
                n2,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            candidates_3 = 0.8 * torch.randn(
                n3,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            valid_1 = candidates_1[self.min_distance(candidates_1, min_distance=0.45)]
            valid_2 = candidates_2[self.min_distance(candidates_2, min_distance=0.45)]
            valid_3 = candidates_3[self.min_distance(candidates_3, min_distance=0.45)]

            candidates = torch.cat([valid_1, valid_2, valid_3], dim=0)

            if candidates.shape[0] > 0:
                perm = torch.randperm(
                    candidates.shape[0],
                    generator=g,
                    device=self.device,
                )

                candidates = candidates[perm]

                n_take = min(total_needed, candidates.shape[0])
                accepted.append(candidates[:n_take])
                total_needed -= n_take

        if total_needed > 0:
            raise RuntimeError("Failed to generate enough valid configurations.")

        positions = torch.cat(accepted, dim=0)[:batch_size]
        return positions

                
    def min_distance(self, candidates, min_distance=0.80):
        r_ij_abs, _ = self.distance_and_distance_vecs(candidates)

        iu = torch.triu_indices(self.N, self.N, offset=1, device=candidates.device)
        rij = r_ij_abs[:, iu[0], iu[1]]

        return torch.all(rij > min_distance, dim=1)


    def distance_and_distance_vecs(self, r):
        """
        Calculate pairwise displacement vectors and distances.

        Parameters:
            r: shape (batch_size, N, dim)

        Returns:
            r_ij_abs: shape (batch_size, N, N)
            r_ij:     shape (batch_size, N, N, dim)
        """
        # Pairwise displacement vectors
        r_ij_abs, r_ij = distance_and_distance_vec(r)   # (B, N, N), (B, N, N, dim)

        return r_ij_abs, r_ij

    def particle_pos2(self, batch_size, seed=17, max_tries=1000):
        """
        Interacting sampler for anisotropic trap.

        Samples an ellipsoidal radial variable s, where

            s^2 = x^2 + y^2 + omega_z z^2

        Then maps to

            x = s n_x
            y = s n_y
            z = s n_z / sqrt(omega_z)

        This gives configurations adapted to the trap shape.
        """

        g = torch.Generator(device=self.device).manual_seed(seed)

        accepted = []
        total_needed = batch_size
        tries = 0

        # Tune these
        s_mean = 2.0
        s_std = 0.5
        min_dist = 0.2

        while total_needed > 0 and tries < max_tries:
            tries += 1

            n_propose = 4 * total_needed

            # sample ellipsoidal radius
            s = torch.normal(
                mean=s_mean,
                std=s_std,
                size=(n_propose, self.N),
                generator=g,
                device=self.device,
                dtype=self.dtype,
            )

            s = torch.clamp(s, min=0.05)

            # random directions on unit sphere
            directions = torch.randn(
                n_propose,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            )

            directions = directions / (
                torch.linalg.norm(directions, dim=-1, keepdim=True) + 1e-12
            )

            # ellipsoidal coordinates
            positions = s.unsqueeze(-1) * directions

            # compress z according to anisotropic trap
            if self.dim >= 3:
                positions[:, :, 2] /= self.omega_z

            valid_mask = self.min_distance(
                positions,
                min_distance=min_dist,
            )

            valid = positions[valid_mask]

            if valid.shape[0] > 0:
                n_take = min(total_needed, valid.shape[0])
                accepted.append(valid[:n_take])
                total_needed -= n_take

        if total_needed > 0:
            raise RuntimeError(
                "particle_pos2 failed to generate enough valid configurations."
            )

        return torch.cat(accepted, dim=0)[:batch_size]
        