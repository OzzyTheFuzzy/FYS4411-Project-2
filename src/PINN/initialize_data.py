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

    def particle_positions(self, batch_size, width=1.0, seed=17, max_tries=100):
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
        n1 = total_needed // 3
        n2 = total_needed // 3
        n3 = total_needed - n1 - n2  # takes the remainder to ensure n1+n2+n3 = batch_size
        
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

            candidates_3 = 0.5 * torch.randn(
                n3,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype,
            ) * sigmas

            candidates = torch.cat([candidates_1, candidates_2, candidates_3], dim=0)
            # scramble the particle positions to avoid any ordering bias
            perm = torch.randperm(
                candidates.shape[0],
                generator=g,
                device=self.device,
            )

            candidates = candidates[perm]
            valid_mask = self.valid_hard_core(candidates)

            if valid_mask.any():
                accepted.append(candidates[valid_mask])
                total_needed -= valid_mask.sum().item()

        if total_needed > 0:
            raise RuntimeError(
                f"Could not generate enough valid configurations with hard-core radius {self.a}. "
                f"Still missing {total_needed} after {max_tries} tries. Fix width of initialisation."
            )

        positions = torch.cat(accepted, dim=0)[:batch_size]
        # return positions with only valid configurations, shape (batch_size, N, dim)

        return positions
            
    def valid_hard_core(self, candidates):      
        """
        Safeguard for the hard core condition

        Parameters:
            candidates: shape (batch_size, N, dim)

        Returns:
            torch.BoolTensor: Mask of valid configurations
        """
        r_ij_abs, r_ij = self.distance_and_distance_vecs(candidates)  # (B, N, N), (B, N, N, dim)

        iu = torch.triu_indices(self.N, self.N, offset=1, device=candidates.device)
        rij = r_ij_abs[:, iu[0], iu[1]]  # (B, num_pairs)
        if self.dim == 1 or self.dim==2:
             # in 1D we only care about the absolute distance between particles
            eps=5e-2

            return torch.all(rij > self.a + eps, dim=1)
        
        else:
            return torch.all(rij > self.a, dim=1)

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

    