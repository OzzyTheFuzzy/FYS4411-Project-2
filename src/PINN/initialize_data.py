#initialize_data.py
import torch

from scipy.stats.qmc import LatinHypercube

class InitializeData:

    """
    Class for initialising data for PINN training
    Parameters:
    - N (int): Number of particles
    - dim (int): Dimensionality of the particles
    - device (torch.device): Device to use for computation
    - dtype (torch.dtype): Data type for tensors
    - hard_core_radius (float): Radius of the hard core interaction

    """


    def __init__(self, N, dim, device=None, dtype=torch.float32, hard_core_radius=0.0):
        self.N = N
        self.dim = dim
        self.a = hard_core_radius
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
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

        # No hard core: fast path
        if self.a == 0.0 or self.N <= 1:
            positions = width * torch.randn(
                batch_size,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype
            )
            return positions

        # Hard-core case: rejection sampling 
        accepted = []
        total_needed = batch_size
        tries = 0

        while total_needed > 0 and tries < max_tries:
            tries += 1

            candidates = width * torch.randn(
                total_needed,
                self.N,
                self.dim,
                generator=g,
                device=self.device,
                dtype=self.dtype
            )

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
        return positions    
        
    def valid_hard_core(self, candidates):      
        """
        Safeguard for the hard core condition

        Parameters:
            candidates: shape (batch_size, N, dim)

        Returns:
            torch.BoolTensor: Mask of valid configurations
        """
        if self.a == 0.0 or self.N <= 1:
            return torch.ones(candidates.shape[0], device=candidates.device, dtype=torch.bool)

        r_ij_abs, _ = self.distance_and_distance_vec(candidates)  # (B, N, N), (B, N, N, dim)

        iu = torch.triu_indices(self.N, self.N, offset=1, device=candidates.device)
        rij = r_ij_abs[:, iu[0], iu[1]]  # (B, num_pairs)

        return torch.all(rij > self.a, dim=1)

    def distance_and_distance_vec(self, r):
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

    