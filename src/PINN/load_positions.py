#load_pos
import numpy as np
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent

data_dir = project_root / "positions"
model_dir = project_root / "models"

def load_pos_memmap(n_samples, N, d, beta=None, a=0.0):
    """
    Load memmap WITHOUT loading into RAM.
    """

    beta_str = "None" if beta is None else str(beta)
    a_str = str(a)
    
    filename = f"r_all_N{N}_d{d}_beta{beta_str}_a{a_str}.dat"
    path = data_dir / filename
    print(f"Loading positions from {filename}.")
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    r_all = np.memmap(
        path,
        dtype="float64",
        mode="r",
        shape=(n_samples, N, d),
    )

    return r_all

def load_positions_txt(N, d, beta=None, a=0.0):
    """
    Load positions from text file with columns:
    step | particle index | coordinates
    """

    beta_str = "None" if beta is None else str(beta)
    a_str = str(a)

    filename = f"r_all_N{N}_d{d}_beta{beta_str}_a{a_str}.dat"
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    data = np.loadtxt(path, comments="#")

    # coordinates start at column 2
    coords = data[:, 2:2+d]

    # reshape into (samples, N, d)
    positions = coords.reshape(-1, N, d)

    return positions

def load_pos_general(a, beta, N, d):


    beta = beta.detach().cpu().item() if beta is not None and hasattr(beta, "detach") else beta

    if a==0.0:
        samples = 800000

    else:
        samples = 524288

    if a==1.0:
        if beta ==1.0:
            beta=None
            positions = load_positions_txt(
                            N=N,
                            d=d,
                            beta=beta,
                            a=a,
                            )

        else:
            positions = load_positions_txt(
                            N=N,
                            d=d,
                            beta=beta,
                            a=a,
                            )
    else:
        if beta ==1.0:
            beta=None
            positions=load_pos_memmap(
                                n_samples=samples,
                                N=N,
                                d=d,
                                beta=beta,
                                a=a,
                            )
        else:
            print(beta)
            positions=load_pos_memmap(
                                n_samples=samples,
                                N=N,
                                d=d,
                                beta=beta,
                                a=a,
                            )
    return positions