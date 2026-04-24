# must run this script from P1 
# write in the terminal: python3 data/.py/plot_E_vs_alpha_hs.py 10
# the previous line is an example to obtain the (N,D)=(10,3) plot. You can change the particle number.
# this script only generates the E vs alpha plot in the interacting case and for D=3

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_scan_file(filepath: Path):
    """
    Expected file format:
    # alpha  energy  cpu_seconds  acceptance
    0.2      35.12   1.28         0.78
    ...

    We only use:
      column 0 -> alpha
      column 1 -> energy
    """
    data = np.loadtxt(filepath, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"File {filepath} does not contain at least 2 columns.")

    alpha = data[:, 0]
    energy = data[:, 1]
    return alpha, energy


def plot_energy_vs_alpha(txt_path: Path, pdf_path: Path, N: int, D: int = 3):
    alpha, energy = read_scan_file(txt_path)

    idx_min = np.argmin(energy)
    alpha_min = alpha[idx_min]
    energy_min = energy[idx_min]

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(alpha, energy, marker="o", linestyle="-")
    plt.scatter([alpha_min], [energy_min], zorder=3)

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle E \rangle$")
    plt.title(rf"Interacting case: $N={N}$, $D={D}$")
    plt.grid(True)
    plt.tight_layout()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved plot to: {pdf_path}")
    print(f"Minimum in scan:")
    print(f"  alpha = {alpha_min:.6f}")
    print(f"  E     = {energy_min:.8f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot E vs alpha for one interacting hard-sphere scan."
    )
    parser.add_argument(
        "N",
        type=int,
        help="Particle number N (for example: 10, 50, 100)"
    )
    parser.add_argument(
        "--D",
        type=int,
        default=3,
        help="Number of dimensions (default: 3)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]   # P1/
    txt_dir = base_dir / "data" / ".txt"
    pdf_dir = base_dir / "data" / ".pdf"

    txt_file = txt_dir / f"E_vs_alpha_hs_mode_bf_N{args.N}_D{args.D}.txt"
    pdf_file = pdf_dir / f"E_vs_alpha_hs_mode_bf_N{args.N}_D{args.D}.pdf"

    if not txt_file.exists():
        raise FileNotFoundError(f"Could not find input file: {txt_file}")

    plot_energy_vs_alpha(txt_file, pdf_file, N=args.N, D=args.D)


if __name__ == "__main__":
    main()