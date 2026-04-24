# write in the terminal for one case: python3 data/.py/plot_onebody_densities.py 10
# write in the terminal for the three cases (10,50,100): python3 data/.py/plot_onebody_densities.py all

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_density_file(filepath: Path):
    """
    Reads a two-column density file:
    # z density
    z0 rho0
    z1 rho1
    ...
    or
    # rho density
    r0 rho0
    r1 rho1
    ...
    """
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"File {filepath} does not contain at least 2 columns.")

    x = data[:, 0]
    y = data[:, 1]
    return x, y


def make_single_plot(x_with, y_with, x_noj, y_noj, xlabel, ylabel, title, outfile):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(x_with, y_with, label="With Jastrow")
    plt.plot(x_noj, y_noj, label="Without Jastrow", linestyle="--")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()
    print(f"Saved: {outfile}")


def plot_case(base_dir: Path, N: int, D: int):
    txt_dir = base_dir / "data" / ".txt"
    pdf_dir = base_dir / "data" / ".pdf"

    # z density files
    z_with_file = txt_dir / f"density_z_hs_withJ_mode_bf_N{N}_D{D}.txt"
    z_noj_file  = txt_dir / f"density_z_hs_noJ_mode_bf_N{N}_D{D}.txt"

    # rho density files
    rho_with_file = txt_dir / f"density_rho_hs_withJ_mode_bf_N{N}_D{D}.txt"
    rho_noj_file  = txt_dir / f"density_rho_hs_noJ_mode_bf_N{N}_D{D}.txt"

    for f in [z_with_file, z_noj_file, rho_with_file, rho_noj_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing input file: {f}")

    # Read files
    z_with_x, z_with_y = read_density_file(z_with_file)
    z_noj_x, z_noj_y   = read_density_file(z_noj_file)

    rho_with_x, rho_with_y = read_density_file(rho_with_file)
    rho_noj_x, rho_noj_y   = read_density_file(rho_noj_file)

    # Output PDFs
    z_out = pdf_dir / f"density_z_compare_hs_N{N}_D{D}.pdf"
    rho_out = pdf_dir / f"density_rho_compare_hs_N{N}_D{D}.pdf"

    make_single_plot(
        z_with_x, z_with_y,
        z_noj_x, z_noj_y,
        xlabel=r"$z$",
        ylabel=r"$\rho_z(z)$",
        title=rf"One-body density along $z$: $N={N}$, $D={D}$",
        outfile=z_out
    )

    make_single_plot(
        rho_with_x, rho_with_y,
        rho_noj_x, rho_noj_y,
        xlabel=r"$\rho=\sqrt{x^2+y^2}$",
        ylabel=r"$\rho_\rho(\rho)$",
        title=rf"One-body density in transverse plane: $N={N}$, $D={D}$",
        outfile=rho_out
    )


def plot_all(base_dir: Path, D: int):
    for N in [10, 50, 100]:
        plot_case(base_dir, N, D)


def main():
    parser = argparse.ArgumentParser(
        description="Plot one-body densities with and without Jastrow factor."
    )
    parser.add_argument(
        "N",
        nargs="?",
        default="all",
        help="Particle number N (10, 50, 100) or 'all'"
    )
    parser.add_argument(
        "--D",
        type=int,
        default=3,
        help="Number of dimensions (default: 3)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]  # P1/

    if str(args.N).lower() == "all":
        plot_all(base_dir, args.D)
    else:
        N = int(args.N)
        plot_case(base_dir, N, args.D)


if __name__ == "__main__":
    main()