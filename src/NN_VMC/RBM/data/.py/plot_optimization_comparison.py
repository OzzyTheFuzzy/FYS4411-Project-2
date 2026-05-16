#!/usr/bin/env python3
"""
Plot several RBM optimization-energy curves for the same (N,D).

Run:
    python3 data/.py/plot_optimization_comparison.py --N 10 --D 3

The script searches recursively inside:
    data/.txt/

for files containing:
    optimization
    _N<N>_D<D>

and saves a PDF figure in:
    data/.pdf/
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def decimal_tag_to_float_string(tag: str) -> str:
    """Convert file tags such as 0p005 or 2p82843 into 0.005 or 2.82843."""
    return tag.replace("p", ".")


def extract_label(path: Path) -> str:
    """
    Build a compact curve label from the filename.

    Example:
    optimization_interac_is_dt_0p02_lr_0p005_Nh_4_gamma_2p82843_N10_D3.txt
    -> IS, lr=0.005, Nh=4, gamma=2.82843
    """
    name = path.stem

    parts = []

    if "_bf_" in name or name.startswith("optimization_bf"):
        parts.append("BF")
    elif "_is_" in name or name.startswith("optimization_is"):
        parts.append("IS")

    lr_match = re.search(r"_lr_([^_]+)", name)
    if lr_match:
        parts.append(f"lr={decimal_tag_to_float_string(lr_match.group(1))}")

    nh_match = re.search(r"_Nh_([0-9]+)", name)
    if nh_match:
        parts.append(f"$N_h$={nh_match.group(1)}")

    gamma_match = re.search(r"_gamma_([^_]+)", name)
    if gamma_match:
        parts.append(r"$\gamma$=" + decimal_tag_to_float_string(gamma_match.group(1)))

    if not parts:
        return name

    return ", ".join(parts)


def load_optimization_file(path: Path):
    """
    Read an optimization file with format:
        # iter energy acceptance cpu_seconds
        0 65.4 0.99 2.9
        ...

    Returns:
        iterations, energies
    """
    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"{path} has fewer than two columns.")

    iterations = data[:, 0].astype(int)
    energies = data[:, 1].astype(float)

    return iterations, energies


def find_project_root() -> Path:
    """
    This script is expected to live in:
        project_root/data/.py/

    Therefore:
        parents[0] -> data/.py
        parents[1] -> data
        parents[2] -> project_root
    """
    return Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description="Plot RBM optimization energy curves for all files matching a given N and D."
    )
    parser.add_argument("--N", type=int, default=None, help="Number of particles")
    parser.add_argument("--D", type=int, default=None, help="Number of dimensions")
    parser.add_argument(
        "--txt-dir",
        type=str,
        default=None,
        help="Optional custom directory containing optimization .txt files",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Optional custom output directory for the PDF figure",
    )
    args = parser.parse_args()

    N = args.N
    D = args.D

    if N is None:
        N = int(input("Enter number of particles N: "))

    if D is None:
        D = int(input("Enter number of dimensions D: "))

    project_root = find_project_root()

    txt_dir = Path(args.txt_dir) if args.txt_dir is not None else project_root / "data" / ".txt" / "non-elliptical_interaction"
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir is not None else project_root / "data" / ".pdf"

    pdf_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*optimization*_N{N}_D{D}*.txt"
    files = sorted(txt_dir.rglob(pattern))

    if not files:
        print(f"No files found in {txt_dir} matching pattern:")
        print(f"  {pattern}")
        return

    print(f"Found {len(files)} optimization file(s):")
    for f in files:
        print(f"  {f}")

    fig, ax = plt.subplots(figsize=(10, 7))

    plotted = 0

    for path in files:
        try:
            iterations, energies = load_optimization_file(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        finite_mask = np.isfinite(energies)

        if not np.any(finite_mask):
            print(f"Skipping {path}: no finite energy values.")
            continue

        label = extract_label(path)

        ax.plot(
            iterations[finite_mask],
            energies[finite_mask],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=label,
        )

        if np.any(~finite_mask):
            first_bad_iter = iterations[~finite_mask][0]
            print(f"Warning: {path.name} contains non-finite energy starting at iteration {first_bad_iter}.")

        best_index = np.argmin(energies[finite_mask])
        best_iter = iterations[finite_mask][best_index]
        best_energy = energies[finite_mask][best_index]
        print(f"Best finite energy for {path.name}: E = {best_energy:.12f} at iter {best_iter}")

        plotted += 1

    if plotted == 0:
        print("No valid curves to plot.")
        return

    ax.set_xlabel("Optimization step",fontsize=20)
    ax.set_ylabel("Energy",fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)

    fig.tight_layout()

    outfile = pdf_dir / f"optimization_energy_comparison_N{N}_D{D}.pdf"
    fig.savefig(outfile, bbox_inches="tight")

    print(f"\nSaved PDF figure to:")
    print(f"  {outfile}")


if __name__ == "__main__":
    main()
