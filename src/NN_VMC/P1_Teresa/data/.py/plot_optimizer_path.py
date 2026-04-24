# must run this script from P1 
# write in the terminal: python3 data/.py/plot_optimizer_path.py --mode is --dt 0.02 --N 1 --D 1
                    # or: python3 data/.py/plot_optimizer_path.py --mode bf --N 1 --D 1
# the previous line is an example to obtain the (N,D)=(1,1) plot. You can change those numbers.

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def dt_tag(dt: float) -> str:
    s = f"{dt:.6f}".replace(".", "p").rstrip("0")
    if s.endswith("p"):
        s += "0"
    return s


def build_input_path(base_txt: Path, mode: str, N: int, D: int, dt: float | None) -> Path:
    if mode == "is":
        if dt is None:
            raise ValueError("Importance sampling mode requires --dt")
        filename = f"opt_path_mode_is_dt_{dt_tag(dt)}_N{N}_D{D}.txt"
    else:
        filename = f"opt_path_mode_bf_N{N}_D{D}.txt"
    return base_txt / filename


def main():
    parser = argparse.ArgumentParser(
        description="Plot optimizer path from opt_path_mode_...txt files."
    )
    parser.add_argument("--mode", choices=["bf", "is"], required=True,
                        help="Optimization mode: bf or is")
    parser.add_argument("--N", type=int, required=True,
                        help="Number of particles")
    parser.add_argument("--D", type=int, required=True,
                        help="Number of dimensions")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step for importance sampling mode")
    args = parser.parse_args()

    # This script is intended to live in P1/data/.py/
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    txt_dir = data_dir / ".txt"
    pdf_dir = data_dir / ".pdf"

    input_path = build_input_path(txt_dir, args.mode, args.N, args.D, args.dt)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find optimizer file:\n{input_path}")

    # File format:
    # # iter alpha energy gradient H
    df = pd.read_csv(
        input_path,
        delim_whitespace=True,
        comment="#",
        names=["iter", "alpha", "energy", "gradient", "H"]
    )

    # Build output filename suffix
    suffix = f"mode_{args.mode}"
    if args.mode == "is":
        suffix += f"_dt_{dt_tag(args.dt)}"
    suffix += f"_N{args.N}_D{args.D}"

    # -----------------------------
    # Plot alpha vs iteration
    # -----------------------------
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(df["iter"], df["alpha"], marker="o", linewidth=1.2)
    plt.xticks(df["iter"])
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$")
    plt.title(rf"Optimizer convergence of $\alpha$ for $N={args.N}, D={args.D}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out2 = pdf_dir / f"opt_alpha_vs_iter_{suffix}.pdf"
    plt.savefig(out2)
    plt.close()

    print(f"Saved:\n  {out2}")


if __name__ == "__main__":
    main()