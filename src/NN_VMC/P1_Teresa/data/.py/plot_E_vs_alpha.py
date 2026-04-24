# must run this script from P1 
# write in the terminal: python3 data/.py/plot_E_vs_alpha.py --N 1 --D 1
# the previous line is an example to obtain the (N,D)=(100,3) plot. You can change those numbers.

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def read_scan(path):
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        names=["alpha", "energy", "cpu_seconds", "acceptance"]
    )
    return df.sort_values("alpha").reset_index(drop=True)


def dt_tag(dt, ndp=6):
    tag = f"{dt:.{ndp}f}".replace(".", "p").rstrip("0")
    if tag.endswith("p"):
        tag += "0"
    return tag


def find_is_file(N, D, dt):
    candidates = [
        f"data/.txt/E_vs_alpha_mode_is_dt_{dt_tag(dt, 6)}_N{N}_D{D}.txt",
        f"data/.txt/E_vs_alpha_mode_is_dt_{dt_tag(dt, 3)}_N{N}_D{D}.txt",
        f"data/.txt/E_vs_alpha_mode_is_dt_{str(dt).replace('.', 'p')}_N{N}_D{D}.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output pdf path (default: data/.pdf/E_vs_alpha_multi_N{N}_D{D}.pdf)"
    )
    args = parser.parse_args()

    N, D = args.N, args.D
    dt_values = [0.005, 0.01, 0.02, 0.03]

    bf_file = f"data/.txt/E_vs_alpha_mode_bf_N{N}_D{D}.txt"
    if not os.path.exists(bf_file):
        raise FileNotFoundError(f"Missing brute-force file: {bf_file}")

    plt.figure(figsize=(7.0, 5.0))

    # Brute force: gray dashed line
    df_bf = read_scan(bf_file)
    plt.plot(
        df_bf["alpha"],
        df_bf["energy"],
        linestyle="--",
        color="gray",
        marker="o",
        markersize=2.5,
        linewidth=2.0,
        label="Brute force"
    )

    # Importance sampling: markers only
    markers = ["o", "s", "^", "D"]
    for dt, mk in zip(dt_values, markers):
        is_file = find_is_file(N, D, dt)
        if is_file is None:
            print(f"Warning: no file found for dt={dt}")
            continue

        df_is = read_scan(is_file)
        plt.plot(
            df_is["alpha"],
            df_is["energy"],
            linestyle="None",
            marker=mk,
            markersize=3.0,
            label=f"IS, dt={dt}"
        )

    plt.xlabel(r"Variational parameter, $\alpha$")
    plt.ylabel(r"Estimated energy, $\langle E \rangle$")
    plt.title(f"$E(\\alpha)$ for N={N}, D={D}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = args.out or f"data/.pdf/E_vs_alpha_multi_N{N}_D{D}.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()