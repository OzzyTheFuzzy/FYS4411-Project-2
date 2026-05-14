# must run this script from P1 
# Run example: python3 data/.py/statistical_analysis.py <path to "e_history" file>
                    
import numpy as np
import argparse
from pathlib import Path


def load_replica_histories(filename: str) -> dict[int, np.ndarray]:
    """
    Read a file with format either

    old format:
    # step local_energy
    1 749.99995695
    2 749.999956226
    ...

    or new format:
    # replica step local_energy
    0 1 1.49794085909
    0 2 1.49780849826
    ...

    Returns:
        dict mapping replica_id -> 1D array of local energies
    """
    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]

    # Old format: treat as a single replica with id 0
    if ncols == 2:
        return {0: data[:, 1].astype(float)}

    # New format: replica, step, local_energy
    if ncols == 3:
        replicas = data[:, 0].astype(int)
        steps = data[:, 1].astype(int)
        energies = data[:, 2].astype(float)

        histories = {}
        for rep in np.unique(replicas):
            mask = (replicas == rep)
            rep_steps = steps[mask]
            rep_energies = energies[mask]

            # sort by step to be safe
            order = np.argsort(rep_steps)
            histories[int(rep)] = rep_energies[order]

        return histories

    raise ValueError(
        f"Unsupported file format: expected 2 or 3 columns, got {ncols}"
    )


def autocorrelation_lag1(x: np.ndarray) -> float:
    """
    Estimate lag-1 autocorrelation coefficient rho_1.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    var = np.var(x, ddof=0)
    if len(x) < 2 or var == 0.0:
        return 0.0
    gamma1 = np.mean((x[:-1] - mu) * (x[1:] - mu))
    return gamma1 / var


def blocking_analysis(x: np.ndarray) -> dict:
    """
    Flyvbjerg-Petersen blocking analysis.

    Returns:
        {
            'mean': mean of original data,
            'variance_mean': blocked estimate of Var(mean),
            'stderr': blocked standard error,
            'k_stop': blocking level selected,
            'n_used': number of data points used,
            'd': number of blocking levels,
            's': variances at each level,
            'gamma': lag-1 autocovariances at each level,
            'M': test statistic array
        }
    """
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least 2 samples for blocking analysis.")

    # Implementation assumes n = 2^d.
    # Truncate to the largest power of two <= len(x).
    d = int(np.floor(np.log2(len(x))))
    n_used = 2 ** d
    x = x[:n_used].copy()

    mu = np.mean(x)

    # Arrays for each blocking level
    s = np.zeros(d)
    gamma = np.zeros(d)

    x_block = x.copy()

    for i in range(d):
        n = len(x_block)

        # lag-1 autocovariance
        gamma[i] = (1.0 / n) * np.sum((x_block[:-1] - mu) * (x_block[1:] - mu))

        # variance
        s[i] = np.var(x_block, ddof=0)

        # block transformation: average neighboring pairs
        x_block = 0.5 * (x_block[0::2] + x_block[1::2])

    # Flyvbjerg-Petersen test statistic
    M = np.cumsum((((gamma / s) ** 2) * 2 ** np.arange(1, d + 1)[::-1])[::-1])[::-1]

    # Tabulated q values from the lecture-note implementation
    q = np.array([
        6.634897, 9.210340, 11.344867, 13.276704, 15.086272,
        16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
        24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
        31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
        38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
        45.641683, 46.962942, 48.278236, 49.587884, 50.892181
    ])

    if d > len(q):
        raise ValueError(
            f"Blocking depth d={d} is larger than available q-table ({len(q)}). "
            "Use fewer samples or extend the q-table."
        )

    k_stop = d - 1
    for k in range(d):
        if M[k] < q[k]:
            k_stop = k
            break

    if k_stop >= d - 1:
        print("Warning: blocking reached the last level. Consider using more data.")

    variance_mean = s[k_stop] / (2 ** (d - k_stop))
    stderr = np.sqrt(variance_mean)

    return {
        "mean": mu,
        "variance_mean": variance_mean,
        "stderr": stderr,
        "k_stop": k_stop,
        "n_used": n_used,
        "d": d,
        "s": s,
        "gamma": gamma,
        "M": M,
    }


def naive_statistics(x: np.ndarray) -> dict:
    """
    Mean and naive standard error assuming independent samples.
    """
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    stderr = np.sqrt(var / len(x))
    return {"mean": mean, "variance": var, "stderr": stderr}


def save_blocking_report(outfile: str, results: dict, naive: dict, rho1: float) -> None:
    with open(outfile, "w") as f:
        f.write("# Blocking analysis report\n")
        f.write(f"n_used {results['n_used']}\n")
        f.write(f"d {results['d']}\n")
        f.write(f"mean {results['mean']:.12f}\n")
        f.write(f"naive_stderr {naive['stderr']:.12e}\n")
        f.write(f"blocked_stderr {results['stderr']:.12e}\n")
        f.write(f"variance_mean_blocked {results['variance_mean']:.12e}\n")
        f.write(f"k_stop {results['k_stop']}\n")
        f.write(f"lag1_corr_original {rho1:.12e}\n")
        f.write("\n")
        f.write("# level variance autocovariance M\n")
        for i in range(results["d"]):
            f.write(
                f"{i} "
                f"{results['s'][i]:.12e} "
                f"{results['gamma'][i]:.12e} "
                f"{results['M'][i]:.12e}\n"
            )


def save_summary_report(outfile: str, summary_rows: list[dict]) -> None:
    with open(outfile, "w") as f:
        f.write("# replica n_read n_used mean naive_stderr blocked_stderr k_stop lag1_corr\n")
        for row in summary_rows:
            f.write(
                f"{row['replica']} "
                f"{row['n_read']} "
                f"{row['n_used']} "
                f"{row['mean']:.12f} "
                f"{row['naive_stderr']:.12e} "
                f"{row['blocked_stderr']:.12e} "
                f"{row['k_stop']} "
                f"{row['rho1']:.12e}\n"
            )

        if len(summary_rows) > 1:
            means = np.array([row["mean"] for row in summary_rows], dtype=float)
            mean_of_means = np.mean(means)
            std_of_means = np.std(means, ddof=1)
            stderr_of_means = std_of_means / np.sqrt(len(means))

            f.write("\n")
            f.write("# combined replica statistics\n")
            f.write(f"n_replicas {len(summary_rows)}\n")
            f.write(f"mean_of_replica_means {mean_of_means:.12f}\n")
            f.write(f"std_of_replica_means {std_of_means:.12e}\n")
            f.write(f"stderr_of_replica_means {stderr_of_means:.12e}\n")


def main():
    parser = argparse.ArgumentParser(description="Blocking analysis for VMC energy-history files.")
    parser.add_argument("infile", help="Input energy history .txt file")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. If not given, uses the same folder as infile."
    )
    args = parser.parse_args()

    infile_path = Path(args.infile)
    outdir = Path(args.outdir) if args.outdir is not None else infile_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    histories = load_replica_histories(args.infile)

    summary_rows = []

    print("=== Blocking analysis by replica ===")
    print(f"Input file: {args.infile}")
    print(f"Replicas found: {len(histories)}")

    for replica in sorted(histories.keys()):
        energies = histories[replica]

        naive = naive_statistics(energies)
        rho1 = autocorrelation_lag1(energies)
        blocked = blocking_analysis(energies)

        report_file = outdir / f"{infile_path.stem}_replica{replica}_blocking_report.txt"
        #save_blocking_report(str(report_file), blocked, naive, rho1)

        summary_rows.append({
            "replica": replica,
            "n_read": len(energies),
            "n_used": blocked["n_used"],
            "mean": blocked["mean"],
            "naive_stderr": naive["stderr"],
            "blocked_stderr": blocked["stderr"],
            "k_stop": blocked["k_stop"],
            "rho1": rho1,
        })

        print(f"\nReplica {replica}")
        print(f"  Samples read         : {len(energies)}")
        print(f"  Samples used         : {blocked['n_used']}  (largest power of 2)")
        print(f"  Mean energy          : {blocked['mean']:.12f}")
        print(f"  Naive stderr         : {naive['stderr']:.12e}")
        print(f"  Blocked stderr       : {blocked['stderr']:.12e}")
        print(f"  Stopping level k     : {blocked['k_stop']}")
        print(f"  Lag-1 corr (original): {rho1:.6e}")

    summary_file = outdir / f"{infile_path.stem}_blocking_summary.txt"
    save_summary_report(str(summary_file), summary_rows)

    print(f"\nSaved summary report to: {summary_file}")

    if len(summary_rows) > 1:
        means = np.array([row["mean"] for row in summary_rows], dtype=float)
        mean_of_means = np.mean(means)
        std_of_means = np.std(means, ddof=1)
        stderr_of_means = std_of_means / np.sqrt(len(means))

        print("\n=== Replica-to-replica summary ===")
        print(f"Mean of replica means      : {mean_of_means:.12f}")
        print(f"Std. dev. of replica means : {std_of_means:.12e}")
        print(f"Std. err. of replica means : {stderr_of_means:.12e}")


if __name__ == "__main__":
    main()