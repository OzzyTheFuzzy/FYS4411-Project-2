# write in the terminal: python3 data/.py/plot_cpu_time_bf_vs_is.py

import os
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
N_values = [1, 10, 100, 500]
D_values = [1, 2, 3]

# Change this if you used another dt in the importance-sampling filenames
dt_tag = "0p005"

# -----------------------------
# Paths relative to this script
# -----------------------------
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
txt_dir = base_dir / ".txt"
pdf_dir = base_dir / ".pdf"

output_file = pdf_dir / "cpu_time_vs_N_bf_vs_is.pdf"

# -----------------------------
# Helper to read one result file
# Expected format:
# # N D mode stepParam omega alpha eval energy cpu_seconds acceptance
# 1 1 is 0.005 1 0.5 an 0.5 1.234396715 0.999913
# -----------------------------
def read_result_file(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError(f"No data found in {filepath}")

    parts = lines[-1].split()

    result = {
        "N": int(parts[0]),
        "D": int(parts[1]),
        "mode": parts[2],
        "stepParam": float(parts[3]),
        "omega": float(parts[4]),
        "alpha": float(parts[5]),
        "eval": parts[6],
        "energy": float(parts[7]),
        "cpu_seconds": float(parts[8]),
        "acceptance": float(parts[9]),
    }
    return result

# -----------------------------
# Collect data
# -----------------------------
bf_data = {D: {"N": [], "cpu": []} for D in D_values}
is_data = {D: {"N": [], "cpu": []} for D in D_values}

for D in D_values:
    for N in N_values:
        bf_file = txt_dir / f"final_energy_mode_bf_eval_an_N{N}_D{D}.txt"
        is_file = txt_dir / f"final_energy_mode_is_dt_{dt_tag}_eval_an_N{N}_D{D}.txt"

        if bf_file.exists():
            r = read_result_file(bf_file)
            bf_data[D]["N"].append(r["N"])
            bf_data[D]["cpu"].append(r["cpu_seconds"])
        else:
            print(f"Warning: missing file {bf_file}")

        if is_file.exists():
            r = read_result_file(is_file)
            is_data[D]["N"].append(r["N"])
            is_data[D]["cpu"].append(r["cpu_seconds"])
        else:
            print(f"Warning: missing file {is_file}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7.5, 5.2))

colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

for D in D_values:
    if bf_data[D]["N"]:
        plt.plot(
            bf_data[D]["N"],
            bf_data[D]["cpu"],
            marker="o",
            linestyle="-",
            color=colors[D],
            label=f"Brute force, D={D}",
        )

    if is_data[D]["N"]:
        plt.plot(
            is_data[D]["N"],
            is_data[D]["cpu"],
            marker="s",
            linestyle="--",
            color=colors[D],
            label=f"Importance sampling, D={D}",
        )

plt.xlabel("Number of particles, $N$")
plt.ylabel("CPU time (s)")
plt.title("CPU time vs $N$: brute force vs importance sampling")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Optional: uncomment if you prefer logarithmic x-axis
plt.xscale("log")

plt.savefig(output_file)
plt.close()

print(f"Plot saved to {output_file}")