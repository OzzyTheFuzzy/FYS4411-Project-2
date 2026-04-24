# must run this script from P1 
# write in the terminal: python3 data/.py/plot_time_vs_N.py

import pandas as pd
import matplotlib.pyplot as plt

# Ask user to select dimension
print("Select the time vs N plot:")
print("  1 - 1D")
print("  2 - 2D")
print("  3 - 3D")
print("  4 - All (1D, 2D, and 3D combined)")

choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

if choice == "1":
    dim = "1D"
    input_file = "data/.txt/1D_localenergy.txt"
    output_file = "data/.pdf/time_vs_N_1D.pdf"
elif choice == "2":
    dim = "2D"
    input_file = "data/.txt/2D_localenergy.txt"
    output_file = "data/.pdf/time_vs_N_2D.pdf"
elif choice == "3":
    dim = "3D"
    input_file = "data/.txt/3D_localenergy.txt"
    output_file = "data/.pdf/time_vs_N_3D.pdf"
elif choice == "4":
    dim = "all"
    output_file = "data/.pdf/time_vs_N_all.pdf"
else:
    print("Invalid choice. Please enter 1, 2, 3, or 4.")
    exit(1)

def read_data(input_file):
    df = pd.read_csv(
        input_file,
        delim_whitespace=True,
        comment="#",
        names=["N", "D", "method", "h", "energy", "cpu_seconds", "acceptance"]
    )
    return df

# Plot
plt.figure(figsize=(6.5, 4.5))

if choice == "4":
    colors = {"1D": ("tab:blue", "tab:cyan"),
              "2D": ("tab:orange", "tab:red"),
              "3D": ("tab:green", "tab:olive")}

    for d in ["1D", "2D", "3D"]:
        df = read_data(f"data/.txt/{d}_localenergy.txt")
        df_analytic  = df[df["method"] == "analytic"].sort_values("N")
        df_numerical = df[df["method"] == "numerical"].sort_values("N")
        c_an, c_nu = colors[d]
        plt.plot(df_analytic["N"],  df_analytic["cpu_seconds"],  marker="o", color=c_an, label=f"{d} – Analytical")
        plt.plot(df_numerical["N"], df_numerical["cpu_seconds"], marker="s", color=c_nu, linestyle="--", label=f"{d} – Numerical")

    plt.title("CPU time vs number of particles (1D, 2D, 3D)")

else:
    df = read_data(input_file)
    df_analytic  = df[df["method"] == "analytic"].sort_values("N")
    df_numerical = df[df["method"] == "numerical"].sort_values("N")
    plt.plot(df_analytic["N"],  df_analytic["cpu_seconds"],  marker="o", label="Analytical Laplacian")
    plt.plot(df_numerical["N"], df_numerical["cpu_seconds"], marker="s", label="Numerical Laplacian")
    plt.title(f"CPU time vs number of particles in {dim}")

plt.xscale("log")
plt.xlabel("Number of particles, $N$")
plt.ylabel("CPU time (s)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

# Save
plt.savefig(output_file)
plt.close()

print(f"Plot saved to {output_file}")