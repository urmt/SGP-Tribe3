import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# GLOBALS
# ==================================================

GLOBAL_SEED = 42
rng = np.random.default_rng(GLOBAL_SEED)

N_VALUES = [16, 32, 64]
K_VALUES = np.linspace(0.0, 4.0, 24)

SIGMA = 1.0

DT = 0.05
STEPS = 2500
TRANSIENT = 1200
TRIALS = 5

# ==================================================
# KURAMOTO
# ==================================================

def sim_kuramoto(N, K, seed):

    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2*np.pi, N)
    omega = rng.normal(0, SIGMA, N)

    R_trace = []

    for _ in range(STEPS):

        phase_diff = theta[None, :] - theta[:, None]

        coupling = np.sum(np.sin(phase_diff), axis=1)

        theta += DT * (
            omega +
            (K / N) * coupling
        )

        z = np.mean(np.exp(1j * theta))
        R = np.abs(z)

        R_trace.append(R)

    return np.array(R_trace)

# ==================================================
# RUN SWEEP
# ==================================================

results = {}

for N in N_VALUES:

    mean_Rs = []
    chis = []
    binders = []

    print(f"RUNNING N={N}")

    for K in K_VALUES:

        late_R = []

        for trial in range(TRIALS):

            seed = (
                GLOBAL_SEED
                + N * 1000
                + int(K * 1000)
                + trial * 777
            )

            R_trace = sim_kuramoto(
                N=N,
                K=K,
                seed=seed
            )

            late = R_trace[TRANSIENT:]

            late_R.append(np.mean(late))

        late_R = np.array(late_R)

        mean_R = np.mean(late_R)

        chi = N * np.var(late_R)

        r2 = np.mean(late_R ** 2)
        r4 = np.mean(late_R ** 4)

        if r2 > 0:
            binder = 1.0 - (r4 / (3.0 * (r2 ** 2)))
        else:
            binder = np.nan

        mean_Rs.append(mean_R)
        chis.append(chi)
        binders.append(binder)

        print(
            f"N={N} "
            f"K={K:.2f} "
            f"R={mean_R:.3f} "
            f"chi={chi:.3f} "
            f"binder={binder:.3f}"
        )

    results[N] = {
        "R": np.array(mean_Rs),
        "chi": np.array(chis),
        "binder": np.array(binders),
    }

# ==================================================
# PLOT: ORDER PARAMETER
# ==================================================

plt.figure(figsize=(8,6))

for N in N_VALUES:
    plt.plot(
        K_VALUES,
        results[N]["R"],
        marker='o',
        label=f"N={N}"
    )

plt.xlabel("K")
plt.ylabel("Mean R")
plt.title("Kuramoto Order Parameter")
plt.legend()
plt.grid(True)

plt.savefig(
    "binder_test_order_parameter.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# PLOT: SUSCEPTIBILITY
# ==================================================

plt.figure(figsize=(8,6))

for N in N_VALUES:
    plt.plot(
        K_VALUES,
        results[N]["chi"],
        marker='o',
        label=f"N={N}"
    )

plt.xlabel("K")
plt.ylabel("Susceptibility")
plt.title("Kuramoto Susceptibility")
plt.legend()
plt.grid(True)

plt.savefig(
    "binder_test_susceptibility.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# PLOT: BINDER
# ==================================================

plt.figure(figsize=(8,6))

for N in N_VALUES:
    plt.plot(
        K_VALUES,
        results[N]["binder"],
        marker='o',
        label=f"N={N}"
    )

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel("K")
plt.ylabel("Binder Cumulant")
plt.title("Binder Crossing Test")
plt.legend()
plt.grid(True)

plt.savefig(
    "binder_test_crossing.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ==================================================
# SUMMARY
# ==================================================

print("\nDONE")

for N in N_VALUES:

    peak_idx = np.argmax(results[N]["chi"])

    print(
        f"N={N} "
        f"peak_chi={results[N]['chi'][peak_idx]:.3f} "
        f"K_peak={K_VALUES[peak_idx]:.3f}"
    )

print("\nFILES:")
print("binder_test_order_parameter.png")
print("binder_test_susceptibility.png")
print("binder_test_crossing.png")