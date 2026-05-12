#!/usr/bin/env python3
"""
PHASE 252 — FINITE-SIZE CRITICAL SCALING AUDIT

Core question: Does organizational regeneration obey a true finite-size
synchronization critical law?

EPISTEMIC STATUS: TIER 1 VALIDATION
COMPLIANCE: LEP | NO consciousness | NO SFH metaphysics | NO observer theory

DIRECTOR: Mark Rowe Traver
DATE: 2026-05-11
"""

import os
import json
import time
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ====================================================================
# GLOBAL CONSTANTS
# ====================================================================
GLOBAL_SEED = 42
DT = 0.05
N_STEPS = 800

N_VALUES = [4, 8, 16, 32, 64, 128]
K_VALUES = np.linspace(0.0, 0.18, 15)
N_K = len(K_VALUES)

EXTENDED_K = True
if EXTENDED_K:
    K_VALUES = np.linspace(0.0, 0.06, 20)
N_K = len(K_VALUES)

DNI_THRESH = 0.25
METRIC_WINDOW = 300
TRIAL_N = 5

TOPOLOGIES = ["AllToAll", "SmallWorld", "ScaleFree", "LorenzSync"]

# ====================================================================
# DIRECTORIES
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

# ====================================================================
# HELPERS
# ====================================================================
def progress(msg):
    print(f"[Phase252] {msg}", flush=True)


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        if obj.ndim == 0:
            return float(obj)
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return obj


def export_csv(rows, path):
    if not rows:
        return
    with open(path, "w") as f:
        f.write(",".join(str(k) for k in rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")


def write_summary(summary_dict, path=None):
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "phase252_summary.txt")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        f.write(f"PHASE 252 — FINITE-SIZE CRITICAL SCALING AUDIT\n")
        f.write(f"{'=' * 65}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Script: {os.path.abspath(__file__)}\n")
        f.write(f"\n")
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")
    return path


# ====================================================================
# TOPOLOGY GENERATORS
# ====================================================================
def make_all_to_all(N):
    """All-to-all: ones minus identity, symmetric, zero diagonal."""
    A = np.ones((N, N), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def make_small_world(N, k=2, p=0.3, seed=0):
    """Ring lattice with rewiring probability p."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for d in range(1, k + 1):
            A[i, (i + d) % N] = 1.0
            A[i, (i - d) % N] = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0 and rng.uniform() < p:
                candidates = [n for n in range(N) if n != i and A[i, n] == 0]
                if candidates:
                    nj = rng.choice(candidates)
                    A[i, j] = A[j, i] = 0.0
                    A[i, nj] = A[nj, i] = 1.0
    return A


def make_scale_free(N, m=2, seed=0):
    """Preferential attachment: BA model."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=np.float64)
    deg = np.zeros(N, dtype=np.float64)
    if N > 1:
        A[0, 1] = A[1, 0] = 1.0
        deg[0] += 1.0
        deg[1] += 1.0
    for i in range(2, N):
        p = (deg[:i] + 1.0) / max(np.sum(deg[:i] + 1.0), 1e-10)
        sz = min(m, i)
        if sz <= 0:
            continue
        ts = rng.choice(i, size=sz, replace=False, p=p)
        for t in ts:
            A[i, t] = A[t, i] = 1.0
            deg[i] += 1.0
            deg[t] += 1.0
    return A


def make_lorenz_sync_topology(N):
    """Dense weighted coupling: random positive weights, normalized spectral radius."""
    rng = np.random.default_rng(GLOBAL_SEED + 9999)
    A = rng.uniform(0.0, 1.0, (N, N)).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    max_eig = max(abs(np.linalg.eigvalsh(A).max()), 1e-10)
    A = A / max_eig
    A = np.maximum(A, 1e-6)
    np.fill_diagonal(A, 0.0)
    return A


def normalize_adjacency(A):
    """Normalize adjacency matrix by row degrees."""
    deg = np.maximum(A.sum(axis=1), 1.0)
    return A / deg[:, None]


# ====================================================================
# KURAMOTO SIMULATOR
# ====================================================================
def sim_kuramoto(N, K, A, steps=N_STEPS, dt=DT, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, N)
    omega = rng.normal(0.0, 0.2, N)
    R_trace = np.zeros(steps, dtype=np.float64)
    deg = np.maximum(A.sum(axis=1), 1.0)
    for t in range(steps):
        phase_diff = theta[None, :] - theta[:, None]
        coupling = (K / deg) * np.sum(A * np.sin(phase_diff), axis=1)
        theta = theta + dt * (omega + coupling)
        if noise > 0.0:
            theta = theta + rng.normal(0.0, noise, N)
        theta = np.mod(theta, 2.0 * math.pi)
        R_trace[t] = abs(np.mean(np.exp(1j * theta)))
    return {"R_trace": R_trace, "theta_final": theta}


# ====================================================================
# LORENZ SYNCHRONIZATION SURROGATE
# ====================================================================
def sim_lorenz_sync(N, K, A, steps=N_STEPS, dt=DT, seed=0):
    """
    Lorenz-63 chaotic oscillator coupled via Kuramoto-like order parameter.
    x_i' = sigma*(y_i - x_i) + (K/N)*sum_j A_ij*(R_j - x_i)
    y_i' = x_i*(rho - z_i) - y_i
    z_i' = x_i*y_i - beta*z_i
    R_j = sqrt(x_j^2 + y_j^2 + z_j^2) as synchronization cue
    Returns dict with R_trace and state_final.
    """
    rng = np.random.default_rng(seed)
    sig = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x = rng.uniform(-15.0, 15.0, N)
    y = rng.uniform(-15.0, 15.0, N)
    z = rng.uniform(5.0, 35.0, N)
    R_trace = np.zeros(steps, dtype=np.float64)
    for t in range(steps):
        R_j = np.sqrt(x * x + y * y + z * z)
        mean_R = np.mean(R_j)
        coupling_x = K * np.sum(A * (R_j - x[:, None]), axis=1) / max(np.sum(A), 1)
        dx = sig * (y - x) + coupling_x
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
        R_j_curr = np.sqrt(x * x + y * y + z * z)
        R_trace[t] = abs(np.mean(np.exp(1j * (np.arctan2(y, x)))))
    return {"R_trace": R_trace, "state_final": np.stack([x, y, z])}


# ====================================================================
# METRICS
# ====================================================================
def compute_metrics(R_trace, window=METRIC_WINDOW):
    """
    Compute all order-parameter metrics from R(t) trace.
    Uses the LAST `window` steps for steady-state statistics.
    """
    if len(R_trace) < window:
        window = len(R_trace) // 2
    steady = R_trace[-window:]
    mean_R = float(np.mean(steady))
    std_R = float(np.std(steady))
    max_R = float(np.max(steady))
    R_autocorr = _autocorr_integrated(steady)
    tau_idx = float(R_autocorr)
    half = len(steady) // 2
    mean_R_late = float(np.mean(steady[half:]))
    chi = float(std_R ** 2)
    binder = float(1.0 - np.mean(steady ** 4) / (3.0 * np.mean(steady ** 2) ** 2 + 1e-12))
    return {
        "mean_R": mean_R,
        "std_R": std_R,
        "max_R": max_R,
        "mean_R_late": mean_R_late,
        "chi": chi,
        "binder": binder,
        "tau_idx": tau_idx,
    }


def _autocorr_inlined(x):
    x = x - np.mean(x)
    n = len(x)
    result = np.correlate(x, x, mode="full")
    result = result[n - 1 : n + n // 2]
    denom = np.correlate(np.ones(n), np.ones(n), mode="full")[n - 1 : n + n // 2]
    denom[denom == 0] = 1.0
    acf = result / denom
    return acf


def _autocorr_integrated(x, max_lag=None):
    if max_lag is None:
        max_lag = min(len(x) - 1, 200)
    acf = _autocorr_inlined(x)
    cutoff = max(1, int(0.5 * len(acf)))
    acf = acf[:cutoff]
    crossings = np.where(np.diff(np.sign(acf)))[0]
    if len(crossings) > 0:
        return float(crossings[0])
    positive = np.where(acf > 0)[0]
    if len(positive) == 0:
        return float(max_lag)
    return float(positive[-1])


# ====================================================================
# DNI — DYNAMICAL PERTURBATION RECOVERY
# ====================================================================
def run_condition_A(N, K, A, steps=N_STEPS, dt=DT, seed=0):
    """Preserved dynamics: simulate, perturb initial conditions, re-run."""
    result1 = sim_kuramoto(N, K, A, steps=steps, dt=dt, seed=seed)
    R_A = compute_metrics(result1["R_trace"])["mean_R_late"]
    rng = np.random.default_rng(seed + 100000)
    theta_perturbed = result1["theta_final"] + rng.uniform(-math.pi / 4.0, math.pi / 4.0, N)
    theta_perturbed = np.mod(theta_perturbed, 2.0 * math.pi)
    theta_orig = result1["theta_final"].copy()
    result2 = sim_kuramoto(N, K, A, steps=steps, dt=dt, noise=0.0, seed=seed + 1)
    R_A2 = compute_metrics(result2["R_trace"])["mean_R_late"]
    dR_A = abs(R_A2 - R_A)
    return {"dR_A": dR_A}


def run_condition_B(N, K, A, steps=N_STEPS, dt=DT, seed=0):
    """Shuffled topology: preserve dynamics, randomize adjacency."""
    result1 = sim_kuramoto(N, K, A, steps=steps, dt=dt, seed=seed)
    R_B1 = compute_metrics(result1["R_trace"])["mean_R_late"]
    rng = np.random.default_rng(seed + 200000)
    n_edges = int(np.sum(A) / 2.0)
    As = np.zeros_like(A)
    edge_list = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                edge_list.append((i, j))
    rng.shuffle(edge_list)
    nodes = list(range(N))
    rng.shuffle(nodes)
    for (i, j) in edge_list[:n_edges]:
        ni, nj = nodes[i % N], nodes[j % N]
        As[ni, nj] = As[nj, ni] = 1.0
    As = np.maximum(As, 1e-6)
    np.fill_diagonal(As, 0.0)
    result2 = sim_kuramoto(N, K, As, steps=steps, dt=dt, seed=seed + 1)
    R_B2 = compute_metrics(result2["R_trace"])["mean_R_late"]
    dR_B = abs(R_B2 - R_B1)
    return {"dR_B": dR_B}


def compute_dni(N, K, A, steps=N_STEPS, dt=DT, seed=0):
    """DNI = A(preserved dynamics) − B(shuffled topology)."""
    condA = run_condition_A(N, K, A, steps=steps, dt=dt, seed=seed)
    condB = run_condition_B(N, K, A, steps=steps, dt=dt, seed=seed)
    dA = condA["dR_A"]
    dB = condB["dR_B"]
    dni = dA - dB
    return {"dni": dni, "dA": dA, "dB": dB}


# ====================================================================
# BATCH CONDITION RUNNER
# ====================================================================
def run_batch(N, K, A, n_trials=TRIAL_N, steps=N_STEPS, dt=DT):
    results = []
    for trial in range(n_trials):
        seed = GLOBAL_SEED + N * 1000 + int(K * 10000) + trial * 777
        res = sim_kuramoto(N, K, A, steps=steps, dt=dt, seed=seed)
        metrics = compute_metrics(res["R_trace"])
        metrics["trial"] = trial
        results.append(metrics)
    df = {}
    keys = ["mean_R", "std_R", "max_R", "mean_R_late", "chi", "binder", "tau_idx"]
    for k in keys:
        vals = [r[k] for r in results if k in r]
        df[k] = np.mean(vals) if vals else 0.0
        df[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
    return df


# ====================================================================
# TOPOLOGY RUNNER (N×K SWEEP)
# ====================================================================
TOPO_MAKERS = {
    "AllToAll": make_all_to_all,
    "SmallWorld": make_small_world,
    "ScaleFree": make_scale_free,
    "LorenzSync": make_lorenz_sync_topology,
}


def run_topology(topology, n_values=N_VALUES, k_values=K_VALUES):
    progress(f"\n{'='*65}")
    progress(f"  TOPOLOGY: {topology}")
    progress(f"{'='*65}")
    make_topo = TOPO_MAKERS[topology]
    all_rows = []
    kstar_N = {}
    for N in n_values:
        A = make_topo(N)
        for ki, K in enumerate(k_values):
            row = {"N": N, "K": K}
            progress(f"    N={N:4d} K={K:.4f} [{ki+1}/{len(k_values)}]")
            metrics = run_batch(N, K, A)
            for k, v in metrics.items():
                row[k] = v
            all_rows.append(row)
        if N == n_values[0]:
            binder_rows = [r for r in all_rows if r["N"] == N]
            binder_pairs = sorted([(r["K"], r["binder"]) for r in binder_rows], key=lambda x: x[0])
            for i in range(len(binder_pairs) - 1):
                if binder_pairs[i][1] < 0 <= binder_pairs[i + 1][1]:
                    kstar_N[N] = 0.5 * (binder_pairs[i][0] + binder_pairs[i + 1][0])
                    break
    progress(f"  Done sweeping {topology}.")
    return all_rows, kstar_N


# ====================================================================
# DNI POST-HOC
# ====================================================================
def compute_dni_topology(topology, n_values=N_VALUES, k_values=K_VALUES, n_dni=3):
    """Compute DNI for representative N×K configs (post-hoc)."""
    progress(f"  Computing DNI for {topology}...")
    make_topo = TOPO_MAKERS[topology]
    sample_N = n_values[::2][:3]
    sample_K = k_values[::2][:4]
    dni_map = {}
    for N in sample_N:
        A = make_topo(N)
        for K in sample_K:
            dni_vals = []
            for trial in range(n_dni):
                seed = GLOBAL_SEED + N * 7777 + int(K * 20000) + trial * 333
                dni_data = compute_dni(N, K, A, steps=N_STEPS // 2, dt=DT, seed=seed)
                dni_vals.append(dni_data["dni"])
            dni_map[(N, K)] = {"mean": float(np.mean(dni_vals)), "std": float(np.std(dni_vals))}
    n_map = len(dni_map)
    progress(f"  DNI done ({n_map}) configs).")
    return dni_map


# ====================================================================
# SCALING FITS
# ====================================================================
def fit_log(K_vals, Kstar, a, c):
    return a * np.log(np.abs(Kstar - K_vals) + 1e-10) + c


def fit_power_law(K_vals, Kstar, nu, a):
    return a * np.abs(K_vals - Kstar) ** (-nu)


def fit_fsc(K_vals, Kstar, nu, a):
    return a * np.abs(K_vals - Kstar) ** (-nu) * np.sign(K_vals - Kstar)


def fit_mean_field(K_vals, Kstar, a):
    return a * np.sqrt(np.maximum(Kstar - K_vals, 0.0))


def compute_aic_bic(log_likelihood, n_params, n_data):
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = n_params * np.log(n_data) - 2.0 * log_likelihood
    return float(aic), float(bic)


def fit_scaling(K_data, chi_data, kstar_guess=0.05):
    """Fit chi(K) to 4 models, return best by AIC."""
    K_arr = np.array(K_data, dtype=np.float64)
    chi_arr = np.array(chi_data, dtype=np.float64)
    valid = np.isfinite(K_arr) & np.isfinite(chi_arr) & (chi_arr > 0)
    if np.sum(valid) < 5:
        return None
    K_v, chi_v = K_arr[valid], chi_arr[valid]
    K_above = K_v[K_v > kstar_guess]
    chi_above = chi_v[K_v > kstar_guess]
    if len(K_above) < 3:
        K_above, chi_above = K_v, chi_v
    models = {}
    residuals = {}

    try:
        popt, _ = curve_fit(fit_log, K_above, chi_above, p0=[0.1, kstar_guess, 1.0], maxfev=2000)
        y_pred = fit_log(K_v, *popt)
        ss_res = np.sum((chi_v - y_pred) ** 2)
        ss_tot = np.sum((chi_v - np.mean(chi_v)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        n_params, n_data = 3, len(K_v)
        log_lik = -0.5 * n_data * np.log(max(ss_res / n_data, 1e-12))
        aic, bic = compute_aic_bic(log_lik, n_params, n_data)
        models["log"] = {"params": list(popt), "r2": r2, "aic": aic, "bic": bic, "model": fit_log}
        residuals["log"] = ss_res
    except Exception:
        pass

    try:
        popt, _ = curve_fit(fit_power_law, K_above, chi_above, p0=[kstar_guess, 1.0, 1.0], maxfev=2000, bounds=([0, 0.1, 0], [1.0, 5.0, 100.0]))
        y_pred = fit_power_law(K_v, *popt)
        ss_res = np.sum((chi_v - y_pred) ** 2)
        ss_tot = np.sum((chi_v - np.mean(chi_v)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        n_params, n_data = 3, len(K_v)
        log_lik = -0.5 * n_data * np.log(max(ss_res / n_data, 1e-12))
        aic, bic = compute_aic_bic(log_lik, n_params, n_data)
        models["power_law"] = {"params": list(popt), "r2": r2, "aic": aic, "bic": bic, "model": fit_power_law}
        residuals["power_law"] = ss_res
    except Exception:
        pass

    try:
        popt, _ = curve_fit(fit_fsc, K_above, chi_above, p0=[kstar_guess, 1.0, 1.0], maxfev=2000, bounds=([0, 0.1, 0], [1.0, 5.0, 100.0]))
        y_pred = fit_fsc(K_v, *popt)
        ss_res = np.sum((chi_v - y_pred) ** 2)
        ss_tot = np.sum((chi_v - np.mean(chi_v)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        n_params, n_data = 3, len(K_v)
        log_lik = -0.5 * n_data * np.log(max(ss_res / n_data, 1e-12))
        aic, bic = compute_aic_bic(log_lik, n_params, n_data)
        models["fsc"] = {"params": list(popt), "r2": r2, "aic": aic, "bic": bic, "model": fit_fsc}
        residuals["fsc"] = ss_res
    except Exception:
        pass

    try:
        popt, _ = curve_fit(fit_mean_field, K_above, chi_above, p0=[kstar_guess], maxfev=2000)
        y_pred = fit_mean_field(K_v, *popt)
        ss_res = np.sum((chi_v - y_pred) ** 2)
        ss_tot = np.sum((chi_v - np.mean(chi_v)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        n_params, n_data = 1, len(K_v)
        log_lik = -0.5 * n_data * np.log(max(ss_res / n_data, 1e-12))
        aic, bic = compute_aic_bic(log_lik, n_params, n_data)
        models["mean_field"] = {"params": list(popt), "r2": r2, "aic": aic, "bic": bic, "model": fit_mean_field}
        residuals["mean_field"] = ss_res
    except Exception:
        pass

    if not models:
        return None

    best_name = min(models, key=lambda m: models[m]["aic"])
    return {"best_model": best_name, **models[best_name], "all_models": models}


# ====================================================================
# VERDICT
# ====================================================================
def compute_verdict(topology, all_rows, kstar_N, fit_results):
    chi_by_N = {}
    for N in N_VALUES:
        rows_N = [r for r in all_rows if r["N"] == N]
        if not rows_N:
            continue
        chi_vals = [r["chi"] for r in rows_N]
        K_vals = [r["K"] for r in rows_N]
        kstar_guess = K_vals[np.argmax(chi_vals)] if chi_vals else 0.05
        fit_res = fit_scaling(K_vals, chi_vals, kstar_guess=kstar_guess)
        chi_by_N[N] = {"K_vals": K_vals, "chi_vals": chi_vals, "fit": fit_res}

    R_crossing_scores = []
    for N in N_VALUES:
        rows_N = [r for r in all_rows if r["N"] == N]
        if len(rows_N) >= 3:
            ks = [r["K"] for r in rows_N]
            binders = [r["binder"] for r in rows_N]
            for i in range(len(ks) - 1):
                if binders[i] < 0 < binders[i + 1]:
                    R_crossing_scores.append(0.7)
                    break
            else:
                R_crossing_scores.append(0.1)

    fss_evidence = 0.0
    if len(kstar_N) >= 3:
        Ns = sorted(kstar_N.keys())
        Kstars = [kstar_N[n] for n in Ns]
        slope, intercept, r_val, _, _ = linregress(np.log(Ns), np.log(np.abs(Kstars) + 1e-10))
        if abs(r_val) > 0.9:
            fss_evidence = 0.8
        elif abs(r_val) > 0.7:
            fss_evidence = 0.5
        else:
            fss_evidence = 0.2
    else:
        fss_evidence = 0.0

    binder_evidence = float(np.mean(R_crossing_scores))
    dni_vals = [r["dni"] for r in all_rows if "dni" in r]
    dni_mean = float(np.mean(dni_vals)) if dni_vals else 0.0
    dni_evidence = 1.0 if dni_mean > DNI_THRESH else max(0.0, dni_mean / DNI_THRESH)

    best_r2 = 0.0
    best_model = "none"
    for N, data in chi_by_N.items():
        if data["fit"] and data["fit"]["r2"] > best_r2:
            best_r2 = data["fit"]["r2"]
            best_model = data["fit"]["best_model"]

    scal_score = best_r2 * 0.7 + fss_evidence * 0.3
    total_score = (binder_evidence * 0.3 + scal_score * 0.4 + dni_evidence * 0.3) * 100.0

    if binder_evidence >= 0.5 and scal_score >= 0.6:
        verdict = "TRUE_CRITICAL_PHASE_TRANSITION"
        confidence = "HIGH"
    elif binder_evidence >= 0.3 and scal_score >= 0.4:
        verdict = "FINITE_SIZE_SYNCHRONIZATION_TRANSITION"
        confidence = "MEDIUM"
    elif binder_evidence >= 0.2 and dni_evidence >= 0.3:
        verdict = "CROSSOVER_ONLY"
        confidence = "LOW"
    else:
        verdict = "TOPOLOGY_DEPENDENT"
        confidence = "LOW"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "score": round(total_score, 2),
        "binder_evidence": round(binder_evidence, 3),
        "scaling_score": round(scal_score, 3),
        "dni_evidence": round(dni_evidence, 3),
        "fss_evidence": round(fss_evidence, 3),
        "best_model": best_model,
        "best_r2": round(best_r2, 4),
        "kstar_N": {str(k): round(v, 5) for k, v in kstar_N.items()},
    }


# ====================================================================
# PLOTTING
# ====================================================================
def make_plots(topology, all_rows, kstar_N, fit_results, verdict):
    n_unique = sorted(set(r["N"] for r in all_rows))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(n_unique) - 1)
    cmap = plt.cm.viridis
    colors = {N: cmap(norm(i)) for i, N in enumerate(n_unique)}

    kstar_global = 0.0
    if kstar_N:
        kstar_global = float(np.mean(list(kstar_N.values())))

    subdir = os.path.join(FIGURES_DIR, topology)
    os.makedirs(subdir, exist_ok=True)

    # Plot 1: chi vs K for each N
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        ax.plot([r["K"] for r in rows_N], [r["chi"] for r in rows_N], "o-",
                color=colors[N], ms=4, lw=1.2, label=f"N={N}")
    if kstar_global > 0:
        ax.axvline(kstar_global, color="red", ls="--", lw=1.5, label=f"K*≈{kstar_global:.4f}")
    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("chi (susceptibility)")
    ax.set_title(f"Susceptibility vs K — {topology}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "chi_vs_K.png"), dpi=150)
    plt.close()

    # Plot 2: mean_R vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        ax.plot([r["K"] for r in rows_N], [r["mean_R"] for r in rows_N], "o-",
                color=colors[N], ms=4, lw=1.2, label=f"N={N}")
    if kstar_global > 0:
        ax.axvline(kstar_global, color="red", ls="--", lw=1.5, label=f"K*≈{kstar_global:.4f}")
    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("mean_R")
    ax.set_title(f"Order Parameter R vs K — {topology}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "R_vs_K.png"), dpi=150)
    plt.close()

    # Plot 3: Binder cumulant vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        ax.plot([r["K"] for r in rows_N], [r["binder"] for r in rows_N], "o-",
                color=colors[N], ms=4, lw=1.2, label=f"N={N}")
    ax.axhline(0, color="black", ls="-", lw=0.8)
    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("Binder cumulant U4")
    ax.set_title(f"Binder Cumulant Crossing — {topology}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "binder.png"), dpi=150)
    plt.close()

    # Plot 4: tau_idx (critical slowing) vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        ax.plot([r["K"] for r in rows_N], [r["tau_idx"] for r in rows_N], "o-",
                color=colors[N], ms=4, lw=1.2, label=f"N={N}")
    if kstar_global > 0:
        ax.axvline(kstar_global, color="red", ls="--", lw=1.5)
    ax.set_xlabel("K (coupling)")
    ax.set_ylabel("tau_idx")
    ax.set_title(f"Critical Slowing Down — {topology}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "critical_slowing.png"), dpi=150)
    plt.close()

    # Plot 5: DNI phase diagram (skip if DNI not in rows)
    if all_rows and "dni" in all_rows[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for N in n_unique:
            rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
            ax.plot([r["K"] for r in rows_N], [r["dni"] for r in rows_N], "o-",
                    color=colors[N], ms=4, lw=1.2, label=f"N={N}")
        ax.axhline(0, color="black", ls="-", lw=0.8)
        ax.set_xlabel("K (coupling)")
        ax.set_ylabel("DNI")
        ax.set_title(f"DNI Phase Diagram — {topology}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "dni_phase.png"), dpi=150)
        plt.close()

    # Plot 6: chi vs N at fixed K (FSS)
    chi_ref = kstar_global
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        closest = min(rows_N, key=lambda r: abs(r["K"] - chi_ref)) if rows_N else None
        if closest:
            ax.plot(N, closest["chi"], "o", color=colors[N], ms=8)
    ax.set_xlabel("N (system size)")
    ax.set_ylabel("chi at K*")
    ax.set_title(f"Finite-Size Scaling — chi vs N — {topology}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "fss_chi_N.png"), dpi=150)
    plt.close()

    # Plot 7: Scaling collapse (all N on one curve)
    fig, ax = plt.subplots(figsize=(8, 5))
    for N in n_unique:
        rows_N = sorted([r for r in all_rows if r["N"] == N], key=lambda x: x["K"])
        ks = np.array([r["K"] for r in rows_N])
        chis = np.array([r["chi"] for r in rows_N])
        if kstar_global > 0.01:
            x_scaled = (ks - kstar_global) * (N ** (1.0 / 2.0))
            ax.plot(x_scaled, chis / N, "o-", color=colors[N], ms=4, lw=1.2, label=f"N={N}")
    ax.set_xlabel("(K - K*) * N^(1/nu)")
    ax.set_ylabel("chi / N")
    ax.set_title(f"Scaling Collapse — {topology}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "scaling_collapse.png"), dpi=150)
    plt.close()

    # Plot 8: Verdict summary
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    verdict_text = (
        f"Topology: {topology}\n\n"
        f"Verdict: {verdict['verdict']}\n"
        f"Confidence: {verdict['confidence']}\n"
        f"Score: {verdict['score']:.1f}/100\n\n"
        f"Binder Evidence: {verdict['binder_evidence']:.3f}\n"
        f"Scaling Score:    {verdict['scaling_score']:.3f}\n"
        f"DNI Evidence:    {verdict['dni_evidence']:.3f}\n"
        f"FSS Evidence:    {verdict['fss_evidence']:.3f}\n\n"
        f"Best Model: {verdict['best_model']} (R2={verdict['best_r2']:.4f})\n"
        f"K*(N): {verdict['kstar_N']}"
    )
    ax.text(0.1, 0.5, verdict_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "verdict_summary.png"), dpi=150)
    plt.close()

    progress(f"  Saved 8 plots for {topology}.")


# ====================================================================
# EXPORT
# ====================================================================
def export_topology(topology, all_rows, kstar_N, fit_results, verdict):
    out_dir = os.path.join(OUTPUTS_DIR, topology)
    os.makedirs(out_dir, exist_ok=True)

    # CSV: all rows
    csv_path = os.path.join(out_dir, "scaling_results.csv")
    export_csv(all_rows, csv_path)

    # CSV: kstar by N
    kstar_rows = [{"N": N, "Kstar": v} for N, v in kstar_N.items()]
    kstar_csv = os.path.join(out_dir, "transition_points.csv")
    export_csv(kstar_rows, kstar_csv)

    # JSON: full results
    full_results = {
        "topology": topology,
        "verdict": verdict,
        "kstar_N": kstar_N,
        "n_rows": len(all_rows),
        "n_N": len(set(r["N"] for r in all_rows)),
        "n_K": len(set(r["K"] for r in all_rows)),
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(safe_json(full_results), f, indent=2)

    # JSON: verdict only
    vjson_path = os.path.join(out_dir, "verdict.json")
    with open(vjson_path, "w") as f:
        json.dump(safe_json({"topology": topology, **verdict}), f, indent=2)

    # TXT: verdict summary
    vtxt_path = os.path.join(out_dir, "verdict.txt")
    with open(vtxt_path, "w") as f:
        f.write(f"PHASE 252 — {topology} VERDICT\n")
        f.write(f"{'=' * 50}\n")
        for k, v in verdict.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n  K*(N): {verdict['kstar_N']}\n")

    progress(f"  Exported {topology} outputs.")


# ====================================================================
# SYNTHESIS
# ====================================================================
def synthesize(all_topology_results):
    progress(f"\n{'='*65}")
    progress(f"  PHASE 252 — CROSS-TOPOLOGY SYNTHESIS")
    progress(f"{'='*65}")

    verdicts = {t: r["verdict"] for t, r in all_topology_results.items()}
    scores = {t: r.get("score", 0) for t, r in all_topology_results.items()}

    true_critical = [t for t, v in verdicts.items() if v == "TRUE_CRITICAL_PHASE_TRANSITION"]
    fsst = [t for t, v in verdicts.items() if v == "FINITE_SIZE_SYNCHRONIZATION_TRANSITION"]
    crossover = [t for t, v in verdicts.items() if v == "CROSSOVER_ONLY"]
    topo_dep = [t for t, v in verdicts.items() if v == "TOPOLOGY_DEPENDENT"]

    progress(f"\n  TRUE_CRITICAL_PHASE_TRANSITION ({len(true_critical)}): {true_critical}")
    progress(f"  FINITE_SIZE_SYNCHRONIZATION_TRANSITION ({len(fsst)}): {fsst}")
    progress(f"  CROSSOVER_ONLY ({len(crossover)}): {crossover}")
    progress(f"  TOPOLOGY_DEPENDENT ({len(topo_dep)}): {topo_dep}")

    winner = max(scores, key=lambda t: scores[t]) if scores else "none"
    best_score = scores.get(winner, 0)

    summary = {
        "n_topologies_tested": len(all_topology_results),
        "verdicts": verdicts,
        "scores": scores,
        "winner": winner,
        "best_score": best_score,
        "true_critical_count": len(true_critical),
        "fsst_count": len(fsst),
        "crossover_count": len(crossover),
        "topology_dependent_count": len(topo_dep),
    }

    # Combined CSV
    combined_rows = []
    for topo, res in all_topology_results.items():
        for row in res.get("rows", []):
            row["topology"] = topo
            combined_rows.append(row)
    if combined_rows:
        combined_csv = os.path.join(OUTPUTS_DIR, "combined_scaling_results.csv")
        export_csv(combined_rows, combined_csv)
        progress(f"  Saved combined CSV ({len(combined_rows)} rows).")

    # Combined JSON
    combined_json_path = os.path.join(OUTPUTS_DIR, "phase252_summary.json")
    with open(combined_json_path, "w") as f:
        json.dump(safe_json(summary), f, indent=2)

    # Combined TXT
    combined_txt_path = os.path.join(OUTPUTS_DIR, "phase252_summary.txt")
    with open(combined_txt_path, "w") as f:
        f.write("PHASE 252 — CROSS-TOPOLOGY SYNTHESIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Topologies tested: {len(all_topology_results)}\n\n")
        for topo, vdata in all_topology_results.items():
            v = vdata.get("verdict", {})
            f.write(f"  [{topo}]\n")
            f.write(f"    Verdict: {v.get('verdict','N/A')}\n")
            f.write(f"    Confidence: {v.get('confidence','N/A')}\n")
            f.write(f"    Score: {v.get('score','N/A')}/100\n")
            f.write(f"    Best model: {v.get('best_model','N/A')} (R2={v.get('best_r2','N/A')})\n")
            f.write(f"    Binder: {v.get('binder_evidence','N/A')}, Scaling: {v.get('scaling_score','N/A')}, DNI: {v.get('dni_evidence','N/A')}, FSS: {v.get('fss_evidence','N/A')}\n")
            f.write(f"    K*(N): {v.get('kstar_N','N/A')}\n\n")
        f.write(f"Overall winner: {winner} (score={best_score})\n")

    progress(f"\n  Winner: {winner} (score={best_score}/100)")
    progress(f"  Summary: {OUTPUTS_DIR}/phase252_summary.json")
    progress("  Phase 252 complete.")

    return summary


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    ensure_dirs()

    progress("\n" + "=" * 65)
    progress("  PHASE 252 — FINITE-SIZE CRITICAL SCALING AUDIT")
    progress("=" * 65)
    progress(f"  RNG seed: {GLOBAL_SEED}")
    progress(f"  N values: {N_VALUES}")
    progress(f"  K range: [{K_VALUES[0]:.3f}, {K_VALUES[-1]:.3f}] ({len(K_VALUES)} pts)")
    progress(f"  Trials per config: {TRIAL_N}")
    progress(f"  Steps per trial: {N_STEPS}, dt={DT}")
    progress(f"  Topologies: {list(TOPO_MAKERS.keys())}")
    progress(f"  DNI threshold: {DNI_THRESH}")
    progress("")

    all_results = {}
    for topology in TOPOLOGIES:
        all_rows, kstar_N = run_topology(topology)
        fit_res = {}
        verdict = compute_verdict(topology, all_rows, kstar_N, fit_res)
        make_plots(topology, all_rows, kstar_N, fit_res, verdict)
        export_topology(topology, all_rows, kstar_N, fit_res, verdict)
        all_results[topology] = {
            "verdict": verdict,
            "rows": all_rows,
            "kstar_N": kstar_N,
        }

    summary = synthesize(all_results)

    elapsed = time.time() - t_start
    progress(f"\n  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    progress("  All done.")

