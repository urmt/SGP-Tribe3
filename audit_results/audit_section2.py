import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import time
import json
from pathlib import Path
import sys

# RULES & RESTRICTIONS
# 10. USE FIXED RANDOM SEED = 42 everywhere possible.
np.random.seed(42)

audit_dir = "audit_outputs"
os.makedirs(audit_dir, exist_ok=True)

# Add project paths
sys.path.insert(0, "./sgp-tribe3/analysis/estimator_ablation/")
from estimators import participation_ratio

K_VALUES = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_BOOTSTRAP = 1000
N_RANDOM_INIT = 1000
N_SAMPLES = 500

def model_sigmoid(x, L, k0, beta, b):
    return L / (1 + np.exp(-beta * (x - k0))) + b

def compute_dk_profiles(data, k_values):
    """Compute individual D(i, k) for all points and all k."""
    n_points = data.shape[0]
    profiles = np.zeros((n_points, len(k_values)))
    
    for j, k in enumerate(k_values):
        k_use = min(k, n_points - 1)
        nbrs = NearestNeighbors(n_neighbors=k_use + 1).fit(data)
        _, all_indices = nbrs.kneighbors(data)
        
        for i in range(n_points):
            neighbors = data[all_indices[i]]
            cov = np.cov(neighbors.T)
            eigenvals = np.linalg.eigvalsh(cov)
            eigenvals = np.abs(np.sort(eigenvals)[::-1])
            profiles[i, j] = participation_ratio(eigenvals)
    return profiles

def run_section_2():
    start_time = time.time()
    print("Executing Section 2: Parameter Identifiability...")
    
    # Load TRIBE data
    try:
        raw_data = np.load('./sgp-tribe3/reproducibility/raw_data.npz')
        data = raw_data['sgp_nodes'][:N_SAMPLES]
    except Exception as e:
        print(f"FAILED: Data loading error: {e}")
        return

    print(f"  Computing D(i, k) profiles for {N_SAMPLES} samples...")
    # Compute profiles once
    profiles = compute_dk_profiles(data, K_VALUES)
    k_array = np.array(K_VALUES)
    
    # Original fit
    mean_profile = np.mean(profiles, axis=0)
    p0 = [np.max(mean_profile), np.median(k_array), 0.1, np.min(mean_profile)]
    bounds = ([0, 0, 0, -np.inf], [np.inf, 1000, 10, np.inf])
    
    try:
        popt_orig, pcov_orig = curve_fit(model_sigmoid, k_array, mean_profile, p0=p0, bounds=bounds)
    except Exception as e:
        print(f"FAILED: Original fit failed: {e}")
        return

    # 1. Bootstrap
    print(f"  Running {N_BOOTSTRAP} bootstrap resamples...")
    bootstrap_params = []
    for _ in range(N_BOOTSTRAP):
        resampled_idx = np.random.choice(N_SAMPLES, N_SAMPLES, replace=True)
        resampled_mean = np.mean(profiles[resampled_idx], axis=0)
        try:
            popt, _ = curve_fit(model_sigmoid, k_array, resampled_mean, p0=popt_orig, bounds=bounds)
            bootstrap_params.append(popt)
        except:
            pass
            
    bootstrap_params = np.array(bootstrap_params)
    
    # 2. Randomized Initializations
    print(f"  Running {N_RANDOM_INIT} randomized initializations...")
    random_init_params = []
    success_count = 0
    for _ in range(N_RANDOM_INIT):
        # Randomize p0 around original fit
        p0_rand = popt_orig * (1 + 0.5 * np.random.randn(4))
        # Ensure k0 and beta stay positive in p0
        p0_rand[1] = max(p0_rand[1], 1.0)
        p0_rand[2] = max(p0_rand[2], 0.001)
        try:
            popt, _ = curve_fit(model_sigmoid, k_array, mean_profile, p0=p0_rand, bounds=bounds)
            random_init_params.append(popt)
            success_count += 1
        except:
            pass
            
    random_init_params = np.array(random_init_params)
    
    # Metrics
    cov_matrix = np.cov(bootstrap_params.T)
    corr_matrix = np.corrcoef(bootstrap_params.T)
    ci_lower = np.percentile(bootstrap_params, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_params, 97.5, axis=0)
    convergence_rate = success_count / N_RANDOM_INIT
    failed_fit_rate = (N_BOOTSTRAP - len(bootstrap_params)) / N_BOOTSTRAP
    
    metrics = {
        "parameter_names": ["L", "k0", "beta", "b"],
        "original_params": popt_orig.tolist(),
        "covariance_matrix": cov_matrix.tolist(),
        "correlation_matrix": corr_matrix.tolist(),
        "confidence_intervals": {
            "lower": ci_lower.tolist(),
            "upper": ci_upper.tolist()
        },
        "convergence_rate": convergence_rate,
        "failed_fit_rate": failed_fit_rate
    }
    
    with open(os.path.join(audit_dir, "identifiability_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {os.path.join(audit_dir, 'identifiability_metrics.json')}")

    # Figures
    # 1. Covariance Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(4), metrics["parameter_names"])
    plt.yticks(range(4), metrics["parameter_names"])
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center')
    plt.title("Figure: Parameter Correlation Matrix (Sigmoid)")
    plt.savefig(os.path.join(audit_dir, "Figure_PARAMETER_COVARIANCE.png"), dpi=300)
    plt.close()

    # 2. Parameter Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, name in enumerate(metrics["parameter_names"]):
        ax = axes[i//2, i%2]
        ax.hist(bootstrap_params[:, i], bins=30, color='skyblue', edgecolor='black')
        ax.axvline(popt_orig[i], color='red', linestyle='--', label='Original')
        ax.set_title(f"Distribution: {name}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, "Figure_PARAMETER_DISTRIBUTIONS.png"), dpi=300)
    plt.close()

    # 3. Convergence Diagnostics
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.scatter(np.arange(len(random_init_params)), random_init_params[:, i], alpha=0.3, label=metrics["parameter_names"][i])
    plt.title(f"Convergence Diagnostics (Rate: {convergence_rate:.1%})")
    plt.xlabel("Initialization Index")
    plt.ylabel("Final Parameter Value")
    plt.legend()
    plt.savefig(os.path.join(audit_dir, "Figure_CONVERGENCE_DIAGNOSTICS.png"), dpi=300)
    plt.close()

    runtime = time.time() - start_time
    print(f"Section 2 Complete. Runtime: {runtime:.2f}s")
    
    # Log execution status
    with open(os.path.join(audit_dir, "execution_notes_section2.txt"), "w") as f:
        f.write(f"Section 2: Parameter Identifiability\n")
        f.write(f"Exact filepath: {os.path.join(audit_dir, 'identifiability_metrics.json')}\n")
        f.write(f"Sample size: {N_SAMPLES} points, {N_BOOTSTRAP} bootstrap, {N_RANDOM_INIT} random init\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
        f.write(f"Status: Success\n")

if __name__ == "__main__":
    run_section_2()
