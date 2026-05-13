import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import time
import sys

# RULES & RESTRICTIONS
# 10. USE FIXED RANDOM SEED = 42 everywhere possible.
np.random.seed(42)

audit_dir = "audit_outputs"
os.makedirs(audit_dir, exist_ok=True)

sys.path.insert(0, "./sgp-tribe3/analysis/estimator_ablation/")
from estimators import participation_ratio

K_VALUES = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_SAMPLES = 500

def model_sigmoid(x, L, k0, beta, b):
    return L / (1 + np.exp(-beta * (x - k0))) + b

def compute_dk_profile(data, k_values, mode='original'):
    n_points = data.shape[0]
    mean_profile = []
    
    # Pre-fit neighbors for modes that need them
    k_max = min(max(k_values), n_points - 1)
    if mode in ['original', 'covariance_scrambling']:
        nbrs = NearestNeighbors(n_neighbors=k_max + 1).fit(data)
        _, all_indices = nbrs.kneighbors(data)
    
    for k in k_values:
        k_use = min(k, n_points - 1)
        point_dims = []
        
        if mode == 'random_neighbors':
            all_indices = np.array([np.random.choice(n_points, k_use, replace=False) for _ in range(n_points)])
        elif mode == 'shuffled':
            nbrs = NearestNeighbors(n_neighbors=k_use + 1).fit(data)
            _, all_indices = nbrs.kneighbors(data)
        
        n_to_compute = min(n_points, 300) 
        for i in range(n_to_compute):
            if mode == 'random_neighbors':
                neighbor_indices = all_indices[i]
            else:
                neighbor_indices = all_indices[i][:k_use]
                
            neighbors = data[neighbor_indices]
            cov = np.cov(neighbors.T)
            
            if mode == 'covariance_scrambling':
                # Scramble the covariance matrix elements
                flat = cov.flatten()
                np.random.shuffle(flat)
                cov = flat.reshape(cov.shape)
                # Ensure it's somewhat symmetric for eigvalsh (though scrambling breaks PD)
                cov = (cov + cov.T) / 2
                
            eigenvals = np.linalg.eigvalsh(cov)
            eigenvals = np.abs(np.sort(eigenvals)[::-1])
            point_dims.append(participation_ratio(eigenvals))
            
        mean_profile.append(np.mean(point_dims))
    return np.array(mean_profile)

def run_section_3():
    start_time = time.time()
    print("Executing Section 3: Null Model Destruction Test...")
    
    # Load TRIBE data
    try:
        raw_data = np.load('./sgp-tribe3/reproducibility/raw_data.npz')
        data_orig = raw_data['sgp_nodes'][:N_SAMPLES]
    except Exception as e:
        print(f"FAILED: Data loading error: {e}")
        return

    k_array = np.array(K_VALUES)
    results = []

    # 1. Original
    print("  Computing original D(k)...")
    profile_orig = compute_dk_profile(data_orig, K_VALUES, mode='original')
    p0 = [np.max(profile_orig), np.median(k_array), 0.1, np.min(profile_orig)]
    bounds = ([0, 0, 0, -np.inf], [np.inf, 1000, 10, np.inf])
    popt_orig, _ = curve_fit(model_sigmoid, k_array, profile_orig, p0=p0, bounds=bounds)
    r2_orig = 1 - np.sum((profile_orig - model_sigmoid(k_array, *popt_orig))**2) / np.sum((profile_orig - np.mean(profile_orig))**2)
    results.append({"Model": "Original", "R2": r2_orig, "Profile": profile_orig.tolist()})

    # 2. Shuffled Manifold
    print("  Computing shuffled manifold D(k)...")
    data_shuffled = data_orig.copy()
    for i in range(data_shuffled.shape[1]):
        np.random.shuffle(data_shuffled[:, i])
    profile_shuffled = compute_dk_profile(data_shuffled, K_VALUES, mode='shuffled')
    try:
        popt, _ = curve_fit(model_sigmoid, k_array, profile_shuffled, p0=p0, bounds=bounds)
        r2_shuffled = 1 - np.sum((profile_shuffled - model_sigmoid(k_array, *popt))**2) / np.sum((profile_shuffled - np.mean(profile_shuffled))**2)
    except:
        r2_shuffled = np.nan
    results.append({"Model": "Shuffled Manifold", "R2": r2_shuffled, "Profile": profile_shuffled.tolist()})

    # 3. Random Neighbor Reassignment
    print("  Computing random neighbor D(k)...")
    profile_rand_nb = compute_dk_profile(data_orig, K_VALUES, mode='random_neighbors')
    try:
        popt, _ = curve_fit(model_sigmoid, k_array, profile_rand_nb, p0=p0, bounds=bounds)
        r2_rand_nb = 1 - np.sum((profile_rand_nb - model_sigmoid(k_array, *popt))**2) / np.sum((profile_rand_nb - np.mean(profile_rand_nb))**2)
    except:
        r2_rand_nb = np.nan
    results.append({"Model": "Random Neighbors", "R2": r2_rand_nb, "Profile": profile_rand_nb.tolist()})

    # 4. Covariance Scrambling
    print("  Computing covariance scrambling D(k)...")
    profile_cov_scramble = compute_dk_profile(data_orig, K_VALUES, mode='covariance_scrambling')
    try:
        popt, _ = curve_fit(model_sigmoid, k_array, profile_cov_scramble, p0=p0, bounds=bounds)
        r2_cov_scramble = 1 - np.sum((profile_cov_scramble - model_sigmoid(k_array, *popt))**2) / np.sum((profile_cov_scramble - np.mean(profile_cov_scramble))**2)
    except:
        r2_cov_scramble = np.nan
    results.append({"Model": "Covariance Scrambling", "R2": r2_cov_scramble, "Profile": profile_cov_scramble.tolist()})

    # Save CSV
    df = pd.DataFrame(results)[["Model", "R2"]]
    df.to_csv(os.path.join(audit_dir, "null_model_results.csv"), index=False)
    print(f"  Saved: {os.path.join(audit_dir, 'null_model_results.csv')}")

    # Figure
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(K_VALUES, res["Profile"], 'o-', label=f"{res['Model']} (R²={res['R2']:.3f})")
    plt.xscale('log')
    plt.xlabel("k")
    plt.ylabel("D(k)")
    plt.title("Figure: Null Model Destruction Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(audit_dir, "Figure_NULL_MODEL_DESTRUCTION.png"), dpi=300)
    plt.close()

    runtime = time.time() - start_time
    print(f"Section 3 Complete. Runtime: {runtime:.2f}s")
    
    # Log execution status
    with open(os.path.join(audit_dir, "execution_notes_section3.txt"), "w") as f:
        f.write(f"Section 3: Null Model Destruction\n")
        f.write(f"Exact filepath: {os.path.join(audit_dir, 'null_model_results.csv')}\n")
        f.write(f"Sample size: {N_SAMPLES} points\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
        f.write(f"Status: Success\n")

if __name__ == "__main__":
    run_section_3()
