#!/usr/bin/env python3
"""
Experiment Set 2: Controlled Synthetic Variants (Simplified)

This script tests how individual parameters affect the growth-saturation profile.

Output:
- analysis/results/synthetic_sweeps/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common_functions import (
    set_seed, compute_local_covariance, normalize_curve, compute_alignment
)

K_VALUES = [5, 10, 20, 50, 100]
N_SAMPLES = 500

def compute_deff_fast(data, k_values=K_VALUES, n_samples=N_SAMPLES):
    """Fast D_eff computation."""
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        data = data[indices]
    
    D_eff_dict = {}
    for k in k_values:
        k_use = min(k, len(data) - 1)
        D_eff = compute_local_covariance(data, k_use)
        D_eff_dict[k] = np.mean(D_eff)
    
    return D_eff_dict

def run_sweeps():
    """Run simplified sweeps."""
    print("=" * 60)
    print("Experiment Set 2: Synthetic Variants (Simplified)")
    print("=" * 60)
    
    results_dir = Path(__file__).parent / "results" / "synthetic_sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    
    results = []
    
    dims = [5, 10, 20, 50, 100]
    print(f"\nA. Dimensionality sweep...")
    for d in dims:
        data = np.random.randn(N_SAMPLES, d)
        D_eff = compute_deff_fast(data)
        norm, D_max = normalize_curve(D_eff)
        results.append({
            'sweep': 'dimensionality',
            'parameter': d,
            'mean_D_eff': np.mean(list(D_eff.values())),
            'D_max': D_max
        })
        print(f"  d={d}: mean_D_eff={np.mean(list(D_eff.values())):.2f}")
    
    sparsities = [0.0, 0.5, 0.9]
    print(f"\nB. Sparsity sweep...")
    for s in sparsities:
        data = np.random.randn(N_SAMPLES, 50)
        mask = np.random.rand(N_SAMPLES, 50) < s
        data[mask] = 0
        D_eff = compute_deff_fast(data)
        norm, D_max = normalize_curve(D_eff)
        results.append({
            'sweep': 'sparsity',
            'parameter': s,
            'mean_D_eff': np.mean(list(D_eff.values())),
            'D_max': D_max
        })
        print(f"  sparsity={s}: mean_D_eff={np.mean(list(D_eff.values())):.2f}")
    
    decays = [0.02, 0.1, 0.5]
    print(f"\nC. Eigenvalue decay sweep...")
    for decay in decays:
        eigenvalues = np.exp(-decay * np.arange(50))
        L = np.diag(np.sqrt(eigenvalues))
        Q, _ = np.linalg.qr(np.random.randn(50, 50))
        X = np.random.randn(N_SAMPLES, 50)
        data = X @ Q.T @ L
        D_eff = compute_deff_fast(data)
        norm, D_max = normalize_curve(D_eff)
        results.append({
            'sweep': 'decay',
            'parameter': decay,
            'mean_D_eff': np.mean(list(D_eff.values())),
            'D_max': D_max
        })
        print(f"  decay={decay}: mean_D_eff={np.mean(list(D_eff.values())):.2f}")
    
    noises = [0.01, 0.1, 0.5]
    print(f"\nD. Noise sweep...")
    base = np.random.randn(N_SAMPLES, 50)
    for sigma in noises:
        data = base + sigma * np.random.randn(N_SAMPLES, 50)
        D_eff = compute_deff_fast(data)
        norm, D_max = normalize_curve(D_eff)
        results.append({
            'sweep': 'noise',
            'parameter': sigma,
            'mean_D_eff': np.mean(list(D_eff.values())),
            'D_max': D_max
        })
        print(f"  sigma={sigma}: mean_D_eff={np.mean(list(D_eff.values())):.2f}")
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "sweep_summary.csv", index=False)
    
    print(f"\nResults saved to {results_dir}")
    print("\n" + df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    try:
        results = run_sweeps()
        print("\n[SUCCESS] Synthetic sweep experiments completed successfully")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
