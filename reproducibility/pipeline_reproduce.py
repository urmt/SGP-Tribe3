#!/usr/bin/env python3
"""
Pipeline Reproduction Script for SGP-Tribe3 Scale-Dependent Dimensionality Analysis

This script reproduces key results from the manuscript:
- D_eff(k) curves
- Beta(k) analysis
- Phase model fitting

Usage:
    python pipeline_reproduce.py

Output:
    - D_eff_curves.csv
    - beta_curves.csv
    - phase_model_results.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import sqrtm
from scipy.stats import pearsonr

def load_raw_data():
    """Load raw TRIBE data."""
    data = np.load(Path(__file__).parent / "raw_data.npz")
    return {
        'sgp_nodes': data['sgp_nodes'],
        'streams': data['streams'],
        'edge_weights': data['edge_weights']
    }

def compute_local_covariance(points, k):
    """Compute local covariance for all points."""
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    D_eff_values = []
    for i in range(len(points)):
        neighbor_indices = indices[i]
        neighbors = points[neighbor_indices]
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            D_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            D_eff = 1.0
        D_eff_values.append(D_eff)
    
    return np.array(D_eff_values)

def compute_D_eff_k(data, k_values=[5, 10, 20, 50, 100, 200, 500], n_samples=2000):
    """Compute D_eff(k) for all systems."""
    np.random.seed(42)
    
    results = {}
    for system_name, points in data.items():
        if system_name == 'metadata':
            continue
        
        # Subsample if needed
        if len(points) > n_samples:
            indices = np.random.choice(len(points), n_samples, replace=False)
            points_sub = points[indices]
        else:
            points_sub = points
        
        D_eff_values = {}
        for k in k_values:
            k_actual = min(k, len(points_sub) - 1)
            D_eff = compute_local_covariance(points_sub, k_actual)
            D_eff_values[k] = np.mean(D_eff)
        
        results[system_name] = D_eff_values
        print(f"Computed D_eff(k) for {system_name}")
    
    return results

def compute_beta_k(D_eff_dict, k_values=[5, 10, 20, 50, 100, 200, 500]):
    """Compute beta(k) using sliding window log-log regression."""
    from scipy.stats import linregress
    
    beta_results = {}
    for system, D_eff_values in D_eff_dict.items():
        beta_values = []
        log_k = np.log(k_values)
        
        for i in range(len(k_values) - 2):
            window_k = log_k[i:i+3]
            window_D = [D_eff_values[k] for k in k_values[i:i+3]]
            if len(window_k) >= 2 and np.std(window_D) > 0:
                slope, _, _, _, _ = linregress(window_k, window_D)
                beta_values.append(slope)
            else:
                beta_values.append(np.nan)
        
        beta_results[system] = beta_values
        print(f"Computed beta(k) for {system}")
    
    return beta_results

def fit_phase_model(beta_dict, D_eff_dict, k_values):
    """Fit bilinear phase model: dβ/dlog(k) = a·β + b·D_eff + c·(β·D_eff)"""
    from scipy.optimize import curve_fit
    
    results = {}
    
    for system in beta_dict.keys():
        beta = np.array(beta_dict[system])
        D_eff = np.array([D_eff_dict[system][k] for k in k_values[1:len(beta)+1]])
        
        # Compute dbeta/dlog(k)
        dbeta = np.diff(beta) / np.diff(np.log(k_values[:len(beta)+1]))
        
        # Fit bilinear model
        def bilinear_model(X, a, b, c):
            beta_i, D_eff_i = X
            return a * beta_i + b * D_eff_i + c * (beta_i * D_eff_i)
        
        try:
            popt, pcov = curve_fit(bilinear_model, (beta[:-1], D_eff[:-1]), dbeta, p0=[1, 0, -1])
            residuals = dbeta - bilinear_model((beta[:-1], D_eff[:-1]), *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((dbeta - np.mean(dbeta))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results[system] = {
                'a': popt[0],
                'b': popt[1],
                'c': popt[2],
                'R_squared': r_squared
            }
        except Exception as e:
            results[system] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 'R_squared': np.nan}
        
        print(f"Fitted phase model for {system}: R² = {results[system]['R_squared']:.4f}")
    
    return results

def main():
    print("=" * 60)
    print("SGP-Tribe3 Pipeline Reproduction")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading raw data...")
    data = load_raw_data()
    print(f"  Loaded {len(data['sgp_nodes'])} stimuli")
    
    # Compute D_eff(k)
    print("\n[2/4] Computing D_eff(k) curves...")
    k_values = [5, 10, 20, 50, 100, 200, 500]
    D_eff_results = compute_D_eff_k(data, k_values)
    
    # Save D_eff curves
    df_deff = pd.DataFrame(D_eff_results)
    df_deff.index = k_values
    df_deff.index.name = 'k'
    df_deff.to_csv(Path(__file__).parent / "D_eff_curves.csv")
    print("  Saved D_eff_curves.csv")
    
    # Compute beta(k)
    print("\n[3/4] Computing beta(k) curves...")
    beta_results = compute_beta_k(D_eff_results, k_values)
    
    # Save beta curves
    df_beta = pd.DataFrame(beta_results)
    df_beta.to_csv(Path(__file__).parent / "beta_curves.csv")
    print("  Saved beta_curves.csv")
    
    # Fit phase model
    print("\n[4/4] Fitting phase model...")
    phase_results = fit_phase_model(beta_results, D_eff_results, k_values)
    
    # Save phase model results
    df_phase = pd.DataFrame(phase_results).T
    df_phase.to_csv(Path(__file__).parent / "phase_model_results.csv")
    print("  Saved phase_model_results.csv")
    
    print("\n" + "=" * 60)
    print("Pipeline reproduction complete!")
    print("=" * 60)
    
    # Print summary
    print("\nSummary of Phase Model Fits:")
    for system, params in phase_results.items():
        print(f"  {system}: a={params['a']:.3f}, b={params['b']:.3f}, c={params['c']:.3f}, R²={params['R_squared']:.4f}")

if __name__ == "__main__":
    main()
