#!/usr/bin/env python3
"""
Common Analysis Functions for Multiscale Dimensionality Pipeline

This module provides reusable functions for:
- Local covariance computation
- Participation ratio (D_eff)
- Beta dynamics
- Normalization
- Curve alignment
- Synthetic data generation

All functions are deterministic and reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, linregress
from scipy.linalg import sqrtm
import json
from pathlib import Path

N_SAMPLES = 2000
K_VALUES = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

def compute_local_covariance(points, k):
    """
    Compute local covariance for all points using k-nearest neighbors.
    
    Parameters:
    -----------
    points : ndarray (n_samples, n_dims)
        Data points
    k : int
        Number of nearest neighbors
        
    Returns:
    --------
    D_eff_values : ndarray
        Effective dimensionality for each point
    """
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

def participation_ratio(cov_matrix):
    """
    Compute participation ratio from covariance matrix.
    
    Parameters:
    -----------
    cov_matrix : ndarray
        Covariance matrix
        
    Returns:
    --------
    D_eff : float
        Effective dimensionality (participation ratio)
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) > 0:
        D_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        D_eff = 1.0
    
    return D_eff

def compute_D_eff_curve(points, k_values=K_VALUES, n_samples=N_SAMPLES, seed=42):
    """
    Compute D_eff(k) curve for a dataset.
    
    Parameters:
    -----------
    points : ndarray
        Data points (n_samples, n_dims)
    k_values : list
        Neighborhood sizes to evaluate
    n_samples : int
        Number of samples to use (None for all)
    seed : int
        Random seed
        
    Returns:
    --------
    D_eff_dict : dict
        Dictionary mapping k -> mean D_eff
    """
    set_seed(seed)
    
    if n_samples is not None and len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        points_sub = points[indices]
    else:
        points_sub = points
    
    D_eff_dict = {}
    for k in k_values:
        k_actual = min(k, len(points_sub) - 1)
        D_eff = compute_local_covariance(points_sub, k_actual)
        D_eff_dict[k] = np.mean(D_eff)
    
    return D_eff_dict

def compute_beta(D_eff_dict, k_values=K_VALUES):
    """
    Compute beta(k) = dD_eff/dlog(k) using sliding window.
    
    Parameters:
    -----------
    D_eff_dict : dict
        Dictionary mapping k -> mean D_eff
    k_values : list
        Neighborhood sizes
        
    Returns:
    --------
    beta_dict : dict
        Dictionary mapping k -> beta value
    beta_gradient : float
        Mean beta gradient (dbeta/dlog(k))
    """
    k_vals = sorted(D_eff_dict.keys())
    D_vals = [D_eff_dict[k] for k in k_vals]
    log_k = np.log(k_vals)
    
    beta_dict = {}
    for i, k in enumerate(k_vals[:-1]):
        if i < len(D_vals) - 1:
            delta_D = D_vals[i+1] - D_vals[i]
            delta_log_k = log_k[i+1] - log_k[i]
            if delta_log_k > 0:
                beta_dict[k] = delta_D / delta_log_k
            else:
                beta_dict[k] = np.nan
    
    log_k_arr = np.array(log_k)
    beta_arr = np.array([beta_dict.get(k, np.nan) for k in k_vals[:-1]])
    valid_idx = ~np.isnan(beta_arr)
    
    if np.sum(valid_idx) >= 2:
        slope, _, _, _, _ = linregress(log_k_arr[valid_idx][:-1], beta_arr[valid_idx][1:])
        beta_gradient = slope
    else:
        beta_gradient = np.nan
    
    return beta_dict, beta_gradient

def normalize_curve(D_eff_dict, k_values=K_VALUES):
    """
    Normalize D_eff curve by max value.
    
    Parameters:
    -----------
    D_eff_dict : dict
        Dictionary mapping k -> mean D_eff
    k_values : list
        Neighborhood sizes
        
    Returns:
    --------
    norm_dict : dict
        Normalized D_eff values
    D_max : float
        Maximum D_eff value
    """
    k_vals = sorted(D_eff_dict.keys())
    D_vals = np.array([D_eff_dict[k] for k in k_vals])
    D_max = np.max(D_vals)
    
    if D_max > 0:
        norm_vals = D_vals / D_max
    else:
        norm_vals = D_vals
    
    norm_dict = {k: norm_vals[i] for i, k in enumerate(k_vals)}
    
    return norm_dict, D_max

def compute_alignment(curve1, curve2, k_values=K_VALUES):
    """
    Compute Pearson correlation between two normalized curves.
    
    Parameters:
    -----------
    curve1, curve2 : dict
        Normalized D_eff dictionaries
    k_values : list
        Common k values
        
    Returns:
    --------
    r : float
        Pearson correlation
    """
    k_common = sorted(set(curve1.keys()) & set(curve2.keys()))
    v1 = np.array([curve1[k] for k in k_common])
    v2 = np.array([curve2[k] for k in k_common])
    
    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    
    r, _ = pearsonr(v1, v2)
    return r

def compute_saturation_ratio(D_eff_dict, k_values=K_VALUES):
    """
    Compute saturation ratio: D_eff at large k / D_max.
    """
    k_vals = sorted(D_eff_dict.keys())
    top_k = k_vals[-3:]
    D_mean_top = np.mean([D_eff_dict[k] for k in top_k])
    D_max = max([D_eff_dict[k] for k in k_vals])
    
    if D_max > 0:
        return D_mean_top / D_max
    return 1.0

def compute_growth_rate(D_eff_dict, k_range=(5, 50)):
    """
    Compute growth rate in log-log space for early k values.
    """
    k_vals = sorted([k for k in D_eff_dict.keys() if k <= k_range[1]])
    D_vals = np.array([D_eff_dict[k] for k in k_vals])
    log_k = np.log(k_vals)
    
    if len(k_vals) >= 2 and np.std(D_vals) > 0:
        slope, _, _, _, _ = linregress(log_k, D_vals)
        return slope
    return np.nan

def compute_spectral_entropy(cov_matrix):
    """
    Compute spectral entropy from covariance matrix.
    
    Parameters:
    -----------
    cov_matrix : ndarray
        Covariance matrix
        
    Returns:
    --------
    H : float
        Spectral entropy
    """
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    H = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
    
    return H

def generate_isotropic_gaussian(n_samples=N_SAMPLES, n_dims=9, seed=42):
    """Generate isotropic Gaussian data."""
    set_seed(seed)
    return np.random.randn(n_samples, n_dims)

def generate_spectrum_matched_gaussian(target_eigenvalues, n_samples=N_SAMPLES, seed=42):
    """
    Generate Gaussian data with specified eigenvalue spectrum.
    
    Parameters:
    -----------
    target_eigenvalues : ndarray
        Target eigenvalues
    n_samples : int
        Number of samples
    seed : int
        Random seed
        
    Returns:
    --------
    data : ndarray
        Generated data
    """
    set_seed(seed)
    n_dims = len(target_eigenvalues)
    
    eigenvalues = np.sort(target_eigenvalues)[::-1]
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    eigenvalues = eigenvalues * n_dims
    
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    X = np.random.randn(n_samples, n_dims)
    data = X @ Q.T @ L
    
    return data

def generate_correlated_gaussian(n_samples=N_SAMPLES, n_dims=50, decay_rate=0.1, seed=42):
    """Generate Gaussian with exponentially decaying eigenvalues."""
    set_seed(seed)
    
    eigenvalues = np.exp(-decay_rate * np.arange(n_dims))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    X = np.random.randn(n_samples, n_dims)
    data = X @ Q.T @ L
    
    return data

def generate_sparse_data(n_samples=N_SAMPLES, n_dims=50, sparsity=0.9, seed=42):
    """Generate sparse Gaussian data."""
    set_seed(seed)
    
    X = np.random.randn(n_samples, n_dims)
    mask = np.random.rand(n_samples, n_dims) < sparsity
    X[mask] = 0
    
    return X

def generate_curved_manifold(n_samples=N_SAMPLES, n_dims=50, n_manifold=5, curvature=0.1, seed=42):
    """Generate data on a curved manifold."""
    set_seed(seed)
    
    t = np.random.randn(n_samples, n_manifold)
    for i in range(n_manifold):
        t[:, i] = t[:, i] / (1 + curvature * np.sum(t[:, :i]**2, axis=1))
    
    B = np.random.randn(n_manifold, n_dims)
    B = B / np.linalg.norm(B, axis=0)
    
    noise = 0.01 * np.random.randn(n_samples, n_dims)
    data = t @ B + noise
    
    return data

def generate_powerlaw_eigenvalues(n_dims=50, exponent=1.0):
    """Generate eigenvalues following power law decay."""
    eigenvalues = np.arange(1, n_dims + 1) ** (-exponent)
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    return eigenvalues

def save_results(results, filepath):
    """Save results to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.csv':
        df = pd.DataFrame(results)
        df.to_csv(filepath)
    elif filepath.suffix == '.npy':
        np.save(filepath, results)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    return filepath

def load_results(filepath):
    """Load results from file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath, index_col=0)
    elif filepath.suffix == '.npy':
        return np.load(filepath, allow_pickle=True).item()
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    return None

def check_validity(D_eff_dict):
    """
    Validate D_eff results.
    
    Returns:
    --------
    is_valid : bool
        True if results are valid
    issues : list
        List of issues found
    """
    issues = []
    
    k_vals = sorted(D_eff_dict.keys())
    D_vals = [D_eff_dict[k] for k in k_vals]
    
    if any(np.isnan(D_vals)):
        issues.append("Contains NaN values")
    
    if any(np.array(D_vals) < 0):
        issues.append("Contains negative D_eff values")
    
    if any(np.isinf(D_vals)):
        issues.append("Contains infinite values")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def log_experiment(name, params, results):
    """Log experiment results."""
    log_entry = {
        'experiment': name,
        'parameters': params,
        'summary': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in results.items()}
    }
    return log_entry

if __name__ == "__main__":
    print("Common Analysis Functions Module")
    print("=" * 50)
    print(f"N_SAMPLES: {N_SAMPLES}")
    print(f"K_VALUES: {K_VALUES}")
    print("\nAvailable functions:")
    print("  - compute_local_covariance()")
    print("  - participation_ratio()")
    print("  - compute_D_eff_curve()")
    print("  - compute_beta()")
    print("  - normalize_curve()")
    print("  - compute_alignment()")
    print("  - generate_*() [data generators]")
