#!/usr/bin/env python3
"""
Estimator implementations for ablation study.

This module provides independent implementations of dimensionality estimators
without reusing participation ratio logic inside other estimators.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def participation_ratio(eigenvalues):
    """
    Participation Ratio (PR) - Reference estimator.
    
    D = (sum(λ_i))^2 / sum(λ_i^2)
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Sorted eigenvalues (descending)
        
    Returns:
    --------
    D_pr : float
        Participation ratio dimensionality
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 1.0
    
    D_pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return D_pr


def mle_intrinsic_dimension(distances, k):
    """
    MLE Intrinsic Dimension (Levina & Bickel 2005).
    
    d_i = (k-1) / sum_{j=1}^{k-1} log(T_k / T_j)
    
    Parameters:
    -----------
    distances : ndarray
        Distances to k nearest neighbors (sorted ascending)
    k : int
        Number of neighbors
        
    Returns:
    --------
    D_mle : float
        MLE intrinsic dimension
    """
    if len(distances) < k:
        return np.nan
    
    T = np.sort(distances)[:k]
    T_k = T[-1]
    T_j = T[:-1]
    
    valid_mask = (T_j > 0) & (T_k > T_j)
    T_j = T_j[valid_mask]
    
    if len(T_j) == 0 or T_k <= 0:
        return np.nan
    
    log_ratios = np.log(T_k / T_j)
    log_ratios = log_ratios[log_ratios > 0]
    
    if len(log_ratios) == 0:
        return np.nan
    
    d_i = (k - 1) / np.sum(log_ratios)
    
    if d_i <= 0 or np.isinf(d_i):
        return np.nan
    
    return d_i


def correlation_dimension(distances_matrix, r_values=None):
    """
    Correlation Dimension (Grassberger-Procaccia 1983).
    
    C(r) = fraction of pairs with distance < r
    D = d log(C(r)) / d log(r)
    
    Parameters:
    -----------
    distances_matrix : ndarray
        Pairwise distance matrix
    r_values : ndarray
        Radius values to evaluate
        
    Returns:
    --------
    D_corr : float
        Correlation dimension estimate
    """
    if distances_matrix is None or len(distances_matrix) == 0:
        return np.nan
    
    n = distances_matrix.shape[0]
    
    if n * n < 100:
        return np.nan
    
    upper_tri = distances_matrix[np.triu_indices(n, k=1)]
    
    if len(upper_tri) < 100:
        return np.nan
    
    if r_values is None:
        percentiles = np.linspace(10, 90, 10)
        r_values = np.percentile(upper_tri, percentiles)
    
    r_values = r_values[r_values > 0]
    
    if len(r_values) < 2:
        return np.nan
    
    C_values = []
    for r in r_values:
        C = np.mean(upper_tri < r)
        C_values.append(C)
    
    C_values = np.array(C_values)
    C_values = C_values[C_values > 0]
    r_values = r_values[:len(C_values)]
    
    if len(C_values) < 2:
        return np.nan
    
    log_r = np.log(r_values)
    log_C = np.log(C_values)
    
    valid_mask = np.isfinite(log_r) & np.isfinite(log_C) & (log_C < 0)
    log_r = log_r[valid_mask]
    log_C = log_C[valid_mask]
    
    if len(log_r) < 2:
        return np.nan
    
    coeffs = np.polyfit(log_r, log_C, 1)
    D_corr = coeffs[0]
    
    if D_corr <= 0 or np.isinf(D_corr) or np.isnan(D_corr):
        return np.nan
    
    return D_corr


def pca_rank_threshold(eigenvalues, epsilon=0.01):
    """
    PCA Rank Threshold dimension.
    
    D = number of eigenvalues where μ_i / sum(μ) > ε
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Sorted eigenvalues (descending)
    epsilon : float
        Threshold (default 0.01)
        
    Returns:
    --------
    D_pca : float
        PCA threshold dimensionality
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 1.0
    
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    
    if total <= 0:
        return 1.0
    
    proportions = eigenvalues / total
    D_pca = np.sum(proportions > epsilon)
    
    return float(D_pca)


def spectral_entropy_dimension(eigenvalues):
    """
    Spectral Entropy Dimension.
    
    p_i = μ_i / sum(μ)
    H = - sum(p_i * log(p_i))
    D = exp(H)
    
    Parameters:
    -----------
    eigenvalues : ndarray
        Sorted eigenvalues (descending)
        
    Returns:
    --------
    D_entropy : float
        Entropy-based dimensionality
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 1.0
    
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    
    if total <= 0:
        return 1.0
    
    p = eigenvalues / total
    p = p[p > 0]
    
    if len(p) == 0:
        return 1.0
    
    H = -np.sum(p * np.log(p))
    D_entropy = np.exp(H)
    
    if np.isnan(D_entropy) or np.isinf(D_entropy):
        return 1.0
    
    return D_entropy


def validate_estimator_values(D_values):
    """
    Validate estimator output values.
    
    Returns:
    --------
    is_valid : bool
    issues : list
    """
    issues = []
    
    if isinstance(D_values, (int, float)):
        D_values = [D_values]
    
    D_values = np.array(D_values)
    
    if np.any(np.isnan(D_values)):
        issues.append("Contains NaN values")
    
    if np.any(np.isinf(D_values)):
        issues.append("Contains infinite values")
    
    if np.any(D_values < 0):
        issues.append("Contains negative values")
    
    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    print("Estimator Module Loaded")
    print("=" * 50)
    print("Available estimators:")
    print("  - participation_ratio()")
    print("  - mle_intrinsic_dimension()")
    print("  - correlation_dimension()")
    print("  - pca_rank_threshold()")
    print("  - spectral_entropy_dimension()")
