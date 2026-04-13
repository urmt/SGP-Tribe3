#!/usr/bin/env python3
"""
Estimator Ablation Study Pipeline

This script evaluates whether the growth-saturation dimensionality profile
is specific to the participation ratio estimator or a more general property.

Output:
- analysis/estimator_ablation/results/*.npy
- analysis/estimator_ablation/results/*.csv
- analysis/estimator_ablation/summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent))
from estimators import (
    participation_ratio, mle_intrinsic_dimension,
    correlation_dimension, pca_rank_threshold,
    spectral_entropy_dimension
)

K_VALUES = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_SAMPLES = 500
SEED = 42

np.random.seed(SEED)

def load_datasets():
    """Load all datasets from reproducibility bundle."""
    print("\n[1/5] Loading datasets...")
    
    data_path = Path(__file__).parent.parent.parent / "reproducibility" / "raw_data.npz"
    raw_data = np.load(data_path)
    
    datasets = {}
    
    try:
        datasets['TRIBE'] = raw_data['sgp_nodes'][:N_SAMPLES]
        print(f"  TRIBE: {datasets['TRIBE'].shape}")
    except Exception as e:
        print(f"  [WARNING] Failed to load TRIBE: {e}")
    
    datasets['Hierarchical'] = generate_hierarchical_gaussian()
    print(f"  Hierarchical: {datasets['Hierarchical'].shape}")
    
    datasets['Correlated'] = generate_correlated_gaussian()
    print(f"  Correlated: {datasets['Correlated'].shape}")
    
    datasets['Sparse'] = generate_sparse_data()
    print(f"  Sparse: {datasets['Sparse'].shape}")
    
    datasets['Manifold'] = generate_curved_manifold()
    print(f"  Manifold: {datasets['Manifold'].shape}")
    
    return datasets


def generate_hierarchical_gaussian():
    """Generate hierarchical Gaussian data."""
    n_dims = 50
    decay = 0.1
    eigenvalues = np.exp(-decay * np.arange(n_dims))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    X = np.random.randn(N_SAMPLES, n_dims)
    
    return X @ Q.T @ L


def generate_correlated_gaussian():
    """Generate correlated Gaussian data."""
    n_dims = 50
    decay = 0.05
    eigenvalues = np.exp(-decay * np.arange(n_dims))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    X = np.random.randn(N_SAMPLES, n_dims)
    
    return X @ Q.T @ L


def generate_sparse_data():
    """Generate sparse data."""
    n_dims = 50
    sparsity = 0.9
    X = np.random.randn(N_SAMPLES, n_dims)
    mask = np.random.rand(N_SAMPLES, n_dims) < sparsity
    X[mask] = 0
    
    return X


def generate_curved_manifold():
    """Generate curved manifold data."""
    n_dims = 50
    n_manifold = 5
    curvature = 0.1
    
    t = np.random.randn(N_SAMPLES, n_manifold)
    for i in range(n_manifold):
        t[:, i] = t[:, i] / (1 + curvature * np.sum(t[:, :i]**2, axis=1))
    
    B = np.random.randn(n_manifold, n_dims)
    B = B / np.linalg.norm(B, axis=0)
    
    noise = 0.01 * np.random.randn(N_SAMPLES, n_dims)
    
    return t @ B + noise


def compute_estimators_for_k(points, k):
    """
    Compute all estimators for a single k value.
    
    Returns:
    --------
    results : dict
    """
    k_use = min(k, len(points) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k_use, metric='euclidean').fit(points)
    all_distances, all_indices = nbrs.kneighbors(points)
    
    pr_values = []
    pca_values = []
    entropy_values = []
    mle_values = []
    
    n_points = min(len(points), 300)
    
    for i in range(n_points):
        neighbor_indices = all_indices[i]
        neighbors = points[neighbor_indices]
        
        cov = np.cov(neighbors.T)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvals = np.sort(eigenvals)[::-1]
        eigenvals = np.abs(eigenvals)
        
        pr_values.append(participation_ratio(eigenvals))
        pca_values.append(pca_rank_threshold(eigenvals, epsilon=0.01))
        entropy_values.append(spectral_entropy_dimension(eigenvals))
        
        d_mle = mle_intrinsic_dimension(all_distances[i], k_use)
        if not np.isnan(d_mle):
            mle_values.append(d_mle)
    
    pr_mean = np.nanmean(pr_values) if len(pr_values) > 0 else np.nan
    pca_mean = np.nanmean(pca_values) if len(pca_values) > 0 else np.nan
    entropy_mean = np.nanmean(entropy_values) if len(entropy_values) > 0 else np.nan
    mle_mean = np.nanmean(mle_values) if len(mle_values) > 0 else np.nan
    
    n_corr = min(100, n_points)
    corr_indices = np.random.choice(n_points, n_corr, replace=False)
    sample_points = points[corr_indices]
    
    dist_matrix = np.zeros((n_corr, n_corr))
    for i in range(n_corr):
        dist_matrix[i] = np.linalg.norm(sample_points - sample_points[i], axis=1)
    
    corr_dim = correlation_dimension(dist_matrix)
    
    return {
        'PR': pr_mean,
        'MLE': mle_mean,
        'Correlation': corr_dim,
        'PCA_Threshold': pca_mean,
        'Entropy': entropy_mean
    }


def check_growth(D_values):
    """Check if curve shows growth."""
    if len(D_values) < 2:
        return False
    
    early = np.mean(D_values[:3])
    late = np.mean(D_values[-3:])
    
    return late > early * 1.05


def check_saturation(D_values):
    """Check if curve shows saturation (slope -> 0 at large k)."""
    if len(D_values) < 4:
        return False
    
    late_slope = D_values[-1] - D_values[-2]
    
    return abs(late_slope) < 0.5 * np.std(D_values)


def run_ablation():
    """Run the full ablation study."""
    print("=" * 70)
    print("ESTIMATOR ABLATION STUDY")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = load_datasets()
    
    all_results = {}
    estimator_names = ['PR', 'MLE', 'Correlation', 'PCA_Threshold', 'Entropy']
    
    print("\n[2/5] Computing estimators for all datasets and k values...")
    
    for dataset_name, points in datasets.items():
        print(f"\n  Processing {dataset_name}...")
        all_results[dataset_name] = {est: [] for est in estimator_names}
        
        for k in K_VALUES:
            print(f"    k={k}...", end=" ")
            
            try:
                est_results = compute_estimators_for_k(points, k)
                
                for est_name in estimator_names:
                    value = est_results.get(est_name, np.nan)
                    if np.isnan(value) or np.isinf(value):
                        value = np.nan
                    all_results[dataset_name][est_name].append(value)
                
                print("done")
                
            except Exception as e:
                print(f"error: {e}")
                for est_name in estimator_names:
                    all_results[dataset_name][est_name].append(np.nan)
    
    print("\n[3/5] Normalizing curves...")
    
    normalized_results = {}
    
    for dataset_name in all_results.keys():
        normalized_results[dataset_name] = {}
        
        for est_name in estimator_names:
            values = np.array(all_results[dataset_name][est_name])
            
            valid_mask = ~np.isnan(values) & (values > 0)
            if np.sum(valid_mask) == 0:
                normalized_results[dataset_name][est_name] = values
                continue
            
            max_val = np.max(values[valid_mask])
            
            if max_val > 0:
                norm_values = values / max_val
            else:
                norm_values = values
            
            normalized_results[dataset_name][est_name] = norm_values
    
    print("\n[4/5] Computing analysis metrics...")
    
    growth_flags = []
    saturation_flags = []
    
    for dataset_name in all_results.keys():
        for est_name in estimator_names:
            values = all_results[dataset_name][est_name]
            
            has_growth = check_growth(values)
            has_saturation = check_saturation(values)
            
            growth_flags.append({
                'dataset': dataset_name,
                'estimator': est_name,
                'growth_present': has_growth
            })
            
            saturation_flags.append({
                'dataset': dataset_name,
                'estimator': est_name,
                'saturation_present': has_saturation
            })
    
    alignment_results = []
    
    for est_name in estimator_names:
        curves_by_dataset = {}
        
        for dataset_name in normalized_results.keys():
            curves_by_dataset[dataset_name] = normalized_results[dataset_name][est_name]
        
        for sys1 in curves_by_dataset.keys():
            for sys2 in curves_by_dataset.keys():
                v1 = curves_by_dataset[sys1]
                v2 = curves_by_dataset[sys2]
                
                valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
                v1_valid = v1[valid_mask]
                v2_valid = v2[valid_mask]
                
                if len(v1_valid) >= 3 and np.std(v1_valid) > 0 and np.std(v2_valid) > 0:
                    r, _ = pearsonr(v1_valid, v2_valid)
                else:
                    r = np.nan
                
                alignment_results.append({
                    'estimator': est_name,
                    'system1': sys1,
                    'system2': sys2,
                    'alignment_r': r
                })
    
    print("\n[5/5] Saving results...")
    
    np.save(results_dir / "raw_curves.npy", all_results)
    np.save(results_dir / "normalized_curves.npy", normalized_results)
    
    growth_df = pd.DataFrame(growth_flags)
    growth_df.to_csv(results_dir / "growth_flags.csv", index=False)
    
    sat_df = pd.DataFrame(saturation_flags)
    sat_df.to_csv(results_dir / "saturation_flags.csv", index=False)
    
    align_df = pd.DataFrame(alignment_results)
    align_df.to_csv(results_dir / "alignment_matrix_per_estimator.csv", index=False)
    
    summary_data = []
    
    for est_name in estimator_names:
        est_alignments = align_df[align_df['estimator'] == est_name]
        off_diag = est_alignments[est_alignments['system1'] != est_alignments['system2']]
        
        mean_r = off_diag['alignment_r'].mean() if len(off_diag) > 0 else np.nan
        std_r = off_diag['alignment_r'].std() if len(off_diag) > 0 else np.nan
        
        est_growth = growth_df[growth_df['estimator'] == est_name]
        growth_count = est_growth['growth_present'].sum()
        
        summary_data.append({
            'estimator': est_name,
            'mean_alignment': mean_r,
            'std_alignment': std_r,
            'growth_count': growth_count,
            'total_tests': len(est_growth)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "estimator_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    
    print("\nGrowth presence by estimator:")
    growth_pivot = growth_df.pivot(index='dataset', columns='estimator', values='growth_present')
    print(growth_pivot.to_string())
    
    print("\nAlignment statistics by estimator:")
    print(summary_df.to_string(index=False))
    
    return all_results, normalized_results, summary_df, growth_df, sat_df, align_df


def generate_summary(all_results, normalized_results, summary_df, growth_df, sat_df, align_df):
    """Generate summary.txt file."""
    
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    
    lines = []
    lines.append("=" * 70)
    lines.append("ESTIMATOR ABLATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("GROWTH-PRESENCE ANALYSIS")
    lines.append("-" * 50)
    
    for est in growth_df['estimator'].unique():
        est_df = growth_df[growth_df['estimator'] == est]
        count = est_df['growth_present'].sum()
        total = len(est_df)
        lines.append(f"  {est}: {count}/{total} datasets show growth")
    
    lines.append("")
    lines.append("SATURATION-PRESENCE ANALYSIS")
    lines.append("-" * 50)
    
    for est in sat_df['estimator'].unique():
        est_df = sat_df[sat_df['estimator'] == est]
        count = est_df['saturation_present'].sum()
        total = len(est_df)
        lines.append(f"  {est}: {count}/{total} datasets show saturation")
    
    lines.append("")
    lines.append("ALIGNMENT ANALYSIS (Cross-system Pearson r)")
    lines.append("-" * 50)
    
    for est in align_df['estimator'].unique():
        est_df = align_df[align_df['estimator'] == est]
        off_diag = est_df[est_df['system1'] != est_df['system2']]
        mean_r = off_diag['alignment_r'].mean()
        std_r = off_diag['alignment_r'].std()
        lines.append(f"  {est}:")
        lines.append(f"    Mean alignment: {mean_r:.4f}")
        lines.append(f"    Std alignment: {std_r:.4f}")
    
    lines.append("")
    lines.append("COMPARISON WITH PARTICIPATION RATIO")
    lines.append("-" * 50)
    
    pr_alignments = align_df[align_df['estimator'] == 'PR']['alignment_r'].values
    other_estimators = ['MLE', 'Correlation', 'PCA_Threshold', 'Entropy']
    
    for est in other_estimators:
        est_df = align_df[align_df['estimator'] == est]
        est_alignments = est_df['alignment_r'].values
        if len(est_alignments) > 0 and len(pr_alignments) > 0:
            mean_diff = np.nanmean(est_alignments) - np.nanmean(pr_alignments)
            shape_desc = "similar" if abs(mean_diff) < 0.1 else "different"
            lines.append(f"  {est} vs PR:")
            lines.append(f"    Mean alignment difference: {mean_diff:.4f}")
            lines.append(f"    Shape comparison: {shape_desc}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    summary_text = "\n".join(lines)
    
    summary_path = Path(__file__).parent / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n[Summary saved to: {summary_path}]")
    
    return summary_text


if __name__ == "__main__":
    try:
        all_results, normalized_results, summary_df, growth_df, sat_df, align_df = run_ablation()
        summary_text = generate_summary(all_results, normalized_results, summary_df, growth_df, sat_df, align_df)
        print("\n[SUCCESS] Estimator ablation completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Ablation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
