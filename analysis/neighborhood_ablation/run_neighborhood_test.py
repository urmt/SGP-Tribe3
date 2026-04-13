#!/usr/bin/env python3
"""
Neighborhood Structure Ablation Pipeline

This script tests whether the growth-saturation dimensionality profile
depends on k-NN locality or arises independently of neighborhood structure.

Three neighborhood conditions:
1. STANDARD k-NN (control) - Euclidean distance based
2. RANDOM NEIGHBORHOOD - Randomly sample k points, destroy locality
3. GLOBAL SAMPLING - Random mini-batch covariance, no neighborhoods

Output:
- analysis/neighborhood_ablation/results/*.npy
- analysis/neighborhood_ablation/results/*.csv
- analysis/neighborhood_ablation/summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

K_VALUES = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
N_SAMPLES = 500
SEED = 42

np.random.seed(SEED)


def load_datasets():
    """Load all datasets."""
    print("\n[1/6] Loading datasets...")
    
    data_path = Path(__file__).parent.parent.parent / "reproducibility" / "raw_data.npz"
    raw_data = np.load(data_path)
    
    datasets = {}
    
    datasets['TRIBE'] = raw_data['sgp_nodes'][:N_SAMPLES]
    print(f"  TRIBE: {datasets['TRIBE'].shape}")
    
    datasets['Hierarchical'] = generate_hierarchical()
    print(f"  Hierarchical: {datasets['Hierarchical'].shape}")
    
    datasets['Correlated'] = generate_correlated()
    print(f"  Correlated: {datasets['Correlated'].shape}")
    
    datasets['Sparse'] = generate_sparse()
    print(f"  Sparse: {datasets['Sparse'].shape}")
    
    datasets['Manifold'] = generate_manifold()
    print(f"  Manifold: {datasets['Manifold'].shape}")
    
    return datasets


def generate_hierarchical():
    """Generate hierarchical Gaussian."""
    n_dims, n_samples, decay = 50, N_SAMPLES, 0.1
    eigenvalues = np.exp(-decay * np.arange(n_dims))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    return np.random.randn(n_samples, n_dims) @ Q.T @ L


def generate_correlated():
    """Generate correlated Gaussian."""
    n_dims, n_samples, decay = 50, N_SAMPLES, 0.05
    eigenvalues = np.exp(-decay * np.arange(n_dims))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(n_dims, n_dims))
    return np.random.randn(n_samples, n_dims) @ Q.T @ L


def generate_sparse():
    """Generate sparse data."""
    X = np.random.randn(N_SAMPLES, 50)
    mask = np.random.rand(N_SAMPLES, 50) < 0.9
    X[mask] = 0
    return X


def generate_manifold():
    """Generate curved manifold."""
    n_dims, n_manifold, n_samples, curv = 50, 5, N_SAMPLES, 0.1
    t = np.random.randn(n_samples, n_manifold)
    for i in range(n_manifold):
        t[:, i] = t[:, i] / (1 + curv * np.sum(t[:, :i]**2, axis=1))
    B = np.random.randn(n_manifold, n_dims)
    B = B / np.linalg.norm(B, axis=0)
    return t @ B + 0.01 * np.random.randn(n_samples, n_dims)


def participation_ratio(eigenvalues):
    """Participation ratio."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def pca_threshold_dimension(eigenvalues, epsilon=0.01):
    """PCA threshold dimension."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    total = np.sum(eigenvalues)
    if total <= 0:
        return 1.0
    proportions = eigenvalues / total
    return float(np.sum(proportions > epsilon))


def entropy_dimension(eigenvalues):
    """Spectral entropy dimension."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    total = np.sum(eigenvalues)
    if total <= 0:
        return 1.0
    p = eigenvalues / total
    p = p[p > 0]
    if len(p) == 0:
        return 1.0
    H = -np.sum(p * np.log(p))
    return np.exp(H)


def standard_knn_neighborhood(points, k):
    """Standard k-NN with Euclidean distance."""
    k_use = min(k, len(points) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_use, metric='euclidean').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    eigenvalues_list = []
    for i in range(min(len(points), 300)):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvalues_list.append(eigenvals)
    
    return eigenvalues_list


def random_neighborhood_neighborhood(points, k):
    """Random neighborhood - destroys locality."""
    k_use = min(k, len(points) - 1)
    n_samples = min(len(points), 300)
    
    eigenvalues_list = []
    for i in range(n_samples):
        indices = np.random.choice(len(points), k_use, replace=False)
        indices = indices[indices != i] if i in indices else indices[:k_use-1]
        if len(indices) < 2:
            indices = np.random.choice(len(points), max(2, k_use), replace=False)
        neighbors = points[indices]
        cov = np.cov(neighbors.T)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvalues_list.append(eigenvals)
    
    return eigenvalues_list


def global_sampling_neighborhood(points, k):
    """Global sampling - random mini-batch covariance."""
    k_use = min(k, len(points) - 1)
    n_batches = min(300, len(points))
    
    eigenvalues_list = []
    for _ in range(n_batches):
        indices = np.random.choice(len(points), k_use, replace=False)
        neighbors = points[indices]
        cov = np.cov(neighbors.T)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvalues_list.append(eigenvals)
    
    return eigenvalues_list


def compute_estimators(eigenvalues_list):
    """Compute all estimators from eigenvalues."""
    pr_vals, pca_vals, ent_vals = [], [], []
    
    for eigenvals in eigenvalues_list:
        eigenvals = np.sort(eigenvals)[::-1]
        pr_vals.append(participation_ratio(eigenvals))
        pca_vals.append(pca_threshold_dimension(eigenvals))
        ent_vals.append(entropy_dimension(eigenvals))
    
    return {
        'PR': np.nanmean(pr_vals) if len(pr_vals) > 0 else np.nan,
        'PCA_Threshold': np.nanmean(pca_vals) if len(pca_vals) > 0 else np.nan,
        'Entropy': np.nanmean(ent_vals) if len(ent_vals) > 0 else np.nan
    }


def check_growth(values):
    """Check if curve shows growth."""
    if len(values) < 2:
        return False
    early = np.mean(values[:3])
    late = np.mean(values[-3:])
    return late > early * 1.05


def check_saturation(values):
    """Check if curve shows saturation."""
    if len(values) < 4:
        return False
    late_slope = values[-1] - values[-2]
    return abs(late_slope) < 0.5 * np.std(values)


def run_neighborhood_ablation():
    """Run the neighborhood ablation study."""
    print("=" * 70)
    print("NEIGHBORHOOD STRUCTURE ABLATION")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = load_datasets()
    
    neighborhood_types = {
        'kNN': standard_knn_neighborhood,
        'Random': random_neighborhood_neighborhood,
        'Global': global_sampling_neighborhood
    }
    
    estimators = ['PR', 'PCA_Threshold', 'Entropy']
    
    all_results = {}
    
    print("\n[2/6] Computing neighborhood conditions...")
    
    for neigh_name, neigh_fn in neighborhood_types.items():
        print(f"\n  {neigh_name} neighborhood:")
        all_results[neigh_name] = {}
        
        for dataset_name, points in datasets.items():
            print(f"    {dataset_name}...", end=" ")
            all_results[neigh_name][dataset_name] = {est: [] for est in estimators}
            
            for k in K_VALUES:
                try:
                    eigenvalues_list = neigh_fn(points, k)
                    est_results = compute_estimators(eigenvalues_list)
                    
                    for est_name in estimators:
                        value = est_results.get(est_name, np.nan)
                        if np.isnan(value) or np.isinf(value):
                            value = np.nan
                        all_results[neigh_name][dataset_name][est_name].append(value)
                except Exception as e:
                    print(f"\n    [ERROR] {e}", end=" ")
                    for est_name in estimators:
                        all_results[neigh_name][dataset_name][est_name].append(np.nan)
            
            print("done")
    
    print("\n[3/6] Normalizing curves...")
    
    normalized_results = {}
    
    for neigh_name in all_results.keys():
        normalized_results[neigh_name] = {}
        
        for dataset_name in all_results[neigh_name].keys():
            normalized_results[neigh_name][dataset_name] = {}
            
            for est_name in estimators:
                values = np.array(all_results[neigh_name][dataset_name][est_name])
                valid_mask = ~np.isnan(values) & (values > 0)
                
                if np.sum(valid_mask) == 0:
                    normalized_results[neigh_name][dataset_name][est_name] = values
                    continue
                
                max_val = np.max(values[valid_mask])
                if max_val > 0:
                    values = values / max_val
                
                normalized_results[neigh_name][dataset_name][est_name] = values
    
    print("\n[4/6] Computing growth and saturation flags...")
    
    growth_flags = []
    saturation_flags = []
    
    for neigh_name in all_results.keys():
        for dataset_name in all_results[neigh_name].keys():
            for est_name in estimators:
                values = all_results[neigh_name][dataset_name][est_name]
                
                growth_flags.append({
                    'neighborhood': neigh_name,
                    'dataset': dataset_name,
                    'estimator': est_name,
                    'growth_present': check_growth(values)
                })
                
                saturation_flags.append({
                    'neighborhood': neigh_name,
                    'dataset': dataset_name,
                    'estimator': est_name,
                    'saturation_present': check_saturation(values)
                })
    
    print("\n[5/6] Computing alignment matrices...")
    
    alignment_results = []
    
    for neigh_name in normalized_results.keys():
        for est_name in estimators:
            for sys1 in normalized_results[neigh_name].keys():
                for sys2 in normalized_results[neigh_name].keys():
                    v1 = normalized_results[neigh_name][sys1][est_name]
                    v2 = normalized_results[neigh_name][sys2][est_name]
                    
                    valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
                    v1_v = v1[valid_mask]
                    v2_v = v2[valid_mask]
                    
                    if len(v1_v) >= 3 and np.std(v1_v) > 0 and np.std(v2_v) > 0:
                        r, _ = pearsonr(v1_v, v2_v)
                    else:
                        r = np.nan
                    
                    alignment_results.append({
                        'neighborhood': neigh_name,
                        'estimator': est_name,
                        'system1': sys1,
                        'system2': sys2,
                        'alignment_r': r
                    })
    
    print("\n[6/6] Saving results...")
    
    np.save(results_dir / "raw_curves.npy", all_results)
    np.save(results_dir / "normalized_curves.npy", normalized_results)
    
    growth_df = pd.DataFrame(growth_flags)
    growth_df.to_csv(results_dir / "growth_flags.csv", index=False)
    
    sat_df = pd.DataFrame(saturation_flags)
    sat_df.to_csv(results_dir / "saturation_flags.csv", index=False)
    
    align_df = pd.DataFrame(alignment_results)
    align_df.to_csv(results_dir / "alignment_matrix.csv", index=False)
    
    summary_data = []
    
    for neigh_name in neighborhood_types.keys():
        neigh_align = align_df[align_df['neighborhood'] == neigh_name]
        off_diag = neigh_align[neigh_align['system1'] != neigh_align['system2']]
        
        mean_align = off_diag['alignment_r'].mean() if len(off_diag) > 0 else np.nan
        std_align = off_diag['alignment_r'].std() if len(off_diag) > 0 else np.nan
        
        neigh_growth = growth_df[growth_df['neighborhood'] == neigh_name]
        growth_count = neigh_growth['growth_present'].sum()
        
        neigh_sat = sat_df[sat_df['neighborhood'] == neigh_name]
        sat_count = neigh_sat['saturation_present'].sum()
        
        summary_data.append({
            'neighborhood': neigh_name,
            'mean_alignment': mean_align,
            'std_alignment': std_align,
            'growth_count': growth_count,
            'saturation_count': sat_count,
            'total_tests': len(neigh_growth)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "neighborhood_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nGrowth presence by neighborhood:")
    growth_pivot = growth_df.pivot_table(
        index='neighborhood', columns='estimator', 
        values='growth_present', aggfunc='sum'
    )
    print(growth_pivot.to_string())
    
    print("\nAlignment by neighborhood:")
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
    lines.append("NEIGHBORHOOD STRUCTURE ABLATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("GROWTH-PRESENCE BY NEIGHBORHOOD")
    lines.append("-" * 50)
    
    for neigh in ['kNN', 'Random', 'Global']:
        neigh_df = growth_df[growth_df['neighborhood'] == neigh]
        count = neigh_df['growth_present'].sum()
        total = len(neigh_df)
        lines.append(f"  {neigh}: {count}/{total} show growth")
    
    lines.append("")
    lines.append("SATURATION-PRESENCE BY NEIGHBORHOOD")
    lines.append("-" * 50)
    
    for neigh in ['kNN', 'Random', 'Global']:
        neigh_df = sat_df[sat_df['neighborhood'] == neigh]
        count = neigh_df['saturation_present'].sum()
        total = len(neigh_df)
        lines.append(f"  {neigh}: {count}/{total} show saturation")
    
    lines.append("")
    lines.append("ALIGNMENT BY NEIGHBORHOOD (Cross-system Pearson r)")
    lines.append("-" * 50)
    
    for _, row in summary_df.iterrows():
        lines.append(f"  {row['neighborhood']}:")
        lines.append(f"    Mean alignment: {row['mean_alignment']:.4f}")
        lines.append(f"    Std alignment: {row['std_alignment']:.4f}")
    
    lines.append("")
    lines.append("COMPARISON: k-NN vs Random vs Global")
    lines.append("-" * 50)
    
    knn_align = summary_df[summary_df['neighborhood'] == 'kNN']['mean_alignment'].values[0]
    rand_align = summary_df[summary_df['neighborhood'] == 'Random']['mean_alignment'].values[0]
    glob_align = summary_df[summary_df['neighborhood'] == 'Global']['mean_alignment'].values[0]
    
    lines.append(f"  kNN: {knn_align:.4f}")
    lines.append(f"  Random: {rand_align:.4f}")
    lines.append(f"  Global: {glob_align:.4f}")
    lines.append("")
    lines.append(f"  kNN vs Random difference: {knn_align - rand_align:.4f}")
    lines.append(f"  kNN vs Global difference: {knn_align - glob_align:.4f}")
    lines.append(f"  Random vs Global difference: {rand_align - glob_align:.4f}")
    
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
        all_results, normalized_results, summary_df, growth_df, sat_df, align_df = run_neighborhood_ablation()
        summary_text = generate_summary(
            all_results, normalized_results, summary_df, growth_df, sat_df, align_df
        )
        print("\n[SUCCESS] Neighborhood ablation completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Ablation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
