#!/usr/bin/env python3
"""
Experiment Set 1: Expanded Null Models

This script tests which aspects of the growth-saturation profile are generic
vs. system-specific by comparing against various null models.

Null Models Tested:
1. Isotropic Gaussian (d=9, d=50)
2. Spectrum-Matched Gaussian
3. Eigenvalue-Shuffled Model
4. Neighborhood-Shuffled Model
5. Distance-Preserving Shuffle

Output:
- analysis/results/null_models/deff_curves.npy
- analysis/results/null_models/beta_curves.npy
- analysis/results/null_models/alignment_matrix.csv
- analysis/results/null_models/null_model_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.insert(0, str(Path(__file__).parent))
from common_functions import (
    N_SAMPLES, K_VALUES, set_seed,
    compute_local_covariance, compute_D_eff_curve,
    compute_beta, normalize_curve, compute_alignment,
    generate_isotropic_gaussian, generate_spectrum_matched_gaussian,
    generate_correlated_gaussian, generate_sparse_data, generate_curved_manifold,
    save_results, check_validity
)

def load_tribe_data():
    """Load TRIBE data from reproducibility bundle."""
    data_path = Path(__file__).parent.parent / "reproducibility" / "raw_data.npz"
    data = np.load(data_path)
    return {
        'tribe': data['sgp_nodes'],
        'streams': data['streams'],
        'edge_weights': data['edge_weights']
    }

def compute_deff_for_system(points, system_name, n_samples=N_SAMPLES):
    """Compute D_eff curve for a system."""
    if points is None:
        return None
    
    k_actual = [min(k, len(points)-1) for k in K_VALUES]
    k_to_use = min(100, len(points)-1)
    
    D_eff = compute_local_covariance(points[:n_samples], k_to_use)
    D_eff_dict = {}
    
    for k in K_VALUES:
        k_use = min(k, len(points) - 1, 100)
        D_eff_k = compute_local_covariance(points[:n_samples], k_use)
        D_eff_dict[k] = np.mean(D_eff_k)
    
    is_valid, issues = check_validity(D_eff_dict)
    if not is_valid:
        print(f"  [WARNING] {system_name}: {issues}")
    
    return D_eff_dict

def generate_eigenvalue_shuffled(points, n_samples=N_SAMPLES, seed=42):
    """Generate data with shuffled eigenvalues."""
    set_seed(seed)
    
    points_sub = points[:n_samples]
    n_dims = points_sub.shape[1]
    
    cov = np.cov(points_sub.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    perm = np.random.permutation(len(eigenvalues))
    eigenvalues_shuffled = eigenvalues[perm]
    
    L_shuffled = np.diag(np.sqrt(np.abs(eigenvalues_shuffled)))
    Q = eigenvectors
    
    X = np.random.randn(n_samples, n_dims)
    data = X @ Q.T @ L_shuffled
    
    return data

def generate_neighborhood_shuffled(points, k=20, n_samples=N_SAMPLES, seed=42):
    """Generate data with shuffled neighborhood relationships."""
    set_seed(seed)
    
    points_sub = points[:n_samples]
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points_sub)
    distances, indices = nbrs.kneighbors(points_sub)
    
    shuffled_data = np.zeros_like(points_sub)
    for i in range(n_samples):
        neighbor_indices = indices[i]
        random_neighbor = np.random.choice(neighbor_indices)
        shuffled_data[i] = points_sub[random_neighbor]
    
    return shuffled_data

def generate_distance_preserving_shuffle(points, n_samples=N_SAMPLES, seed=42):
    """Generate data preserving pairwise distance distribution."""
    set_seed(seed)
    
    points_sub = points[:n_samples]
    n_dims = points_sub.shape[1]
    n_samples_actual = min(n_samples, len(points_sub))
    
    random_dirs = np.random.randn(n_samples_actual, n_dims)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    base_point = np.zeros(n_dims)
    distances = np.random.rand(n_samples_actual) * 5 + 0.1
    
    data = base_point + distances[:, np.newaxis] * random_dirs
    
    return data

def run_null_models():
    """Run all null model experiments."""
    print("=" * 60)
    print("Experiment Set 1: Expanded Null Models")
    print("=" * 60)
    
    results_dir = Path(__file__).parent / "results" / "null_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(42)
    
    print("\n[1/5] Loading TRIBE data...")
    data = load_tribe_data()
    tribe_data = data['tribe']
    print(f"  Loaded TRIBE data: {tribe_data.shape}")
    
    print("\n[2/5] Computing baseline curves (TRIBE + synthetic)...")
    baseline_systems = {}
    
    baseline_systems['TRIBE'] = compute_deff_for_system(tribe_data, 'TRIBE')
    baseline_systems['Hierarchical'] = compute_deff_for_system(
        generate_correlated_gaussian(n_dims=50, decay_rate=0.1), 'Hierarchical'
    )
    baseline_systems['Correlated'] = compute_deff_for_system(
        generate_correlated_gaussian(n_dims=50, decay_rate=0.05), 'Correlated'
    )
    baseline_systems['Sparse'] = compute_deff_for_system(
        generate_sparse_data(n_dims=50, sparsity=0.9), 'Sparse'
    )
    baseline_systems['Manifold'] = compute_deff_for_system(
        generate_curved_manifold(n_dims=50), 'Manifold'
    )
    
    baseline_systems = {k: v for k, v in baseline_systems.items() if v is not None}
    print(f"  Computed baseline for {len(baseline_systems)} systems")
    
    print("\n[3/5] Generating and computing null models...")
    null_models = {}
    
    print("  Isotropic Gaussian (d=9)...")
    null_models['IsoGauss_d9'] = compute_deff_for_system(
        generate_isotropic_gaussian(n_dims=9), 'IsoGauss_d9'
    )
    
    print("  Isotropic Gaussian (d=50)...")
    null_models['IsoGauss_d50'] = compute_deff_for_system(
        generate_isotropic_gaussian(n_dims=50), 'IsoGauss_d50'
    )
    
    print("  Spectrum-Matched Gaussian...")
    cov_tribe = np.cov(tribe_data[:2000].T)
    eigenvalues_tribe = np.sort(np.linalg.eigvalsh(cov_tribe))[::-1]
    eigenvalues_tribe = np.abs(eigenvalues_tribe)
    spectrum_data = generate_spectrum_matched_gaussian(eigenvalues_tribe)
    null_models['SpectrumMatch'] = compute_deff_for_system(spectrum_data, 'SpectrumMatch')
    
    print("  Eigenvalue-Shuffled...")
    eigen_shuffled = generate_eigenvalue_shuffled(tribe_data)
    null_models['EigenShuffle'] = compute_deff_for_system(eigen_shuffled, 'EigenShuffle')
    
    print("  Neighborhood-Shuffled...")
    neigh_shuffled = generate_neighborhood_shuffled(tribe_data)
    null_models['NeighShuffle'] = compute_deff_for_system(neigh_shuffled, 'NeighShuffle')
    
    print("  Distance-Preserving Shuffle...")
    dist_shuffled = generate_distance_preserving_shuffle(tribe_data)
    null_models['DistPreserve'] = compute_deff_for_system(dist_shuffled, 'DistPreserve')
    
    null_models = {k: v for k, v in null_models.items() if v is not None}
    print(f"  Computed {len(null_models)} null models")
    
    print("\n[4/5] Computing normalized curves and alignments...")
    all_systems = {**baseline_systems, **null_models}
    
    normalized_curves = {}
    for name, D_eff_dict in all_systems.items():
        norm_dict, D_max = normalize_curve(D_eff_dict)
        normalized_curves[name] = norm_dict
    
    alignment_matrix = pd.DataFrame(index=all_systems.keys(), columns=all_systems.keys())
    for sys1 in all_systems.keys():
        for sys2 in all_systems.keys():
            r = compute_alignment(normalized_curves[sys1], normalized_curves[sys2])
            alignment_matrix.loc[sys1, sys2] = r
    
    print("\n[5/5] Saving results...")
    save_results(normalized_curves, results_dir / "deff_curves.npy")
    save_results(alignment_matrix.to_dict(), results_dir / "alignment_matrix.csv")
    
    summary_data = []
    for name, D_eff_dict in null_models.items():
        k_vals = sorted(D_eff_dict.keys())
        D_vals = [D_eff_dict[k] for k in k_vals]
        
        mean_deff = np.mean(D_vals)
        D_max = np.max(D_vals)
        D_final = D_eff_dict.get(500, D_vals[-1]) if 500 in D_eff_dict else D_vals[-1]
        sat_ratio = D_final / D_max if D_max > 0 else 1.0
        
        align_to_tribe = compute_alignment(normalized_curves.get('TRIBE', {}), normalized_curves.get(name, {}))
        align_to_synth = []
        for sys_name in ['Hierarchical', 'Correlated', 'Sparse', 'Manifold']:
            if sys_name in normalized_curves and name in normalized_curves:
                r = compute_alignment(normalized_curves[sys_name], normalized_curves[name])
                if not np.isnan(r):
                    align_to_synth.append(r)
        mean_align_synth = np.mean(align_to_synth) if align_to_synth else np.nan
        
        summary_data.append({
            'model': name,
            'mean_D_eff': mean_deff,
            'D_max': D_max,
            'saturation_ratio': sat_ratio,
            'align_to_TRIBE': align_to_tribe,
            'mean_align_to_synthetic': mean_align_synth
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "null_model_summary.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Experiment Set 1 Complete")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nNull Model Summary:")
    print(summary_df.to_string(index=False))
    
    return normalized_curves, alignment_matrix, summary_df

if __name__ == "__main__":
    try:
        curves, alignment, summary = run_null_models()
        print("\n[SUCCESS] All null model experiments completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
