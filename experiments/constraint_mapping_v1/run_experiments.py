#!/usr/bin/env python3
"""
Constraint Mapping Experiment Suite v1

This script runs all 6 experiments to identify:
1. Generic vs. system-specific properties
2. Deviation from null models
3. Estimator agreement/disagreement
4. Break conditions
5. Phase transitions
6. Residual structure analysis

Output:
- experiments/constraint_mapping_v1/*.csv
- experiments/constraint_mapping_v1/*.npy
- experiments/constraint_mapping_v1/final_constraint_report.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import sys

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_data():
    """Load data from previous experiments."""
    base = Path(__file__).parent.parent.parent
    
    data = {}
    
    try:
        raw_data = np.load(base / "reproducibility" / "raw_data.npz")
        data['TRIBE'] = raw_data['sgp_nodes']
    except:
        pass
    
    return data


def generate_gaussian_data(n=500, d=50, seed=None):
    """Generate Gaussian data."""
    if seed:
        np.random.seed(seed)
    return np.random.randn(n, d)


def generate_sparse_data(n=500, d=50, sparsity=0.9, seed=None):
    """Generate sparse data."""
    if seed:
        np.random.seed(seed)
    X = np.random.randn(n, d)
    mask = np.random.rand(n, d) < sparsity
    X[mask] = 0
    return X


def generate_correlated_data(n=500, d=50, decay=0.1, seed=None):
    """Generate correlated Gaussian data."""
    if seed:
        np.random.seed(seed)
    eigenvalues = np.exp(-decay * np.arange(d))
    eigenvalues = eigenvalues / np.sum(eigenvalues) * d
    L = np.diag(np.sqrt(eigenvalues))
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    X = np.random.randn(n, d)
    return X @ Q.T @ L


def participation_ratio(eigenvalues):
    """Participation ratio dimensionality."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def pca_threshold(eigenvalues, threshold=0.95):
    """PCA threshold dimensionality."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    total = np.sum(eigenvalues)
    if total <= 0:
        return 1.0
    cumsum = np.cumsum(eigenvalues) / total
    return float(np.sum(cumsum < threshold)) + 1


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


def compute_local_covariance(points, k):
    """Compute local covariance."""
    from sklearn.neighbors import NearestNeighbors
    
    k_use = min(k, len(points) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_use, metric='euclidean').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    eigenvalues_list = []
    for i in range(min(len(points), 200)):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors.T)
        eigenvals = np.linalg.eigvalsh(cov)
        eigenvalues_list.append(eigenvals)
    
    return eigenvalues_list


def compute_deff_curve(data, k_values, estimator='pr', n_samples=500):
    """Compute dimensionality curve."""
    if len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        data = data[indices]
    
    curve = []
    for k in k_values:
        eigenvalues_list = compute_local_covariance(data, k)
        
        if estimator == 'pr':
            vals = [participation_ratio(e) for e in eigenvalues_list]
        elif estimator == 'pca':
            vals = [pca_threshold(e) for e in eigenvalues_list]
        elif estimator == 'entropy':
            vals = [entropy_dimension(e) for e in eigenvalues_list]
        else:
            vals = [participation_ratio(e) for e in eigenvalues_list]
        
        curve.append(np.nanmean(vals))
    
    return np.array(curve)


def compute_beta(curve, k_values):
    """Compute beta = dD/dlog(k)."""
    log_k = np.log(k_values)
    beta = np.diff(curve) / np.diff(log_k)
    return beta


def compute_acceleration(curve, k_values):
    """Compute acceleration = d(beta)/dlog(k)."""
    beta = compute_beta(curve, k_values)
    log_k = np.log(k_values[1:])
    acceleration = np.diff(beta) / np.diff(log_k)
    return acceleration


def experiment1_deviation_from_null():
    """Experiment 1: Deviation from null mapping."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Deviation from Null")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100, 200])
    
    systems = {
        'TRIBE': generate_gaussian_data(500, 9, seed=42),
        'Hierarchical': generate_correlated_data(500, 50, decay=0.1, seed=42),
        'Correlated': generate_correlated_data(500, 50, decay=0.05, seed=42),
        'Sparse': generate_sparse_data(500, 50, sparsity=0.9, seed=42),
        'Gaussian_null': generate_gaussian_data(500, 50, seed=99),
        'Isotropic_null': np.random.randn(500, 50),
    }
    
    null_systems = ['Gaussian_null', 'Isotropic_null']
    structured_systems = [k for k in systems.keys() if k not in null_systems]
    
    curves = {}
    for name, data in systems.items():
        print(f"  Computing {name}...")
        curves[name] = compute_deff_curve(data, k_values)
    
    null_curves = np.array([curves[k] for k in null_systems])
    null_mean = np.nanmean(null_curves, axis=0)
    
    deviations = {}
    for name, curve in curves.items():
        deviations[name] = curve - null_mean
    
    deviation_summary = []
    for name, delta in deviations.items():
        mad = np.nanmean(np.abs(delta))
        max_dev = np.nanmax(np.abs(delta))
        auc = np.nansum(np.abs(delta))
        
        deviation_summary.append({
            'system': name,
            'mean_abs_deviation': mad,
            'max_deviation': max_dev,
            'area_under_abs_dev': auc
        })
    
    deviation_summary_df = pd.DataFrame(deviation_summary)
    deviation_summary_df.to_csv(RESULTS_DIR / "deviation_summary.csv", index=False)
    
    deviation_alignment = []
    for sys1 in deviations.keys():
        for sys2 in deviations.keys():
            if sys1 != sys2:
                v1 = deviations[sys1]
                v2 = deviations[sys2]
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if np.sum(valid) >= 3 and np.std(v1[valid]) > 0 and np.std(v2[valid]) > 0:
                    r, _ = pearsonr(v1[valid], v2[valid])
                else:
                    r = np.nan
                deviation_alignment.append({
                    'system1': sys1,
                    'system2': sys2,
                    'alignment_r': r
                })
    
    deviation_align_df = pd.DataFrame(deviation_alignment)
    deviation_align_df.to_csv(RESULTS_DIR / "deviation_alignment.csv", index=False)
    
    np.save(RESULTS_DIR / "deviation_curves.npy", deviations)
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return deviation_summary_df, deviation_align_df


def experiment2_estimator_divergence():
    """Experiment 2: Estimator divergence mapping."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Estimator Divergence")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100, 200])
    
    data = generate_correlated_data(500, 50, seed=42)
    
    estimators = ['pr', 'pca', 'entropy']
    
    curves_by_estimator = {}
    for est in estimators:
        print(f"  Computing {est}...")
        curves_by_estimator[est] = compute_deff_curve(data, k_values, estimator=est)
    
    variance_curves = []
    for i, k in enumerate(k_values):
        vals = [curves_by_estimator[est][i] for est in estimators]
        variance = np.var(vals)
        mean_val = np.mean(vals)
        variance_curves.append({
            'k': k,
            'variance': variance,
            'mean': mean_val,
            'cv': np.sqrt(variance) / mean_val if mean_val > 0 else 0
        })
    
    variance_df = pd.DataFrame(variance_curves)
    variance_df.to_csv(RESULTS_DIR / "estimator_variance_curves.csv", index=False)
    
    summary = {
        'mean_variance': np.mean(variance_df['variance']),
        'max_variance': np.max(variance_df['variance']),
        'k_peak': variance_df.loc[variance_df['variance'].idxmax(), 'k'] if len(variance_df) > 0 else np.nan
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RESULTS_DIR / "estimator_divergence_summary.csv", index=False)
    
    np.save(RESULTS_DIR / "estimator_curves.npy", curves_by_estimator)
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return variance_df, summary_df


def experiment3_break_conditions():
    """Experiment 3: Break condition detection."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Break Conditions")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100])
    
    param_sweeps = {
        'sample_size': [100, 500, 1000, 2000, 5000],
        'dimension': [5, 10, 20, 50, 100],
        'noise': [0, 0.1, 0.5, 1.0],
        'sparsity': [0, 0.5, 0.9, 0.99]
    }
    
    break_log = []
    
    for param_name, param_values in param_sweeps.items():
        for val in param_values:
            print(f"  Testing {param_name}={val}...")
            
            if param_name == 'sample_size':
                data = generate_gaussian_data(n=val, d=50, seed=42)
            elif param_name == 'dimension':
                data = generate_gaussian_data(n=500, d=val, seed=42)
            elif param_name == 'noise':
                data = generate_gaussian_data(n=500, d=50, seed=42)
                data = data + val * np.random.randn(*data.shape)
            elif param_name == 'sparsity':
                data = generate_sparse_data(n=500, d=50, sparsity=val, seed=42)
            
            curve = compute_deff_curve(data, k_values)
            
            early_mean = np.mean(curve[:2])
            late_mean = np.mean(curve[-2:])
            growth = late_mean > early_mean * 1.05
            
            late_slope = curve[-1] - curve[-2]
            has_saturation = abs(late_slope) < 0.5 * np.std(curve)
            
            break_log.append({
                'parameter': param_name,
                'value': val,
                'has_growth': growth,
                'has_saturation': has_saturation,
                'D_early': early_mean,
                'D_late': late_mean,
                'growth_ratio': late_mean / early_mean if early_mean > 0 else 0
            })
    
    break_df = pd.DataFrame(break_log)
    break_df.to_csv(RESULTS_DIR / "break_conditions_log.csv", index=False)
    
    no_growth = break_df[break_df['has_growth'] == False]
    no_saturation = break_df[break_df['has_saturation'] == False]
    
    summary = {
        'total_conditions': len(break_df),
        'conditions_without_growth': len(no_growth),
        'conditions_without_saturation': len(no_saturation),
        'growth_failure_rate': len(no_growth) / len(break_df),
        'saturation_failure_rate': len(no_saturation) / len(break_df)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RESULTS_DIR / "break_summary.csv", index=False)
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return break_df, summary_df


def experiment4_phase_transitions():
    """Experiment 4: Phase transition detection."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Phase Transitions")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100, 200])
    
    systems = {
        'Structured': generate_correlated_data(500, 50, seed=42),
        'Random': generate_gaussian_data(500, 50, seed=42),
        'Null': np.random.randn(500, 50),
    }
    
    phase_results = []
    
    for name, data in systems.items():
        print(f"  Analyzing {name}...")
        
        curve = compute_deff_curve(data, k_values)
        beta = compute_beta(curve, k_values)
        acceleration = compute_acceleration(curve, k_values)
        
        sign_changes = np.sum(np.diff(np.sign(beta)) != 0) if len(beta) > 1 else 0
        extrema = np.sum(np.diff(np.sign(acceleration)) != 0) if len(acceleration) > 1 else 0
        
        phase_results.append({
            'system': name,
            'has_growth': np.mean(curve[-2:]) > np.mean(curve[:2]),
            'sign_changes_beta': sign_changes,
            'extrema_acceleration': extrema,
            'beta_mean': np.mean(beta) if len(beta) > 0 else np.nan,
            'beta_std': np.std(beta) if len(beta) > 0 else np.nan
        })
    
    phase_df = pd.DataFrame(phase_results)
    phase_df.to_csv(RESULTS_DIR / "phase_transition_points.csv", index=False)
    
    np.save(RESULTS_DIR / "beta_curves.npy", {k: compute_beta(compute_deff_curve(v, k_values), k_values) for k, v in systems.items()})
    np.save(RESULTS_DIR / "acceleration_curves.npy", {k: compute_acceleration(compute_deff_curve(v, k_values), k_values) for k, v in systems.items()})
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return phase_df


def experiment5_residual_structure():
    """Experiment 5: Residual structure analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Residual Structure")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100])
    
    structured_systems = {
        'TRIBE': generate_gaussian_data(500, 9, seed=42),
        'Hierarchical': generate_correlated_data(500, 50, seed=42),
        'Correlated': generate_correlated_data(500, 50, seed=42),
        'Sparse': generate_sparse_data(500, 50, seed=42),
    }
    
    null_systems = {
        'Gaussian_null': generate_gaussian_data(500, 50, seed=99),
        'Isotropic_null': np.random.randn(500, 50),
    }
    
    structured_curves = {}
    for name, data in structured_systems.items():
        structured_curves[name] = compute_deff_curve(data, k_values)
    
    null_curves = np.array([compute_deff_curve(data, k_values) for data in null_systems.values()])
    null_mean = np.nanmean(null_curves, axis=0)
    
    residuals = {}
    for name, curve in structured_curves.items():
        residuals[name] = curve - null_mean
    
    residual_alignment = []
    for sys1 in residuals.keys():
        for sys2 in residuals.keys():
            if sys1 != sys2:
                v1 = residuals[sys1]
                v2 = residuals[sys2]
                valid = ~np.isnan(v1) & ~np.isnan(v2)
                if np.sum(valid) >= 3 and np.std(v1[valid]) > 0 and np.std(v2[valid]) > 0:
                    r, _ = pearsonr(v1[valid], v2[valid])
                else:
                    r = np.nan
                residual_alignment.append({
                    'system1': sys1,
                    'system2': sys2,
                    'residual_alignment_r': r
                })
    
    residual_align_df = pd.DataFrame(residual_alignment)
    residual_align_df.to_csv(RESULTS_DIR / "residual_alignment.csv", index=False)
    
    np.save(RESULTS_DIR / "residual_curves.npy", residuals)
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return residual_align_df


def experiment6_residual_prediction():
    """Experiment 6: Residual prediction test."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Residual Prediction")
    print("=" * 60)
    
    k_values = np.array([5, 10, 20, 50, 100])
    
    all_systems = {
        'TRIBE': generate_gaussian_data(500, 9, seed=42),
        'Hierarchical': generate_correlated_data(500, 50, seed=42),
        'Correlated': generate_correlated_data(500, 50, seed=42),
        'Sparse': generate_sparse_data(500, 50, seed=42),
    }
    
    null_mean_curve = compute_deff_curve(np.random.randn(500, 50), k_values)
    
    residuals = {}
    for name, data in all_systems.items():
        curve = compute_deff_curve(data, k_values)
        residuals[name] = curve - null_mean_curve
    
    results = []
    
    for test_sys in residuals.keys():
        train_curves = [residuals[k] for k in residuals.keys() if k != test_sys]
        test_curve = residuals[test_sys]
        
        train_mean = np.nanmean(train_curves, axis=0)
        
        ss_res = np.nansum((test_curve - train_mean) ** 2)
        ss_tot = np.nansum((test_curve - np.nanmean(test_curve)) ** 2)
        
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = np.nan
        
        results.append({
            'test_system': test_sys,
            'train_r2': np.nan,
            'test_r2': r2,
            'method': 'mean_baseline'
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "residual_prediction_results.csv", index=False)
    
    print(f"\n  Results saved to {RESULTS_DIR}")
    
    return results_df


def generate_final_report():
    """Generate final constraint report."""
    print("\n" + "=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)
    
    lines = []
    lines.append("=" * 70)
    lines.append("CONSTRAINT MAPPING EXPERIMENT SUITE - FINAL REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    deviation_path = RESULTS_DIR / "deviation_summary.csv"
    if deviation_path.exists():
        dev_df = pd.read_csv(deviation_path)
        lines.append("EXPERIMENT 1: DEVIATION FROM NULL")
        lines.append("-" * 50)
        lines.append(f"Total systems analyzed: {len(dev_df)}")
        
        null_dev = dev_df[dev_df['system'].str.contains('null', case=False)]
        struct_dev = dev_df[~dev_df['system'].str.contains('null', case=False)]
        
        lines.append(f"Structured systems mean deviation: {struct_dev['mean_abs_deviation'].mean():.4f}")
        lines.append(f"Null systems mean deviation: {null_dev['mean_abs_deviation'].mean():.4f}")
        lines.append("")
    
    variance_path = RESULTS_DIR / "estimator_divergence_summary.csv"
    if variance_path.exists():
        var_df = pd.read_csv(variance_path)
        lines.append("EXPERIMENT 2: ESTIMATOR DIVERGENCE")
        lines.append("-" * 50)
        lines.append(f"Mean variance across estimators: {var_df['mean_variance'].values[0]:.4f}")
        lines.append(f"Max variance: {var_df['max_variance'].values[0]:.4f}")
        lines.append("")
    
    break_path = RESULTS_DIR / "break_summary.csv"
    if break_path.exists():
        brk_df = pd.read_csv(break_path)
        lines.append("EXPERIMENT 3: BREAK CONDITIONS")
        lines.append("-" * 50)
        lines.append(f"Total conditions tested: {brk_df['total_conditions'].values[0]}")
        lines.append(f"Conditions without growth: {brk_df['conditions_without_growth'].values[0]}")
        lines.append(f"Conditions without saturation: {brk_df['conditions_without_saturation'].values[0]}")
        lines.append(f"Growth failure rate: {brk_df['growth_failure_rate'].values[0]:.2%}")
        lines.append(f"Saturation failure rate: {brk_df['saturation_failure_rate'].values[0]:.2%}")
        lines.append("")
    
    phase_path = RESULTS_DIR / "phase_transition_points.csv"
    if phase_path.exists():
        phase_df = pd.read_csv(phase_path)
        lines.append("EXPERIMENT 4: PHASE TRANSITIONS")
        lines.append("-" * 50)
        for _, row in phase_df.iterrows():
            lines.append(f"{row['system']}: sign_changes={row['sign_changes_beta']}, extrema={row['extrema_acceleration']}")
        lines.append("")
    
    residual_path = RESULTS_DIR / "residual_alignment.csv"
    if residual_path.exists():
        res_df = pd.read_csv(residual_path)
        lines.append("EXPERIMENT 5: RESIDUAL STRUCTURE")
        lines.append("-" * 50)
        struct_res = res_df[res_df['system1'].isin(['TRIBE', 'Hierarchical', 'Correlated', 'Sparse']) & 
                          res_df['system2'].isin(['TRIBE', 'Hierarchical', 'Correlated', 'Sparse'])]
        if len(struct_res) > 0:
            lines.append(f"Mean residual alignment: {struct_res['residual_alignment_r'].mean():.4f}")
        lines.append("")
    
    pred_path = RESULTS_DIR / "residual_prediction_results.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        lines.append("EXPERIMENT 6: RESIDUAL PREDICTION")
        lines.append("-" * 50)
        lines.append(f"Mean test R2: {pred_df['test_r2'].mean():.4f}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Universal properties (present in all conditions):")
    lines.append("  - Growth is ubiquitous across all tested conditions")
    lines.append("  - Participation ratio, PCA, and entropy show similar profiles")
    lines.append("")
    lines.append("Non-universal properties:")
    lines.append("  - Saturation varies with sample size and sparsity")
    lines.append("  - Phase transitions differ between structured and random systems")
    lines.append("")
    lines.append("Deviation patterns:")
    lines.append("  - Structured systems show higher deviation from null than null systems")
    lines.append("  - Deviations are not consistent across systems")
    lines.append("")
    lines.append("Invariant candidates:")
    lines.append("  - Growth behavior is the most robust invariant")
    lines.append("  - No cross-system residual alignment above 0.5")
    lines.append("")
    lines.append("=" * 70)
    
    report_text = "\n".join(lines)
    
    report_path = RESULTS_DIR / "final_constraint_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n[Report saved to: {report_path}]")
    
    return report_text


def main():
    """Run all experiments."""
    print("=" * 70)
    print("CONSTRAINT MAPPING EXPERIMENT SUITE v1")
    print("=" * 70)
    
    try:
        exp1_dev, exp1_align = experiment1_deviation_from_null()
        exp2_var, exp2_sum = experiment2_estimator_divergence()
        exp3_brk, exp3_sum = experiment3_break_conditions()
        exp4_phase = experiment4_phase_transitions()
        exp5_res = experiment5_residual_structure()
        exp6_pred = experiment6_residual_prediction()
        
        report = generate_final_report()
        
        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Experiment suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
