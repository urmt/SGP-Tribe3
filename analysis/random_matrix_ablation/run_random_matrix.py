#!/usr/bin/env python3
"""
Random Matrix Ablation Study

Test whether multiscale dimensionality growth-saturation emerges from pure
random matrix structure (Wishart behavior), independent of system structure.

Output:
- analysis/random_matrix_ablation/results/*.npy
- analysis/random_matrix_ablation/results/*.csv
- analysis/random_matrix_ablation/final_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import sys

SEED = 42
np.random.seed(SEED)

K_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

DIMENSIONS = [10, 50, 100, 500]
SAMPLE_SIZES = [500, 2000, 10000]
N_RUNS = 10


def participation_ratio(eigenvalues):
    """Participation ratio dimensionality."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def spectral_entropy(eigenvalues):
    """Spectral entropy."""
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


def pca_threshold_95(eigenvalues):
    """PCA threshold: number of components explaining 95% variance."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    total = np.sum(eigenvalues)
    if total <= 0:
        return 1.0
    cumsum = np.cumsum(eigenvalues) / total
    return float(np.sum(cumsum < 0.95)) + 1


def compute_estimators(eigenvalues):
    """Compute all estimators from eigenvalues."""
    eigenvalues = np.sort(eigenvalues)[::-1]
    return {
        'PR': participation_ratio(eigenvalues),
        'Entropy': spectral_entropy(eigenvalues),
        'PCA_95': pca_threshold_95(eigenvalues)
    }


def random_sample_covariance(data, k, seed=None):
    """Sample k points uniformly at random and compute covariance."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    k_use = min(k, n)
    
    indices = np.random.choice(n, k_use, replace=False)
    sample = data[indices]
    
    cov = np.cov(sample.T)
    
    if cov.shape[0] != cov.shape[1]:
        return np.array([1.0])
    
    eigenvalues = np.linalg.eigvalsh(cov)
    return eigenvalues


def generate_random_matrix(d, N, seed=None):
    """Generate X ~ Normal(0, I_d)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(N, d)


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


def run_experiment():
    """Run the full random matrix ablation experiment."""
    print("=" * 70)
    print("RANDOM MATRIX ABLATION STUDY")
    print("=" * 70)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    conditions_completed = 0
    
    print(f"\nRunning {len(DIMENSIONS)} dimensions x {len(SAMPLE_SIZES)} sample sizes x {N_RUNS} runs")
    
    for d in DIMENSIONS:
        for N in SAMPLE_SIZES:
            k_vals = [k for k in K_VALUES if k <= N / 2]
            
            if len(k_vals) == 0:
                print(f"\nSkipping d={d}, N={N} (no valid k values)")
                continue
            
            all_results[(d, N)] = {
                'runs': [],
                'aggregated': {}
            }
            
            print(f"\nProcessing d={d}, N={N}, k_range=[{k_vals[0]}, {k_vals[-1]}]")
            
            run_curves = {est: [] for est in ['PR', 'Entropy', 'PCA_95']}
            
            for run in range(N_RUNS):
                seed = SEED + d * 10000 + N * 100 + run
                
                data = generate_random_matrix(d, N, seed)
                
                for k in k_vals:
                    eigenvalues = random_sample_covariance(data, k, seed)
                    est_results = compute_estimators(eigenvalues)
                    
                    for est_name in est_results.keys():
                        run_curves[est_name].append({
                            'k': k,
                            'value': est_results[est_name]
                        })
                
                conditions_completed += 1
            
            for est_name in ['PR', 'Entropy', 'PCA_95']:
                curves_by_k = {}
                for entry in run_curves[est_name]:
                    k = entry['k']
                    if k not in curves_by_k:
                        curves_by_k[k] = []
                    curves_by_k[k].append(entry['value'])
                
                mean_curve = {k: np.mean(vals) for k, vals in curves_by_k.items()}
                std_curve = {k: np.std(vals) for k, vals in curves_by_k.items()}
                
                all_results[(d, N)]['aggregated'][est_name] = {
                    'mean': mean_curve,
                    'std': std_curve
                }
                all_results[(d, N)]['runs'].append(run_curves[est_name])
            
            print(f"  Completed {N_RUNS} runs")
    
    print(f"\n\nTotal conditions completed: {conditions_completed}")
    
    print("\n[2/4] Computing normalized curves and betas...")
    
    normalized_results = {}
    beta_results = {}
    
    for cond_key in all_results.keys():
        d, N = cond_key
        normalized_results[cond_key] = {}
        beta_results[cond_key] = {}
        
        for est_name in ['PR', 'Entropy', 'PCA_95']:
            mean_curve = all_results[cond_key]['aggregated'][est_name]['mean']
            k_vals = sorted(mean_curve.keys())
            values = np.array([mean_curve[k] for k in k_vals])
            
            max_val = np.max(values) if np.max(values) > 0 else 1.0
            norm_values = values / max_val
            
            normalized_results[cond_key][est_name] = {
                'k': k_vals,
                'values': norm_values.tolist()
            }
            
            beta_vals = []
            for i in range(len(k_vals) - 1):
                delta_D = norm_values[i+1] - norm_values[i]
                delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
                if delta_log_k > 0:
                    beta_vals.append(delta_D / delta_log_k)
                else:
                    beta_vals.append(np.nan)
            
            beta_results[cond_key][est_name] = beta_vals
    
    print("\n[3/4] Computing alignment matrices...")
    
    alignment_results = []
    
    for est_name in ['PR', 'Entropy', 'PCA_95']:
        curve_pairs = []
        
        for cond_key1 in normalized_results.keys():
            for cond_key2 in normalized_results.keys():
                if cond_key1 == cond_key2:
                    continue
                
                curve1 = normalized_results[cond_key1][est_name]
                curve2 = normalized_results[cond_key2][est_name]
                
                k_set1 = set(curve1['k'])
                k_set2 = set(curve2['k'])
                k_common = sorted(k_set1 & k_set2)
                
                if len(k_common) < 3:
                    continue
                
                v1 = np.array([curve1['values'][curve1['k'].index(k)] for k in k_common])
                v2 = np.array([curve2['values'][curve2['k'].index(k)] for k in k_common])
                
                if np.std(v1) > 0 and np.std(v2) > 0:
                    r, _ = pearsonr(v1, v2)
                else:
                    r = np.nan
                
                alignment_results.append({
                    'estimator': est_name,
                    'condition1': str(cond_key1),
                    'condition2': str(cond_key2),
                    'alignment_r': r
                })
    
    print("\n[4/4] Computing summary statistics...")
    
    growth_flags = []
    sat_flags = []
    summary_data = []
    
    for cond_key in all_results.keys():
        d, N = cond_key
        
        for est_name in ['PR', 'Entropy', 'PCA_95']:
            mean_curve = all_results[cond_key]['aggregated'][est_name]['mean']
            k_vals = sorted(mean_curve.keys())
            values = [mean_curve[k] for k in k_vals]
            
            has_growth = check_growth(values)
            has_saturation = check_saturation(values)
            
            growth_flags.append({
                'dimension': d,
                'sample_size': N,
                'estimator': est_name,
                'growth_present': has_growth
            })
            
            sat_flags.append({
                'dimension': d,
                'sample_size': N,
                'estimator': est_name,
                'saturation_present': has_saturation
            })
    
    for est_name in ['PR', 'Entropy', 'PCA_95']:
        est_alignments = [r['alignment_r'] for r in alignment_results if r['estimator'] == est_name]
        est_alignments = [x for x in est_alignments if not np.isnan(x)]
        
        est_growth = [g for g in growth_flags if g['estimator'] == est_name]
        est_sat = [s for s in sat_flags if s['estimator'] == est_name]
        
        summary_data.append({
            'dimension': 'all',
            'sample_size': 'all',
            'estimator': est_name,
            'growth_present': all([g['growth_present'] for g in est_growth]),
            'saturation_present': all([s['saturation_present'] for s in est_sat]) if len(est_sat) > 0 else False,
            'mean_alignment': np.mean(est_alignments) if len(est_alignments) > 0 else np.nan,
            'std_alignment': np.std(est_alignments) if len(est_alignments) > 0 else np.nan
        })
    
    for d in DIMENSIONS:
        for N in SAMPLE_SIZES:
            cond_key = (d, N)
            if cond_key not in all_results:
                continue
            
            for est_name in ['PR', 'Entropy', 'PCA_95']:
                mean_curve = all_results[cond_key]['aggregated'][est_name]['mean']
                k_vals = sorted(mean_curve.keys())
                values = [mean_curve[k] for k in k_vals]
                
                has_growth = check_growth(values)
                has_saturation = check_saturation(values)
                
                est_align = [r['alignment_r'] for r in alignment_results 
                           if r['estimator'] == est_name 
                           and str(d) in r['condition1']]
                est_align = [x for x in est_align if not np.isnan(x)]
                
                summary_data.append({
                    'dimension': d,
                    'sample_size': N,
                    'estimator': est_name,
                    'growth_present': has_growth,
                    'saturation_present': has_saturation,
                    'mean_alignment': np.mean(est_align) if len(est_align) > 0 else np.nan,
                    'std_alignment': np.std(est_align) if len(est_align) > 0 else np.nan
                })
    
    print("\n[5/5] Saving results...")
    
    np.save(results_dir / "raw_curves.npy", all_results)
    np.save(results_dir / "normalized_curves.npy", normalized_results)
    np.save(results_dir / "beta_curves.npy", beta_results)
    
    align_df = pd.DataFrame(alignment_results)
    align_df.to_csv(results_dir / "alignment_matrix.csv", index=False)
    
    growth_df = pd.DataFrame(growth_flags)
    growth_df.to_csv(results_dir / "growth_flags.csv", index=False)
    
    sat_df = pd.DataFrame(sat_flags)
    sat_df.to_csv(results_dir / "saturation_flags.csv", index=False)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "summary_table.csv", index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nTotal conditions completed: {conditions_completed}")
    print(f"Total runs: {conditions_completed * N_RUNS}")
    
    print("\nGrowth presence by estimator:")
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_df = growth_df[growth_df['estimator'] == est]
        count = est_df['growth_present'].sum()
        total = len(est_df)
        print(f"  {est}: {count}/{total}")
    
    print("\nSaturation presence by estimator:")
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_df = sat_df[sat_df['estimator'] == est]
        count = est_df['saturation_present'].sum()
        total = len(est_df)
        print(f"  {est}: {count}/{total}")
    
    print("\nAlignment ranges by estimator:")
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_align = align_df[align_df['estimator'] == est]['alignment_r'].dropna()
        if len(est_align) > 0:
            print(f"  {est}: mean={est_align.mean():.4f}, std={est_align.std():.4f}, min={est_align.min():.4f}, max={est_align.max():.4f}")
    
    return all_results, normalized_results, align_df, growth_df, sat_df, summary_df


def write_final_summary(all_results, align_df, growth_df, sat_df, summary_df):
    """Write final_summary.txt."""
    
    print("\n" + "=" * 70)
    print("WRITING FINAL SUMMARY")
    print("=" * 70)
    
    lines = []
    lines.append("=" * 70)
    lines.append("RANDOM MATRIX ABLATION - FINAL SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    total_conditions = len(all_results)
    total_runs = total_conditions * N_RUNS
    
    lines.append(f"Total conditions tested: {total_conditions}")
    lines.append(f"Total runs completed: {total_runs}")
    lines.append("")
    
    lines.append("GROWTH OBSERVATION")
    lines.append("-" * 50)
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_df = growth_df[growth_df['estimator'] == est]
        count = est_df['growth_present'].sum()
        total = len(est_df)
        all_present = count == total
        lines.append(f"  {est}: {'YES' if all_present else 'NO'} ({count}/{total} conditions)")
    lines.append("")
    
    lines.append("SATURATION OBSERVATION")
    lines.append("-" * 50)
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_df = sat_df[sat_df['estimator'] == est]
        count = est_df['saturation_present'].sum()
        total = len(est_df)
        all_present = count == total
        lines.append(f"  {est}: {'YES' if all_present else 'NO'} ({count}/{total} conditions)")
    lines.append("")
    
    lines.append("ALIGNMENT RANGES (Cross-condition Pearson r)")
    lines.append("-" * 50)
    significant_deviators = []
    for est in ['PR', 'Entropy', 'PCA_95']:
        est_align = align_df[align_df['estimator'] == est]['alignment_r'].dropna()
        if len(est_align) > 0:
            mean_r = est_align.mean()
            std_r = est_align.std()
            lines.append(f"  {est}:")
            lines.append(f"    Mean: {mean_r:.4f}")
            lines.append(f"    Std: {std_r:.4f}")
            lines.append(f"    Min: {est_align.min():.4f}")
            lines.append(f"    Max: {est_align.max():.4f}")
            if std_r > 0.3:
                significant_deviators.append(est)
    lines.append("")
    
    lines.append("ESTIMATORS WITH SIGNIFICANT DEVIATION (std > 0.3)")
    lines.append("-" * 50)
    if len(significant_deviators) > 0:
        for est in significant_deviators:
            lines.append(f"  - {est}")
    else:
        lines.append("  None")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END")
    lines.append("=" * 70)
    
    summary_text = "\n".join(lines)
    
    summary_path = Path(__file__).parent / "final_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n[Summary saved to: {summary_path}]")
    
    return summary_text


if __name__ == "__main__":
    try:
        all_results, normalized_results, align_df, growth_df, sat_df, summary_df = run_experiment()
        final_summary = write_final_summary(
            all_results, align_df, growth_df, sat_df, summary_df
        )
        print("\n[SUCCESS] Random matrix ablation completed successfully")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
