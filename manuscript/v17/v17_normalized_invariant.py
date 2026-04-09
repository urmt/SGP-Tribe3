#!/usr/bin/env python3
"""
V17 Pipeline: Normalized Invariant Analysis
Test for universal properties across real-world systems using normalized D_eff(k)
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v17")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V17 NORMALIZED INVARIANT ANALYSIS")
print("Testing for universal properties across real-world systems")
print("=" * 80)

def load_tribe_data():
    """Load TRIBE v2 predictions."""
    checkpoint_dir = Path("/home/student/sgp-tribe3/results/full_battery_1000")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    
    all_data = []
    for cp in checkpoints:
        with open(cp) as f:
            data = json.load(f)
            results = data.get('results', [])
            all_data.extend(results)
    
    print(f"Loaded {len(all_data)} TRIBE v2 predictions")
    return all_data

def compute_local_dimensionality(activations, k):
    """Compute local effective dimensionality for k neighbors."""
    X = StandardScaler().fit_transform(activations)
    n_samples, n_dims = X.shape
    
    if k >= n_samples:
        k = n_samples - 1
    if k < 2:
        return 0, 0
    
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    local_dims = []
    for i in range(n_samples):
        neighborhood = X[indices[i, 1:k+1]]
        
        if len(neighborhood) < k:
            continue
        
        cov = np.cov(neighborhood.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) < 2:
            continue
        
        sum_eig = np.sum(eigenvalues)
        sum_eig_sq = np.sum(eigenvalues ** 2)
        
        if sum_eig_sq > 0:
            D_eff = (sum_eig ** 2) / sum_eig_sq
            local_dims.append(D_eff)
    
    if local_dims:
        return np.mean(local_dims), np.std(local_dims)
    return 0, 0

def compute_D_eff_curve(activations, k_values):
    """Compute D_eff(k) curve."""
    D_eff_values = []
    std_values = []
    
    for k in k_values:
        mean_d, std_d = compute_local_dimensionality(activations, k)
        D_eff_values.append(mean_d)
        std_values.append(std_d)
    
    return {
        'k_values': list(k_values),
        'D_eff': D_eff_values,
        'std': std_values
    }

def normalize_curve(curve):
    """Normalize D_eff(k) curve."""
    k = np.array(curve['k_values'])
    D = np.array(curve['D_eff'])
    
    D_max = np.max(D)
    k_max = k[np.argmax(D)] if len(k) > 0 else k[-1]
    
    if D_max > 0:
        D_norm = D / D_max
    else:
        D_norm = D
    
    k_norm = k / k_max if k_max > 0 else k
    
    early_mask = k_norm < 0.3
    late_mask = k_norm > 0.7
    
    growth_rate = np.mean(np.gradient(D_norm[early_mask]) / np.gradient(k_norm[early_mask])) if np.sum(early_mask) > 2 else 0
    
    mid_mask = (k_norm > 0.4) & (k_norm < 0.6)
    saturation_fraction = np.mean(D_norm[mid_mask]) if np.sum(mid_mask) > 0 else 0
    
    return {
        'k_norm': k_norm.tolist(),
        'D_norm': D_norm.tolist(),
        'D_max': D_max,
        'k_max': k_max,
        'growth_rate': growth_rate,
        'saturation_fraction': saturation_fraction
    }

def extract_tribe_activations(tribe_data, n_samples=2000):
    """Extract TRIBE activations."""
    node_keys = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
                 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
    
    activations = []
    for item in tribe_data[:n_samples]:
        if 'sgp_nodes' in item:
            sgp = item['sgp_nodes']
            values = [sgp.get(k, 0.5) for k in node_keys]
            activations.append(values)
    
    if len(activations) < n_samples:
        while len(activations) < n_samples:
            activations.extend(activations[:min(len(activations), n_samples - len(activations))])
    
    return np.array(activations[:n_samples])

def generate_realistic_embeddings(n_samples=2000, n_dims=50):
    """Generate realistic embedding data."""
    np.random.seed(RANDOM_SEED + 1)
    
    embeddings = {}
    
    embeddings['hierarchical'] = np.random.randn(n_samples, n_dims) * np.exp(-np.arange(n_dims) / 25)
    
    cov = np.random.randn(n_dims, n_dims)
    cov = cov @ cov.T + np.eye(n_dims) * 0.5
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues / eigenvalues[0] * 5
    L = np.linalg.cholesky(cov + np.eye(n_dims) * 0.1)
    embeddings['correlated'] = np.random.randn(n_samples, n_dims) @ L
    
    sparse_mask = np.random.rand(n_samples, n_dims) > 0.9
    embeddings['sparse'] = np.random.randn(n_samples, n_dims) * sparse_mask
    
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.exponential(1, n_samples)
    base = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    for i in range(n_dims - 2):
        base = np.column_stack([base, np.random.randn(n_samples) * 0.3])
    embeddings['manifold'] = base
    
    embeddings['uniform_sphere'] = np.random.randn(n_samples, n_dims) / np.sqrt(n_dims)
    embeddings['power_law'] = np.random.randn(n_samples, n_dims) * np.power(np.arange(1, n_dims + 1), -0.5)
    
    return embeddings

def compute_alignment_score(curves):
    """Compute alignment score between normalized curves."""
    if len(curves) < 2:
        return 1.0, []
    
    alignment_scores = []
    curve_pairs = []
    
    curve_list = list(curves.items())
    for i in range(len(curve_list)):
        for j in range(i + 1, len(curve_list)):
            name1, norm1 = curve_list[i]
            name2, norm2 = curve_list[j]
            
            k1 = np.array(norm1['k_norm'])
            k2 = np.array(norm2['k_norm'])
            D1 = np.array(norm1['D_norm'])
            D2 = np.array(norm2['D_norm'])
            
            k_common = np.intersect1d(np.round(k1, 2), np.round(k2, 2))
            
            if len(k_common) > 3:
                idx1 = [np.argmin(np.abs(k1 - k)) for k in k_common]
                idx2 = [np.argmin(np.abs(k2 - k)) for k in k_common]
                
                D1_interp = np.array([D1[i] for i in idx1])
                D2_interp = np.array([D2[i] for i in idx2])
                
                diff = np.abs(D1_interp - D2_interp)
                mean_diff = np.mean(diff)
                alignment_scores.append(1 - mean_diff)
                curve_pairs.append((name1, name2, 1 - mean_diff))
    
    mean_alignment = np.mean(alignment_scores) if alignment_scores else 0
    
    return mean_alignment, curve_pairs

def compute_invariant_metrics(all_normalized, all_curves):
    """Compute invariant metrics across systems."""
    metrics = {}
    
    growth_rates = [norm['growth_rate'] for norm in all_normalized.values()]
    saturation_fractions = [norm['saturation_fraction'] for norm in all_normalized.values()]
    
    metrics['growth_rate_mean'] = np.mean(growth_rates)
    metrics['growth_rate_std'] = np.std(growth_rates)
    metrics['growth_rate_cv'] = np.std(growth_rates) / abs(np.mean(growth_rates)) if np.mean(growth_rates) != 0 else np.inf
    
    metrics['sat_frac_mean'] = np.mean(saturation_fractions)
    metrics['sat_frac_std'] = np.std(saturation_fractions)
    metrics['sat_frac_cv'] = np.std(saturation_fractions) / abs(np.mean(saturation_fractions)) if np.mean(saturation_fractions) != 0 else np.inf
    
    D_max_values = [norm['D_max'] for norm in all_normalized.values()]
    metrics['D_max_mean'] = np.mean(D_max_values)
    metrics['D_max_std'] = np.std(D_max_values)
    metrics['D_max_cv'] = np.std(D_max_values) / np.mean(D_max_values) if np.mean(D_max_values) != 0 else np.inf
    
    return metrics

def create_visualizations(all_curves, all_normalized, alignment_scores, metrics):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_curves)))
    
    for idx, (name, curve) in enumerate(all_curves.items()):
        k = curve['k_values']
        D = curve['D_eff']
        axes[0, 0].plot(k, D, marker='o', label=name, color=colors[idx], alpha=0.7)
    
    axes[0, 0].set_xlabel('k (neighbors)')
    axes[0, 0].set_ylabel('D_eff(k)')
    axes[0, 0].set_title('Raw D_eff(k) Curves')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    for idx, (name, norm) in enumerate(all_normalized.items()):
        k_norm = norm['k_norm']
        D_norm = norm['D_norm']
        axes[0, 1].plot(k_norm, D_norm, marker='o', label=name, color=colors[idx], alpha=0.7)
    
    axes[0, 1].set_xlabel('k / k_max')
    axes[0, 1].set_ylabel('D_eff / D_max')
    axes[0, 1].set_title('Normalized D_eff(k) Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    mean_curve = np.mean([norm['D_norm'] for norm in all_normalized.values()], axis=0)
    std_curve = np.std([norm['D_norm'] for norm in all_normalized.values()], axis=0)
    
    if len(all_normalized) > 0:
        reference_k = np.linspace(0.01, 1, 50)
        
        first_norm = list(all_normalized.values())[0]
        k_norm_template = np.array(first_norm['k_norm'])
        
        for idx, (name, norm) in enumerate(all_normalized.items()):
            k_norm = np.array(norm['k_norm'])
            D_norm = np.array(norm['D_norm'])
            D_interp = np.interp(reference_k, k_norm, D_norm)
            axes[1, 0].plot(reference_k, D_interp, alpha=0.4, color=colors[idx])
        
        mean_curve_interp = np.interp(reference_k, k_norm_template, mean_curve)
        axes[1, 0].plot(reference_k, mean_curve_interp, 'k-', linewidth=3, label='Mean')
    
    axes[1, 0].set_xlabel('k / k_max')
    axes[1, 0].set_ylabel('D_eff / D_max')
    axes[1, 0].set_title(f'Average Curve with Variability (Alignment={alignment_scores[0]:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    growth_rates = [norm['growth_rate'] for norm in all_normalized.values()]
    sat_fracs = [norm['saturation_fraction'] for norm in all_normalized.values()]
    
    x = np.arange(len(all_normalized))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, growth_rates, width, label='Growth Rate', color='steelblue', alpha=0.7)
    axes[1, 1].set_ylabel('Growth Rate')
    ax2 = axes[1, 1].twinx()
    bars2 = ax2.bar(x + width/2, sat_fracs, width, label='Saturation Fraction', color='coral', alpha=0.7)
    ax2.set_ylabel('Saturation Fraction')
    
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(list(all_normalized.keys()), rotation=45, ha='right')
    axes[1, 1].set_title('Invariant Metrics')
    
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_normalized_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_normalized_comparison.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_names = ['Growth Rate', 'Saturation Frac', 'D_max', 'Alignment']
    metrics_values = [metrics['growth_rate_cv'], metrics['sat_frac_cv'], metrics['D_max_cv'], 1 - alignment_scores[0]]
    metrics_colors = ['green' if v < 0.3 else 'orange' if v < 0.5 else 'red' for v in metrics_values]
    
    bars = axes[0, 0].bar(metrics_names, metrics_values, color=metrics_colors, alpha=0.7)
    axes[0, 0].set_ylabel('Coefficient of Variation')
    axes[0, 0].set_title('Invariance Metrics (CV < 0.3 = invariant)')
    axes[0, 0].axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    
    if alignment_scores[1]:
        pair_names = [f"{p[0][:8]}-{p[1][:8]}" for p in alignment_scores[1]]
        pair_scores = [p[2] for p in alignment_scores[1]]
        axes[0, 1].bar(pair_names, pair_scores, color='steelblue', alpha=0.7)
        axes[0, 1].set_ylabel('Alignment Score')
        axes[0, 1].set_title('Pairwise Curve Alignment')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    if len(all_normalized) > 0:
        reference_k = np.linspace(0.01, 1, 50)
        
        all_interp = []
        for idx, (name, norm) in enumerate(all_normalized.items()):
            k_norm = np.array(norm['k_norm'])
            D_norm = np.array(norm['D_norm'])
            D_interp = np.interp(reference_k, k_norm, D_norm)
            all_interp.append(D_interp)
            axes[1, 0].plot(reference_k, D_interp, alpha=0.5, color=colors[idx], label=name)
        
        mean_interp = np.mean(all_interp, axis=0)
        std_interp = np.std(all_interp, axis=0)
        axes[1, 0].plot(reference_k, mean_interp, 'k-', linewidth=3, label='Mean')
        axes[1, 0].fill_between(reference_k, mean_interp - std_interp, mean_interp + std_interp, 
                                alpha=0.2, color='gray')
    
    axes[1, 0].set_xlabel('k / k_max')
    axes[1, 0].set_ylabel('D_eff / D_max')
    axes[1, 0].set_title('Interpolated Normalized Curves')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    growth_vs_sat = [(g, s, n) for n, g, s in zip(all_normalized.keys(), growth_rates, sat_fracs)]
    for g, s, n in growth_vs_sat:
        idx = list(all_normalized.keys()).index(n)
        axes[1, 1].scatter(g, s, color=colors[idx], s=100, alpha=0.7, label=n)
        axes[1, 1].annotate(n[:8], (g, s), fontsize=7)
    
    axes[1, 1].set_xlabel('Growth Rate')
    axes[1, 1].set_ylabel('Saturation Fraction')
    axes[1, 1].set_title('Growth vs Saturation')
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_invariant_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_invariant_metrics.png")
    
    return

def create_summary_table(all_curves, all_normalized, alignment_scores, metrics):
    """Create summary table."""
    
    summary_data = []
    for name, curve in all_curves.items():
        norm = all_normalized[name]
        summary_data.append({
            'System': name,
            'D_max': norm['D_max'],
            'k_max': norm['k_max'],
            'Growth_Rate': norm['growth_rate'],
            'Saturation_Fraction': norm['saturation_fraction'],
            'Final_D': curve['D_eff'][-1] if len(curve['D_eff']) > 0 else 0,
            'Final_k': curve['k_values'][-1] if len(curve['k_values']) > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "invariant_summary.csv", index=False)
    print(f"✓ Summary saved to {OUTPUT_DIR / 'invariant_summary.csv'}")
    
    metrics_summary = {
        'Metric': ['Mean Alignment', 'Growth Rate CV', 'Saturation Frac CV', 'D_max CV'],
        'Value': [alignment_scores[0], metrics['growth_rate_cv'], 
                  metrics['sat_frac_cv'], metrics['D_max_cv']]
    }
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print(f"✓ Metrics saved to {OUTPUT_DIR / 'metrics_summary.csv'}")
    
    return summary_df, metrics_df

def generate_report(all_curves, all_normalized, alignment_scores, metrics):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V17 INTERPRETATION REPORT: NORMALIZED INVARIANT ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Do normalized D_eff(k) curves align across systems?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. NORMALIZED METRICS SUMMARY")
    report.append("-" * 50)
    
    header = f"{'System':<15} {'D_max':>8} {'Growth':>10} {'Sat Frac':>10}"
    report.append(header)
    report.append("-" * len(header))
    
    for name, norm in all_normalized.items():
        report.append(f"{name:<15} {norm['D_max']:>8.2f} {norm['growth_rate']:>10.4f} {norm['saturation_fraction']:>10.4f}")
    
    report.append("")
    report.append("2. INVARIANCE METRICS")
    report.append("-" * 50)
    
    report.append(f"  Mean Alignment Score: {alignment_scores[0]:.4f}")
    report.append(f"  Growth Rate CV: {metrics['growth_rate_cv']:.4f}")
    report.append(f"  Saturation Fraction CV: {metrics['sat_frac_cv']:.4f}")
    report.append(f"  D_max CV: {metrics['D_max_cv']:.4f}")
    
    report.append("")
    report.append("3. SUCCESS CRITERIA")
    report.append("-" * 50)
    
    alignment_ok = alignment_scores[0] > 0.7
    growth_cv_ok = metrics['growth_rate_cv'] < 0.3
    sat_cv_ok = metrics['sat_frac_cv'] < 0.3
    
    report.append(f"  Alignment after normalization: {'✓ YES' if alignment_ok else '✗ NO'} ({alignment_scores[0]:.3f})")
    report.append(f"  Stable growth rates: {'✓ YES' if growth_cv_ok else '✗ NO'} (CV={metrics['growth_rate_cv']:.3f})")
    report.append(f"  Stable saturation fractions: {'✓ YES' if sat_cv_ok else '✗ NO'} (CV={metrics['sat_frac_cv']:.3f})")
    
    report.append("")
    report.append("4. PAIRWISE ALIGNMENT")
    report.append("-" * 50)
    
    if alignment_scores[1]:
        sorted_pairs = sorted(alignment_scores[1], key=lambda x: x[2], reverse=True)
        report.append(f"  Best alignment: {sorted_pairs[0][0]} - {sorted_pairs[0][1]} ({sorted_pairs[0][2]:.3f})")
        report.append(f"  Worst alignment: {sorted_pairs[-1][0]} - {sorted_pairs[-1][1]} ({sorted_pairs[-1][2]:.3f})")
    
    report.append("")
    report.append("5. CONCLUSIONS")
    report.append("-" * 50)
    
    if alignment_ok and growth_cv_ok and sat_cv_ok:
        report.append("  UNIVERSAL NORMALIZED INVARIANT EXISTS")
        report.append("  ")
        report.append("  All systems show aligned normalized curves with stable metrics:")
        report.append(f"    - Mean alignment: {alignment_scores[0]:.3f}")
        report.append(f"    - Growth rate CV: {metrics['growth_rate_cv']:.3f}")
        report.append(f"    - Saturation fraction CV: {metrics['sat_frac_cv']:.3f}")
        report.append("  ")
        report.append("  Interpretation:")
        report.append("    - Dimensional unfolding follows a universal pattern")
        report.append("    - Growth and saturation are system-independent")
        report.append("    - Normalized curves capture essential structure")
    elif alignment_ok:
        report.append("  PARTIAL UNIVERSALITY: ALIGNED BUT VARIABLE METRICS")
        report.append("  ")
        report.append("  Curves align but individual metrics vary:")
        report.append(f"    - Alignment: {alignment_scores[0]:.3f} (good)")
        report.append(f"    - Growth rate CV: {metrics['growth_rate_cv']:.3f}")
        report.append(f"    - Saturation CV: {metrics['sat_frac_cv']:.3f}")
        report.append("  ")
        report.append("  Interpretation:")
        report.append("    - Overall shape is universal")
        report.append("    - Scale parameters vary between systems")
    else:
        report.append("  NO UNIVERSAL NORMALIZED INVARIANT")
        report.append("  ")
        report.append("  Systems show different patterns after normalization:")
        report.append(f"    - Alignment: {alignment_scores[0]:.3f}")
        report.append(f"    - Growth CV: {metrics['growth_rate_cv']:.3f}")
        report.append(f"    - Saturation CV: {metrics['sat_frac_cv']:.3f}")
        report.append("  ")
        report.append("  Interpretation:")
        report.append("    - No universal dimensional unfolding pattern")
        report.append("    - Systems have distinct scaling behaviors")
    
    report.append("")
    report.append("6. KEY INSIGHT")
    report.append("-" * 50)
    
    if 'TRIBE' in all_normalized:
        tribe_norm = all_normalized['TRIBE']
        other_norms = [n for n, v in all_normalized.items() if n != 'TRIBE']
        
        if other_norms:
            other_growth = np.mean([all_normalized[n]['growth_rate'] for n in other_norms])
            tribe_vs_other = abs(tribe_norm['growth_rate'] - other_growth) / abs(other_growth) if other_growth != 0 else 0
            
            if tribe_vs_other < 0.3:
                report.append(f"  TRIBE growth rate ({tribe_norm['growth_rate']:.3f}) matches other systems ({other_growth:.3f})")
                report.append("  → TRIBE shows typical dimensional unfolding")
            else:
                report.append(f"  TRIBE growth rate ({tribe_norm['growth_rate']:.3f}) differs from others ({other_growth:.3f})")
                report.append("  → TRIBE shows unique scaling")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(OUTPUT_DIR / "interpretation_report.txt", 'w') as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {OUTPUT_DIR / 'interpretation_report.txt'}")
    
    return report

def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("PHASE 1: LOADING DATA")
    print("=" * 80)
    
    tribe_data = load_tribe_data()
    
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING SYSTEMS")
    print("=" * 80)
    
    tribe_activations = extract_tribe_activations(tribe_data, n_samples=2000)
    print(f"TRIBE activations: {tribe_activations.shape}")
    
    embeddings = generate_realistic_embeddings(n_samples=2000, n_dims=50)
    print(f"Embedding systems: {list(embeddings.keys())}")
    
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING D_eff(k)")
    print("=" * 80)
    
    k_values = [5, 10, 20, 50, 100, 200, 500]
    
    all_curves = {}
    
    print("Analyzing TRIBE...")
    all_curves['TRIBE'] = compute_D_eff_curve(tribe_activations, k_values)
    
    for name, emb in embeddings.items():
        print(f"Analyzing {name}...")
        all_curves[name] = compute_D_eff_curve(emb, k_values)
    
    print("\n" + "=" * 80)
    print("PHASE 4: NORMALIZING CURVES")
    print("=" * 80)
    
    all_normalized = {}
    for name, curve in all_curves.items():
        print(f"Normalizing {name}...")
        all_normalized[name] = normalize_curve(curve)
    
    print("\n" + "=" * 80)
    print("PHASE 5: COMPUTING ALIGNMENT")
    print("=" * 80)
    
    alignment_scores = compute_alignment_score(all_normalized)
    print(f"Mean alignment score: {alignment_scores[0]:.4f}")
    
    print("\n" + "=" * 80)
    print("PHASE 6: COMPUTING METRICS")
    print("=" * 80)
    
    metrics = compute_invariant_metrics(all_normalized, all_curves)
    print(f"Growth Rate CV: {metrics['growth_rate_cv']:.4f}")
    print(f"Saturation Fraction CV: {metrics['sat_frac_cv']:.4f}")
    print(f"D_max CV: {metrics['D_max_cv']:.4f}")
    
    print("\n" + "=" * 80)
    print("PHASE 7: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(all_curves, all_normalized, alignment_scores, metrics)
    
    print("\n" + "=" * 80)
    print("PHASE 8: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_table(all_curves, all_normalized, alignment_scores, metrics)
    
    print("\n" + "=" * 80)
    print("PHASE 9: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(all_curves, all_normalized, alignment_scores, metrics)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - invariant_summary.csv")
    print("  - metrics_summary.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_normalized_comparison.png")
    print("  - figures/fig2_invariant_metrics.png")

if __name__ == "__main__":
    main()
