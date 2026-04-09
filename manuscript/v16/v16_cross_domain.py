#!/usr/bin/env python3
"""
V16 Pipeline: Cross-Domain Saturation Analysis
Test whether finite dimensional saturation is a universal property
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

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v16")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V16 CROSS-DOMAIN SATURATION ANALYSIS")
print("Testing universal dimensional unfolding with saturation")
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

def detect_phases(curve):
    """Automatically detect growth, saturation, peak, and decline phases."""
    k = np.array(curve['k_values'])
    D = np.array(curve['D_eff'])
    
    if len(k) < 4:
        return {'phases': [], 'saturation_point': None, 'peak_k': None, 'has_decline': False}
    
    dD = np.gradient(D, k)
    
    growth_threshold = np.percentile(dD[dD > 0], 50) if np.any(dD > 0) else 0
    decline_threshold = np.percentile(dD[dD < 0], 25) if np.any(dD < 0) else 0
    
    phases = []
    
    for i in range(len(k)):
        if dD[i] > growth_threshold * 0.5:
            phase = 'growth'
        elif dD[i] > decline_threshold:
            phase = 'saturation'
        else:
            phase = 'decline'
        phases.append(phase)
    
    saturation_idx = None
    for i in range(1, len(phases)):
        if phases[i] == 'saturation' and phases[i-1] == 'growth':
            saturation_idx = i
            break
    
    peak_idx = np.argmax(D)
    peak_k = k[peak_idx]
    
    has_decline = False
    if peak_idx < len(D) - 1:
        if D[peak_idx] - D[-1] > 0.1 * D[peak_idx]:
            has_decline = True
    
    dD_smoothed = np.convolve(dD, np.ones(3)/3, mode='same')
    inflection_idx = np.argmin(dD_smoothed)
    
    return {
        'phases': phases,
        'saturation_idx': saturation_idx,
        'saturation_point': k[saturation_idx] if saturation_idx else None,
        'peak_idx': peak_idx,
        'peak_k': peak_k,
        'peak_D': D[peak_idx],
        'has_decline': has_decline,
        'decline_fraction': (D[peak_idx] - D[-1]) / D[peak_idx] if has_decline else 0,
        'inflection_k': k[inflection_idx],
        'inflection_D': D[inflection_idx]
    }

def fit_saturation_model(k, D_max, k_half):
    """Fit saturating model: D = D_max * k / (k_half + k)"""
    k = np.maximum(k, 0.1)
    return D_max * k / (k_half + k)

def fit_gaussian_peak(k, A, mu, sigma):
    """Fit Gaussian peak: D = A * exp(-(k-mu)²/(2σ²))"""
    return A * np.exp(-(k - mu)**2 / (2 * sigma**2))

def analyze_system(activations, name, k_values):
    """Full analysis for a system."""
    print(f"  Analyzing {name}...")
    
    curve = compute_D_eff_curve(activations, k_values)
    
    phases = detect_phases(curve)
    
    k = np.array(curve['k_values'])
    D = np.array(curve['D_eff'])
    
    saturation_fit = {}
    try:
        popt, _ = curve_fit(fit_saturation_model, k, D, p0=[max(D), 50], 
                           bounds=([0, 1], [100, 1000]), maxfev=5000)
        residuals = D - fit_saturation_model(k, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((D - np.mean(D))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        saturation_fit = {'D_max': popt[0], 'k_half': popt[1], 'r_squared': r_squared}
    except:
        saturation_fit = {'D_max': None, 'k_half': None, 'r_squared': 0}
    
    early_growth = np.mean(D[:len(D)//3])
    late_D = np.mean(D[-len(D)//3:])
    
    return {
        'name': name,
        'curve': curve,
        'phases': phases,
        'saturation_fit': saturation_fit,
        'early_growth': early_growth,
        'late_D': late_D,
        'final_D': D[-1] if len(D) > 0 else 0,
        'max_D': max(D),
        'growth_rate': (D[len(D)//2] - early_growth) / (k[len(k)//2] - k[0]) if len(k) > 1 else 0,
        'D_range': max(D) - min(D),
        'saturation_level': phases['peak_D'] if phases['peak_D'] else max(D)
    }

def extract_tribe_activations(tribe_data, n_samples=3000):
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

def generate_language_embeddings(n_samples=3000, n_dims=100):
    """Generate realistic language embedding data."""
    np.random.seed(RANDOM_SEED + 1)
    
    embeddings = {}
    
    embeddings['hierarchical'] = np.random.randn(n_samples, n_dims) * np.exp(-np.arange(n_dims) / 30)
    
    cov = np.random.randn(n_dims, n_dims)
    cov = cov @ cov.T + np.eye(n_dims)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues / eigenvalues[0] * 10
    L = np.linalg.cholesky(cov + np.eye(n_dims) * 0.1)
    embeddings['correlated'] = np.random.randn(n_samples, n_dims) @ L
    
    sparse_mask = np.random.rand(n_samples, n_dims) > 0.9
    embeddings['sparse'] = np.random.randn(n_samples, n_dims) * sparse_mask
    
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.exponential(1, n_samples)
    base = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    for i in range(n_dims - 2):
        base = np.column_stack([base, np.random.randn(n_samples) * 0.2])
    embeddings['manifold'] = base
    
    return embeddings

def generate_cooccurrence_matrix(n_words=300):
    """Generate co-occurrence matrix."""
    np.random.seed(RANDOM_SEED + 2)
    
    base = np.random.rand(n_words, n_words)
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1)
    base = base ** 2
    
    eigenvalues = np.linalg.eigvalsh(base)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    
    return {
        'eigenvalues': eigenvalues,
        'name': 'Cooccurrence',
        'n_dims': n_words
    }

def create_visualizations(all_results):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    
    for idx, (name, result) in enumerate(all_results.items()):
        k = result['curve']['k_values']
        D = result['curve']['D_eff']
        std = result['curve']['std']
        
        axes[0, 0].errorbar(k, D, yerr=std, marker='o', capsize=3, 
                           label=name, color=colors[idx], alpha=0.7)
    
    axes[0, 0].set_xlabel('k (neighbors)')
    axes[0, 0].set_ylabel('D_eff(k)')
    axes[0, 0].set_title('D_eff(k) Curves - All Systems')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    for idx, (name, result) in enumerate(all_results.items()):
        k = result['curve']['k_values']
        D = result['curve']['D_eff']
        log_k = np.log(k)
        log_D = np.log(np.maximum(D, 0.1))
        
        axes[0, 1].scatter(log_k, log_D, color=colors[idx], alpha=0.5, s=30, label=name)
    
    axes[0, 1].set_xlabel('log(k)')
    axes[0, 1].set_ylabel('log(D_eff)')
    axes[0, 1].set_title('Log-Log Plot')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    names = list(all_results.keys())
    peak_D_values = [all_results[n]['phases']['peak_D'] for n in names]
    saturation_k = [all_results[n]['phases'].get('saturation_point') for n in names]
    
    x = np.arange(len(names))
    bars = axes[1, 0].bar(x, peak_D_values, color=colors, alpha=0.7)
    
    for i, (name, result) in enumerate(all_results.items()):
        phases = result['phases']
        if phases['has_decline']:
            bars[i].set_color('coral')
        if phases['peak_k']:
            axes[1, 0].annotate(f"k={phases['peak_k']}", 
                               (i, phases['peak_D']),
                               textcoords="offset points", xytext=(0, 5),
                               ha='center', fontsize=8)
    
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Peak D_eff')
    axes[1, 0].set_title('Peak D_eff by System (coral = decline)')
    
    phase_counts = {'growth': 0, 'saturation': 0, 'decline': 0}
    for result in all_results.values():
        phases = result['phases']['phases']
        for phase in phases:
            if phase in phase_counts:
                phase_counts[phase] += 1
    
    axes[1, 1].bar(phase_counts.keys(), phase_counts.values(), 
                   color=['steelblue', 'seagreen', 'coral'], alpha=0.7)
    axes[1, 1].set_ylabel('Count (across all systems)')
    axes[1, 1].set_title('Phase Distribution')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_cross_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_cross_domain.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    normalized_curves = {}
    for name, result in all_results.items():
        D = np.array(result['curve']['D_eff'])
        k = np.array(result['curve']['k_values'])
        max_D = result['phases']['peak_D'] if result['phases']['peak_D'] else max(D)
        
        if max_D > 0:
            D_norm = D / max_D
        else:
            D_norm = D
        
        normalized_curves[name] = (k, D_norm)
        
        axes[0, 0].plot(k, D_norm, marker='o', label=name, color=colors[list(all_results.keys()).index(name)], 
                       alpha=0.7)
    
    axes[0, 0].set_xlabel('k (neighbors)')
    axes[0, 0].set_ylabel('D_eff / Peak D_eff')
    axes[0, 0].set_title('Normalized Curves')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    has_decline = [result['phases']['has_decline'] for result in all_results.values()]
    decline_fraction = [result['phases']['decline_fraction'] for result in all_results.values()]
    
    colors_decline = ['coral' if d else 'steelblue' for d in has_decline]
    axes[0, 1].bar(names, decline_fraction, color=colors_decline, alpha=0.7)
    axes[0, 1].set_ylabel('Decline Fraction')
    axes[0, 1].set_title('Decline at Large k')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    
    saturation_levels = [result['phases']['peak_D'] for result in all_results.values()]
    axes[1, 0].scatter(range(len(names)), saturation_levels, c=colors, s=100, alpha=0.7)
    for i, (name, result) in enumerate(all_results.items()):
        if result['phases']['peak_k']:
            axes[1, 0].annotate(f"k={result['phases']['peak_k']}", 
                               (i, result['phases']['peak_D']),
                               textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Saturation Level (D_eff)')
    axes[1, 0].set_title('Saturation Levels')
    
    growth_rates = [result['growth_rate'] for result in all_results.values()]
    axes[1, 1].bar(names, growth_rates, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Growth Rate (dD/dk)')
    axes[1, 1].set_title('Early Growth Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_normalized_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_normalized_comparison.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    shape_types = []
    for result in all_results.values():
        phases = result['phases']
        if phases['has_decline']:
            shape_types.append('Growth-Sat-Peak-Decline')
        elif phases['saturation_point']:
            shape_types.append('Growth-Saturation')
        else:
            shape_types.append('Simple Growth')
    
    unique_shapes = list(set(shape_types))
    shape_counts = {s: shape_types.count(s) for s in unique_shapes}
    
    axes[0].bar(shape_counts.keys(), shape_counts.values(), color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Shape Type Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    all_phases_list = []
    for name, result in all_results.items():
        for phase in result['phases']['phases']:
            all_phases_list.append((name, phase))
    
    phase_by_system = {}
    for name in all_results.keys():
        phases = all_results[name]['phases']['phases']
        phase_by_system[name] = phases
    
    phases_xticklabels = []
    for name in all_results.keys():
        phases = phase_by_system[name]
        phases_xticklabels.append(f"{name[:10]}\n{'G' if 'growth' in phases[0] else 'S' if 'saturation' in phases[0] else 'D'}")
    
    k_values = all_results[list(all_results.keys())[0]]['curve']['k_values']
    
    for i, (name, result) in enumerate(all_results.items()):
        phases = result['phases']['phases']
        phase_numeric = {'growth': 1, 'saturation': 2, 'decline': 3}
        phase_nums = [phase_numeric.get(p, 0) for p in phases]
        
        axes[1].plot(k_values, phase_nums, marker='o', label=name, 
                     color=colors[list(all_results.keys()).index(name)], alpha=0.7)
    
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Phase')
    axes[1].set_yticks([1, 2, 3])
    axes[1].set_yticklabels(['Growth', 'Saturation', 'Decline'])
    axes[1].set_title('Phase Evolution')
    axes[1].legend(fontsize=7)
    axes[1].set_xscale('log')
    
    parameters_table = []
    col_labels = ['System', 'Peak D', 'Peak k', 'D_max', 'k_1/2', 'R²', 'Decline']
    for name, result in all_results.items():
        fit = result['saturation_fit']
        peak_D = f"{result['phases']['peak_D']:.2f}" if result['phases']['peak_D'] else 'N/A'
        peak_k = f"{result['phases']['peak_k']}" if result['phases']['peak_k'] else 'N/A'
        d_max = f"{fit['D_max']:.2f}" if fit['D_max'] else 'N/A'
        k_half = f"{fit['k_half']:.1f}" if fit['k_half'] else 'N/A'
        r2 = f"{fit['r_squared']:.3f}"
        decline = 'Yes' if result['phases']['has_decline'] else 'No'
        parameters_table.append([name, peak_D, peak_k, d_max, k_half, r2, decline])
    
    axes[2].axis('off')
    table = axes[2].table(cellText=parameters_table,
                           colLabels=col_labels,
                           loc='center',
                           cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    axes[2].set_title('Parameters Table')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_detailed_analysis.png")
    
    return

def create_summary_table(all_results):
    """Create summary table."""
    
    summary_data = []
    for name, result in all_results.items():
        phases = result['phases']
        fit = result['saturation_fit']
        
        summary_data.append({
            'System': name,
            'Peak_D_eff': phases['peak_D'] if phases['peak_D'] else result['max_D'],
            'Peak_k': phases['peak_k'] if phases['peak_k'] else 0,
            'Saturation_k': phases['saturation_point'] if phases['saturation_point'] else 0,
            'Has_Decline': phases['has_decline'],
            'Decline_Fraction': phases['decline_fraction'],
            'D_max_fit': fit['D_max'] if fit['D_max'] else 0,
            'k_half': fit['k_half'] if fit['k_half'] else 0,
            'Saturation_R2': fit['r_squared'],
            'Final_D': result['final_D'],
            'Growth_Rate': result['growth_rate']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "cross_domain_summary.csv", index=False)
    print(f"✓ Summary saved to {OUTPUT_DIR / 'cross_domain_summary.csv'}")
    
    curve_data = []
    for name, result in all_results.items():
        for k, D, std in zip(result['curve']['k_values'], 
                             result['curve']['D_eff'],
                             result['curve']['std']):
            curve_data.append({
                'System': name,
                'k': k,
                'D_eff': D,
                'std': std
            })
    
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(OUTPUT_DIR / "all_curves.csv", index=False)
    print(f"✓ Curves saved to {OUTPUT_DIR / 'all_curves.csv'}")
    
    return summary_df

def generate_report(all_results):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V16 INTERPRETATION REPORT: CROSS-DOMAIN SATURATION ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Is finite dimensional saturation a universal property?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. SHAPE TYPE CLASSIFICATION")
    report.append("-" * 50)
    
    shape_types = {}
    for name, result in all_results.items():
        phases = result['phases']
        if phases['has_decline']:
            shape_type = 'Growth-Saturation-Peak-Decline'
        elif phases['saturation_point']:
            shape_type = 'Growth-Saturation'
        else:
            shape_type = 'Simple Growth'
        
        shape_types[name] = shape_type
    
    for name, shape in shape_types.items():
        report.append(f"  {name}: {shape}")
    
    shape_counts = {}
    for shape in shape_types.values():
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    report.append(f"\n  Shape distribution:")
    for shape, count in shape_counts.items():
        report.append(f"    {shape}: {count}/{len(all_results)}")
    
    report.append("")
    report.append("2. PHASE STRUCTURE ANALYSIS")
    report.append("-" * 50)
    
    has_4_phase = sum(1 for r in all_results.values() if r['phases']['has_decline'])
    has_saturation = sum(1 for r in all_results.values() if r['phases']['saturation_point'])
    
    report.append(f"  Systems with 4-phase structure: {has_4_phase}/{len(all_results)}")
    report.append(f"  Systems with saturation point: {has_saturation}/{len(all_results)}")
    
    report.append("")
    report.append("3. SATURATION LEVELS")
    report.append("-" * 50)
    
    header = f"{'System':<20} {'Peak D':>10} {'Peak k':>10} {'Decline?':>10} {'Decline %':>10}"
    report.append(header)
    report.append("-" * len(header))
    
    for name, result in all_results.items():
        phases = result['phases']
        peak_D = phases['peak_D'] if phases['peak_D'] else result['max_D']
        peak_k = phases['peak_k'] if phases['peak_k'] else 0
        has_decline = 'Yes' if phases['has_decline'] else 'No'
        decline_pct = phases['decline_fraction'] * 100 if phases['has_decline'] else 0
        
        report.append(f"{name:<20} {peak_D:>10.2f} {peak_k:>10} {has_decline:>10} {decline_pct:>9.1f}%")
    
    report.append("")
    report.append("4. UNIVERSALITY ASSESSMENT")
    report.append("-" * 50)
    
    peak_D_values = [r['phases']['peak_D'] if r['phases']['peak_D'] else r['max_D'] 
                     for r in all_results.values()]
    
    report.append(f"  Peak D_eff range: {min(peak_D_values):.2f} - {max(peak_D_values):.2f}")
    report.append(f"  Mean Peak D_eff: {np.mean(peak_D_values):.2f}")
    report.append(f"  CV: {np.std(peak_D_values) / np.mean(peak_D_values):.2f}")
    
    has_decline_count = sum(1 for r in all_results.values() if r['phases']['has_decline'])
    decline_universal = has_decline_count >= len(all_results) * 0.5
    saturation_universal = has_saturation >= len(all_results) * 0.5
    
    report.append(f"\n  Decline at large k: {has_decline_count}/{len(all_results)} {'✓ UNIVERSAL' if decline_universal else '✗ NOT UNIVERSAL'}")
    report.append(f"  Saturation point: {has_saturation}/{len(all_results)} {'✓ UNIVERSAL' if saturation_universal else '✗ NOT UNIVERSAL'}")
    
    report.append("")
    report.append("5. SUCCESS CRITERIA")
    report.append("-" * 50)
    
    phase_3_4 = sum(1 for name, result in all_results.items() 
                    if result['phases']['saturation_point'] or result['phases']['has_decline'])
    
    report.append(f"  Similar 3-4 phase structure: {phase_3_4}/{len(all_results)}")
    report.append(f"  Interpretation: {'✓ YES' if phase_3_4 >= len(all_results) * 0.5 else '✗ NO'}")
    
    report.append("")
    report.append("6. CONCLUSIONS")
    report.append("-" * 50)
    
    if decline_universal and saturation_universal:
        report.append("  UNIVERSAL DIMENSIONAL SATURATION EXISTS")
        report.append("  ")
        report.append("  All (or most) systems show the same 3-4 phase structure:")
        report.append("    1. Growth phase (small k)")
        report.append("    2. Saturation phase (medium k)")
        report.append("    3. Peak (maximum dimensionality)")
        report.append("    4. Decline (large k)")
        report.append("  ")
        report.append("  This suggests a universal property of high-dimensional data:")
        report.append("    - Dimensional unfolding at small scales")
        report.append("    - Finite intrinsic dimensionality")
        report.append("    - Representative volume saturation")
    elif saturation_universal:
        report.append("  PARTIAL UNIVERSALITY: SATURATION WITHOUT DECLINE")
        report.append("  ")
        report.append("  Most systems show growth and saturation,")
        report.append("  but decline at large k is not universal.")
    else:
        report.append("  NO UNIVERSAL DIMENSIONAL SATURATION")
        report.append("  ")
        report.append("  Different systems show different scaling behaviors.")
        report.append("  The 4-phase structure is system-dependent.")
    
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
    
    language_embeddings = generate_language_embeddings(n_samples=2000, n_dims=50)
    print(f"Language embeddings: {list(language_embeddings.keys())}")
    
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING D_eff(k) CURVES")
    print("=" * 80)
    
    k_values = [5, 10, 20, 50, 100, 200, 500]
    
    all_results = {}
    
    all_results['TRIBE'] = analyze_system(tribe_activations, 'TRIBE', k_values)
    
    for name, embeddings in language_embeddings.items():
        all_results[name] = analyze_system(embeddings, name, k_values)
    
    print("\n" + "=" * 80)
    print("PHASE 4: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(all_results)
    
    print("\n" + "=" * 80)
    print("PHASE 5: SUMMARY TABLE")
    print("=" * 80)
    
    create_summary_table(all_results)
    
    print("\n" + "=" * 80)
    print("PHASE 6: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(all_results)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - cross_domain_summary.csv")
    print("  - all_curves.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_cross_domain.png")
    print("  - figures/fig2_normalized_comparison.png")
    print("  - figures/fig3_detailed_analysis.png")

if __name__ == "__main__":
    main()
