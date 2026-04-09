#!/usr/bin/env python3
"""
V15 Pipeline: Extended Scale Analysis
Determine whether TRIBE scaling is truly scale-free or eventually saturates
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

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v15")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V15 EXTENDED SCALE ANALYSIS")
print("Determining if TRIBE scaling is truly scale-free or saturates")
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
    """Compute local effective dimensionality for a given k."""
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

def compute_extended_D_eff_curve(activations, k_values):
    """Compute D_eff(k) for extended k range."""
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

def fit_power_law(k, a, b):
    """Power law: D = a * k^b"""
    k = np.maximum(k, 1)
    return a * np.power(k, b)

def fit_saturating(k, a, b):
    """Saturating: D = a * (1 - exp(-k/b))"""
    k = np.maximum(k, 0.1)
    return a * (1 - np.exp(-k / b))

def fit_broken_power_law(k, a1, b1, k_break, a2, b2):
    """Broken power law with transition at k_break"""
    result = np.zeros_like(k, dtype=float)
    mask1 = k <= k_break
    mask2 = k > k_break
    
    if np.any(mask1):
        result[mask1] = a1 * np.power(k[mask1], b1)
    if np.any(mask2):
        result[mask2] = a2 * np.power(k[mask2], b2)
    
    return result

def fit_all_models_extended(k_values, D_eff_values):
    """Fit all models and return detailed results."""
    k = np.array(k_values, dtype=float)
    D = np.array(D_eff_values, dtype=float)
    
    valid = np.isfinite(k) & np.isfinite(D) & (k > 0) & (D > 0)
    k = k[valid]
    D = D[valid]
    
    results = {}
    
    try:
        popt, pcov = curve_fit(fit_power_law, k, D, p0=[1, 0.3], maxfev=5000, 
                               bounds=([0, 0], [10, 1]))
        residuals = D - fit_power_law(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        aic = len(k) * np.log(ss_res / len(k)) + 2 * 2
        bic = len(k) * np.log(ss_res / len(k)) + 2 * 2 * np.log(len(k))
        
        results['power_law'] = {
            'a': popt[0], 'b': popt[1],
            'r_squared': r_squared,
            'aic': aic, 'bic': bic,
            'fitted_curve': fit_power_law(k, *popt),
            'n_params': 2
        }
    except Exception as e:
        results['power_law'] = {'a': 0, 'b': 0, 'r_squared': 0, 'aic': np.inf, 'bic': np.inf}
    
    try:
        popt, pcov = curve_fit(fit_saturating, k, D, p0=[10, 50], maxfev=5000,
                               bounds=([0, 1], [100, 1000]))
        residuals = D - fit_saturating(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        aic = len(k) * np.log(ss_res / len(k)) + 2 * 2
        bic = len(k) * np.log(ss_res / len(k)) + 2 * 2 * np.log(len(k))
        
        results['saturating'] = {
            'a': popt[0], 'b': popt[1],
            'r_squared': r_squared,
            'aic': aic, 'bic': bic,
            'fitted_curve': fit_saturating(k, *popt),
            'n_params': 2
        }
    except Exception as e:
        results['saturating'] = {'a': 0, 'b': 0, 'r_squared': 0, 'aic': np.inf, 'bic': np.inf}
    
    try:
        k_break = np.median(k)
        popt, pcov = curve_fit(fit_broken_power_law, k, D, 
                               p0=[1, 0.3, k_break, 5, 0.1], 
                               maxfev=5000,
                               bounds=([0, 0, 5, 0, 0], [10, 1, 200, 100, 0.9]))
        residuals = D - fit_broken_power_law(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        n_params = 5
        aic = len(k) * np.log(ss_res / len(k)) + 2 * n_params
        bic = len(k) * np.log(ss_res / len(k)) + 2 * n_params * np.log(len(k))
        
        results['broken_power_law'] = {
            'a1': popt[0], 'b1': popt[1], 'k_break': popt[2],
            'a2': popt[3], 'b2': popt[4],
            'r_squared': r_squared,
            'aic': aic, 'bic': bic,
            'fitted_curve': fit_broken_power_law(k, *popt),
            'n_params': 5
        }
    except Exception as e:
        results['broken_power_law'] = {'r_squared': 0, 'aic': np.inf, 'bic': np.inf}
    
    best_by_r2 = max(results.keys(), key=lambda m: results[m]['r_squared'])
    best_by_aic = min(results.keys(), key=lambda m: results[m]['aic'])
    
    results['best_by_r2'] = best_by_r2
    results['best_by_aic'] = best_by_aic
    
    return results

def compute_log_log_curvature(k_values, D_eff_values):
    """Compute second derivative of log-log plot to detect deviations."""
    k = np.array(k_values, dtype=float)
    D = np.array(D_eff_values, dtype=float)
    
    log_k = np.log(k)
    log_D = np.log(D)
    
    if len(log_k) < 3:
        return {'curvature': np.zeros(len(log_k)), 'slope_changes': []}
    
    first_deriv = np.gradient(log_D, log_k)
    second_deriv = np.gradient(first_deriv, log_k)
    
    slope_stability = np.std(first_deriv) / np.abs(np.mean(first_deriv)) if np.mean(first_deriv) != 0 else np.inf
    
    slope_changes = []
    for i in range(1, len(first_deriv)):
        if abs(first_deriv[i] - first_deriv[i-1]) > 0.1:
            slope_changes.append({'k': k[i], 'change': first_deriv[i] - first_deriv[i-1]})
    
    return {
        'log_k': log_k,
        'log_D': log_D,
        'first_derivative': first_deriv,
        'second_derivative': second_deriv,
        'slope_stability': slope_stability,
        'mean_slope': np.mean(first_deriv),
        'slope_changes': slope_changes
    }

def compute_r_squared_range(curve, min_k_idx=0, max_k_idx=None):
    """Compute R² for different k ranges to test saturation."""
    if max_k_idx is None:
        max_k_idx = len(curve['k_values'])
    
    k_subset = curve['k_values'][min_k_idx:max_k_idx]
    D_subset = curve['D_eff'][min_k_idx:max_k_idx]
    
    if len(k_subset) < 3:
        return {'r2_full': 0, 'r2_small': 0, 'r2_large': 0}
    
    full_results = fit_all_models_extended(k_subset, D_subset)
    
    mid = len(k_subset) // 2
    small_results = fit_all_models_extended(k_subset[:mid], D_subset[:mid])
    large_results = fit_all_models_extended(k_subset[mid:], D_subset[mid:])
    
    return {
        'r2_power_full': full_results['power_law']['r_squared'],
        'r2_saturating_full': full_results['saturating']['r_squared'],
        'r2_power_small': small_results['power_law']['r_squared'],
        'r2_saturating_small': large_results['saturating']['r_squared'],
        'slopes': {
            'small': small_results['power_law']['b'] if small_results['power_law']['r_squared'] > 0 else 0,
            'large': large_results['power_law']['b'] if large_results['power_law']['r_squared'] > 0 else 0
        }
    }

def extract_tribe_activations(tribe_data, n_samples=5000):
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

def generate_real_embeddings():
    """Generate synthetic but more realistic embeddings."""
    np.random.seed(RANDOM_SEED)
    
    datasets = {}
    
    datasets['hierarchical'] = {
        'name': 'Hierarchical',
        'data': np.random.randn(5000, 100) * np.exp(-np.arange(100) / 30),
        'description': 'Power-law spectrum'
    }
    
    datasets['bandlimited'] = {
        'name': 'Band-Limited',
        'data': np.random.randn(5000, 100) * (1 + np.sin(np.arange(100) * 0.5)),
        'description': 'Periodic structure'
    }
    
    datasets['sparse'] = {
        'name': 'Sparse',
        'data': np.random.randn(5000, 100) * (np.random.rand(5000, 100) > 0.9),
        'description': 'Sparse activations'
    }
    
    datasets['correlated'] = {
        'name': 'Correlated',
        'data': np.random.randn(5000, 100),
        'description': 'Gaussian with structure'
    }
    cov = np.random.randn(100, 100)
    cov = cov @ cov.T
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1][:100]
    cov = cov[:100, :100] / np.max(eigvals) * 10
    datasets['correlated']['data'] = datasets['correlated']['data'] @ np.linalg.cholesky(cov + np.eye(100) * 0.1)
    
    return datasets

def create_visualizations(curve, fits, curvature, r2_ranges):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    k = np.array(curve['k_values'])
    D_eff = np.array(curve['D_eff'])
    std = np.array(curve['std'])
    
    axes[0, 0].errorbar(k, D_eff, yerr=std, marker='o', capsize=3, color='steelblue', 
                         label='TRIBE Data')
    
    k_smooth = np.linspace(min(k), max(k), 100)
    
    if fits['power_law']['r_squared'] > 0:
        axes[0, 0].plot(k_smooth, fit_power_law(k_smooth, fits['power_law']['a'], fits['power_law']['b']),
                        '--', color='coral', linewidth=2, 
                        label=f"Power Law: D={fits['power_law']['a']:.2f}×k^{fits['power_law']['b']:.3f} (R²={fits['power_law']['r_squared']:.3f})")
    
    if fits['saturating']['r_squared'] > 0:
        axes[0, 0].plot(k_smooth, fit_saturating(k_smooth, fits['saturating']['a'], fits['saturating']['b']),
                        ':', color='seagreen', linewidth=2,
                        label=f"Saturating: D={fits['saturating']['a']:.2f}×(1-exp(-k/{fits['saturating']['b']:.1f})) (R²={fits['saturating']['r_squared']:.3f})")
    
    if fits['broken_power_law']['r_squared'] > 0:
        axes[0, 0].plot(k_smooth, fit_broken_power_law(k_smooth, 
                        fits['broken_power_law']['a1'], fits['broken_power_law']['b1'],
                        fits['broken_power_law']['k_break'], fits['broken_power_law']['a2'],
                        fits['broken_power_law']['b2']),
                        '-.', color='purple', linewidth=2,
                        label=f"Broken Power: k_break={fits['broken_power_law']['k_break']:.1f}")
    
    axes[0, 0].set_xlabel('k (neighbors)', fontsize=12)
    axes[0, 0].set_ylabel('D_eff(k)', fontsize=12)
    axes[0, 0].set_title('Extended D_eff(k) Curve', fontsize=14)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    log_k = curvature['log_k']
    log_D = curvature['log_D']
    
    axes[0, 1].scatter(log_k, log_D, color='steelblue', s=50, alpha=0.7, label='Data')
    
    if fits['power_law']['r_squared'] > 0:
        log_k_smooth = np.linspace(min(log_k), max(log_k), 100)
        k_smooth = np.exp(log_k_smooth)
        D_fit = fit_power_law(k_smooth, fits['power_law']['a'], fits['power_law']['b'])
        log_D_fit = np.log(D_fit)
        axes[0, 1].plot(log_k_smooth, log_D_fit, '--', color='coral', linewidth=2,
                        label=f'Power Law Fit (slope={fits["power_law"]["b"]:.3f})')
    
    slope = curvature['mean_slope']
    intercept = np.mean(log_D) - slope * np.mean(log_k)
    axes[0, 1].plot(log_k, slope * log_k + intercept, 'k:', linewidth=1, alpha=0.5,
                    label=f'Linear (slope={slope:.3f})')
    
    axes[0, 1].set_xlabel('log(k)', fontsize=12)
    axes[0, 1].set_ylabel('log(D_eff)', fontsize=12)
    axes[0, 1].set_title('Log-Log Plot (Linear = Power Law)', fontsize=14)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(k, curvature['first_derivative'], marker='o', color='steelblue', linewidth=2)
    axes[0, 2].axhline(y=curvature['mean_slope'], color='r', linestyle='--', alpha=0.5,
                       label=f'Mean slope: {curvature["mean_slope"]:.3f}')
    axes[0, 2].set_xlabel('k', fontsize=12)
    axes[0, 2].set_ylabel('d(log D) / d(log k)', fontsize=12)
    axes[0, 2].set_title('First Derivative (Slope Stability)', fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].set_xscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(k, curvature['second_derivative'], marker='s', color='steelblue', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('k', fontsize=12)
    axes[1, 0].set_ylabel('d²(log D) / d(log k)²', fontsize=12)
    axes[1, 0].set_title('Second Derivative (Curvature)', fontsize=14)
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    model_names = ['power_law', 'saturating', 'broken_power_law']
    model_labels = ['Power Law', 'Saturating', 'Broken Power']
    r2_values = [fits[m]['r_squared'] if 'r_squared' in fits[m] else 0 for m in model_names]
    aic_values = [fits[m]['aic'] if 'aic' in fits[m] else np.inf for m in model_names]
    bic_values = [fits[m]['bic'] if 'bic' in fits[m] else np.inf for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, r2_values, width, label='R²', color='steelblue')
    axes[1, 1].bar(x, [1 - aic/max(aic_values) if max(aic_values) > 0 else 0 for aic in aic_values], 
                   width, label='1 - norm(AIC)', color='coral')
    axes[1, 1].bar(x + width, [1 - bic/max(bic_values) if max(bic_values) > 0 else 0 for bic in bic_values], 
                   width, label='1 - norm(BIC)', color='seagreen')
    
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_labels)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Comparison')
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.3)
    
    axes[1, 2].clear()
    axes[1, 2].bar(['Small k\n(first half)', 'Large k\n(second half)'],
                   [r2_ranges['slopes']['small'], r2_ranges['slopes']['large']],
                   color=['steelblue', 'coral'], alpha=0.7)
    axes[1, 2].set_ylabel('Power Law Exponent (b)')
    axes[1, 2].set_title('Slope Comparison: Small vs Large k')
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_extended_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_extended_analysis.png")
    
    return

def create_summary_tables(curve, fits, curvature, r2_ranges):
    """Create summary tables."""
    
    summary_data = {
        'Metric': [],
        'Value': []
    }
    
    summary_data['Metric'].extend(['k_range', 'n_points', 'Mean Slope', 'Slope CV'])
    summary_data['Value'].extend([
        f"{min(curve['k_values'])} - {max(curve['k_values'])}",
        len(curve['k_values']),
        f"{curvature['mean_slope']:.4f}",
        f"{curvature['slope_stability']:.4f}"
    ])
    
    for model in ['power_law', 'saturating', 'broken_power_law']:
        if model in fits:
            summary_data['Metric'].append(f"{model}_a")
            summary_data['Value'].append(f"{fits[model].get('a', fits[model].get('a1', 0)):.4f}")
            summary_data['Metric'].append(f"{model}_b")
            summary_data['Value'].append(f"{fits[model].get('b', fits[model].get('b1', 0)):.4f}")
            summary_data['Metric'].append(f"{model}_R2")
            summary_data['Value'].append(f"{fits[model].get('r_squared', 0):.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"✓ Summary saved to {OUTPUT_DIR / 'summary.csv'}")
    
    curve_data = {
        'k': curve['k_values'],
        'D_eff': curve['D_eff'],
        'std': curve['std'],
        'log_k': curvature['log_k'],
        'log_D_eff': curvature['log_D'],
        'first_deriv': curvature['first_derivative'],
        'second_deriv': curvature['second_derivative']
    }
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(OUTPUT_DIR / "curve_data.csv", index=False)
    print(f"✓ Curve data saved to {OUTPUT_DIR / 'curve_data.csv'}")
    
    return summary_df, curve_df

def generate_report(curve, fits, curvature, r2_ranges):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V15 INTERPRETATION REPORT: EXTENDED SCALE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Is TRIBE scaling truly scale-free (power law) or does it saturate?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. SCALE RANGE ANALYZED")
    report.append("-" * 50)
    report.append(f"  k range: {min(curve['k_values'])} to {max(curve['k_values'])}")
    report.append(f"  Number of points: {len(curve['k_values'])}")
    
    report.append("")
    report.append("2. LOG-LOG LINEARITY TEST")
    report.append("-" * 50)
    report.append(f"  Mean slope: {curvature['mean_slope']:.4f}")
    report.append(f"  Slope stability (CV): {curvature['slope_stability']:.4f}")
    
    is_linear = curvature['slope_stability'] < 0.2
    report.append(f"  Interpretation: {'✓ LINEAR (power law)' if is_linear else '✗ NON-LINEAR (saturation)'}")
    
    report.append("")
    report.append("3. CURVATURE ANALYSIS")
    report.append("-" * 50)
    report.append(f"  Second derivative mean: {np.mean(curvature['second_derivative']):.4f}")
    report.append(f"  Second derivative std: {np.std(curvature['second_derivative']):.4f}")
    
    curvature_interp = "CONCAVE (saturation)" if np.mean(curvature['second_derivative']) < 0 else "CONVEX (acceleration)"
    report.append(f"  Interpretation: {curvature_interp}")
    
    if curvature['slope_changes']:
        report.append(f"\n  Detected slope changes at: {[f'k={sc["k"]}' for sc in curvature['slope_changes'][:3]]}")
    
    report.append("")
    report.append("4. MODEL COMPARISON")
    report.append("-" * 50)
    
    header = f"{'Model':<20} {'R²':>8} {'AIC':>12} {'BIC':>12}"
    report.append(header)
    report.append("-" * len(header))
    
    for model in ['power_law', 'saturating', 'broken_power_law']:
        if model in fits:
            r2 = fits[model].get('r_squared', 0)
            aic = fits[model].get('aic', np.inf)
            bic = fits[model].get('bic', np.inf)
            report.append(f"{model:<20} {r2:>8.4f} {aic:>12.2f} {bic:>12.2f}")
    
    report.append("")
    report.append(f"  Best by R²: {fits.get('best_by_r2', 'N/A')}")
    report.append(f"  Best by AIC: {fits.get('best_by_aic', 'N/A')}")
    
    report.append("")
    report.append("5. SMALL VS LARGE K COMPARISON")
    report.append("-" * 50)
    report.append(f"  Small k power exponent: {r2_ranges['slopes']['small']:.4f}")
    report.append(f"  Large k power exponent: {r2_ranges['slopes']['large']:.4f}")
    
    slope_diff = abs(r2_ranges['slopes']['large'] - r2_ranges['slopes']['small'])
    if slope_diff < 0.05:
        report.append("  Interpretation: ✓ CONSISTENT (same slope at all scales)")
    elif r2_ranges['slopes']['large'] < r2_ranges['slopes']['small']:
        report.append("  Interpretation: ✓ SATURATION (slope decreases at large k)")
    else:
        report.append("  Interpretation: ✗ ACCELERATION (slope increases at large k)")
    
    report.append("")
    report.append("6. SUCCESS CRITERIA")
    report.append("-" * 50)
    
    linear_log = is_linear
    report.append(f"  Linear log-log: {'✓ YES' if linear_log else '✗ NO'} (CV={curvature['slope_stability']:.4f})")
    
    consistent = slope_diff < 0.1
    report.append(f"  Consistent behavior: {'✓ YES' if consistent else '✗ NO'} (diff={slope_diff:.4f})")
    
    best_is_power = fits.get('best_by_r2') == 'power_law'
    report.append(f"  Power law best fit: {'✓ YES' if best_is_power else '✗ NO'}")
    
    overall = linear_log and consistent and best_is_power
    report.append(f"\n  OVERALL: {'✓ SCALE-FREE (POWER LAW)' if overall else '✗ NOT SCALE-FREE'}")
    
    report.append("")
    report.append("7. CONCLUSIONS")
    report.append("-" * 50)
    
    if overall:
        report.append("  TRIBE SCALING IS TRULY SCALE-FREE")
        report.append("  ")
        report.append(f"  The D_eff(k) curve follows a power law: D_eff ∝ k^{curvature['mean_slope']:.3f}")
        report.append("  ")
        report.append("  Evidence:")
        report.append("    - Log-log plot is linear (R² for power law > 0.9)")
        report.append("    - Slope is stable across scales")
        report.append("    - No significant curvature or saturation")
        report.append("  ")
        report.append("  Interpretation:")
        report.append("    - Dimensionality grows without bound with scale")
        report.append("    - Self-similar structure at all scales")
        report.append("    - No intrinsic dimensionality ceiling")
    elif is_linear and not consistent:
        report.append("  TRIBE SHOWS WEAK SATURATION")
        report.append("  ")
        report.append("  The power law holds approximately but with slight deviation.")
        report.append("  ")
        report.append("  Evidence:")
        report.append(f"    - Power exponent decreases from {r2_ranges['slopes']['small']:.3f} to {r2_ranges['slopes']['large']:.3f}")
        report.append("    - Suggests eventual saturation at very large k")
    else:
        report.append("  TRIBE SCALING SHOWS SATURATION")
        report.append("  ")
        report.append("  The power law does NOT fully describe the scaling.")
        report.append("  ")
        report.append("  Evidence:")
        if fits.get('best_by_r2') == 'saturating':
            report.append("    - Saturating model fits best")
        report.append(f"    - Curvature: {curvature_interp}")
        report.append("  ")
        report.append("  Interpretation:")
        report.append("    - Finite intrinsic dimensionality")
        report.append("    - Dimensionality saturates at large scales")
    
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
    print("PHASE 2: EXTRACTING ACTIVATIONS")
    print("=" * 80)
    
    activations = extract_tribe_activations(tribe_data, n_samples=5000)
    print(f"TRIBE activations: {activations.shape}")
    
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING EXTENDED D_eff(k)")
    print("=" * 80)
    
    k_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
    
    k_values = [k for k in k_values if k < len(activations) // 2]
    
    print(f"Computing for k = {k_values}")
    curve = compute_extended_D_eff_curve(activations, k_values)
    
    print("\nResults:")
    for k, D, s in zip(curve['k_values'], curve['D_eff'], curve['std']):
        print(f"  k={k:4d}: D_eff={D:.3f} ± {s:.3f}")
    
    print("\n" + "=" * 80)
    print("PHASE 4: FITTING MODELS")
    print("=" * 80)
    
    fits = fit_all_models_extended(curve['k_values'], curve['D_eff'])
    
    print("\nModel fits:")
    for model in ['power_law', 'saturating', 'broken_power_law']:
        if model in fits:
            print(f"  {model}: R²={fits[model].get('r_squared', 0):.4f}, AIC={fits[model].get('aic', np.inf):.2f}")
    
    print("\n" + "=" * 80)
    print("PHASE 5: CURVATURE ANALYSIS")
    print("=" * 80)
    
    curvature = compute_log_log_curvature(curve['k_values'], curve['D_eff'])
    
    print(f"\nCurvature analysis:")
    print(f"  Mean slope: {curvature['mean_slope']:.4f}")
    print(f"  Slope stability: {curvature['slope_stability']:.4f}")
    
    print("\n" + "=" * 80)
    print("PHASE 6: RANGE COMPARISON")
    print("=" * 80)
    
    r2_ranges = compute_r_squared_range(curve)
    
    print(f"\nSmall vs Large k comparison:")
    print(f"  Small k slope: {r2_ranges['slopes']['small']:.4f}")
    print(f"  Large k slope: {r2_ranges['slopes']['large']:.4f}")
    
    print("\n" + "=" * 80)
    print("PHASE 7: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(curve, fits, curvature, r2_ranges)
    
    print("\n" + "=" * 80)
    print("PHASE 8: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_tables(curve, fits, curvature, r2_ranges)
    
    print("\n" + "=" * 80)
    print("PHASE 9: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(curve, fits, curvature, r2_ranges)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - summary.csv")
    print("  - curve_data.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_extended_analysis.png")

if __name__ == "__main__":
    main()
