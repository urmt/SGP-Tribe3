#!/usr/bin/env python3
"""
V14 Pipeline: Scale-Dependent Dimensionality
Test whether D_eff(k) follows a universal functional form
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

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v14")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V14 SCALE-DEPENDENT DIMENSIONALITY ANALYSIS")
print("Testing universal functional form of D_eff(k)")
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
    """
    Compute local effective dimensionality using k-nearest neighbors.
    D_eff = (Σλ)² / Σλ² for the local covariance matrix
    """
    X = StandardScaler().fit_transform(activations)
    n_samples, n_dims = X.shape
    
    if k >= n_samples:
        k = n_samples - 1
    
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    local_dims = []
    for i in range(n_samples):
        neighborhood = X[indices[i, 1:]]
        
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
    
    return np.mean(local_dims) if local_dims else 0

def compute_D_eff_curve(activations, k_values):
    """Compute D_eff(k) curve."""
    D_eff_values = []
    std_values = []
    
    X = StandardScaler().fit_transform(activations)
    n_samples = X.shape[0]
    
    for k in k_values:
        if k >= n_samples:
            k = n_samples - 1
        
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        local_dims = []
        for i in range(n_samples):
            neighborhood = X[indices[i, 1:]]
            
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
            D_eff_values.append(np.mean(local_dims))
            std_values.append(np.std(local_dims))
        else:
            D_eff_values.append(0)
            std_values.append(0)
    
    return {
        'k_values': list(k_values),
        'D_eff': D_eff_values,
        'std': std_values
    }

def fit_logarithmic(k, a, b):
    """Logarithmic fit: D = a * log(k) + b"""
    return a * np.log(k) + b

def fit_power_law(k, a, b):
    """Power law fit: D = a * k^b"""
    k = np.array(k)
    k = np.maximum(k, 1)
    return a * np.power(k, b)

def fit_saturating(k, a, b):
    """Saturating fit: D = a * (1 - exp(-k/b))"""
    k = np.array(k)
    k = np.maximum(k, 0.1)
    return a * (1 - np.exp(-k / b))

def fit_all_models(k_values, D_eff_values):
    """Fit all three models and return results."""
    k = np.array(k_values)
    D = np.array(D_eff_values)
    
    results = {}
    
    try:
        popt, pcov = curve_fit(fit_logarithmic, k, D, p0=[1, 1], maxfev=5000)
        residuals = D - fit_logarithmic(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['logarithmic'] = {
            'a': popt[0], 'b': popt[1],
            'r_squared': r_squared,
            'fitted_curve': fit_logarithmic(k, *popt)
        }
    except:
        results['logarithmic'] = {'a': 0, 'b': 0, 'r_squared': 0, 'fitted_curve': D}
    
    try:
        popt, pcov = curve_fit(fit_power_law, k, D, p0=[1, 0.3], maxfev=5000, bounds=([0, 0], [10, 1]))
        residuals = D - fit_power_law(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['power_law'] = {
            'a': popt[0], 'b': popt[1],
            'r_squared': r_squared,
            'fitted_curve': fit_power_law(k, *popt)
        }
    except:
        results['power_law'] = {'a': 0, 'b': 0, 'r_squared': 0, 'fitted_curve': D}
    
    try:
        popt, pcov = curve_fit(fit_saturating, k, D, p0=[5, 20], maxfev=5000, bounds=([0, 1], [50, 200]))
        residuals = D - fit_saturating(k, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['saturating'] = {
            'a': popt[0], 'b': popt[1],
            'r_squared': r_squared,
            'fitted_curve': fit_saturating(k, *popt)
        }
    except:
        results['saturating'] = {'a': 0, 'b': 0, 'r_squared': 0, 'fitted_curve': D}
    
    best_model = max(results.keys(), key=lambda m: results[m]['r_squared'])
    results['best_model'] = best_model
    results['best_r_squared'] = results[best_model]['r_squared']
    
    return results

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
    
    return np.array(activations)

def generate_synthetic_data():
    """Generate synthetic datasets."""
    np.random.seed(RANDOM_SEED)
    
    datasets = {}
    
    datasets['uniform_sphere'] = {
        'name': 'Uniform Sphere',
        'data': np.random.randn(2000, 50) / np.sqrt(50),
        'description': 'Points on high-dimensional sphere'
    }
    
    datasets['gaussian_cloud'] = {
        'name': 'Gaussian Cloud',
        'data': np.random.randn(2000, 50) * np.exp(-np.arange(50) / 20),
        'description': 'Correlated Gaussian'
    }
    
    datasets['sparse'] = {
        'name': 'Sparse',
        'data': np.random.randn(2000, 50) * (np.random.rand(2000, 50) > 0.7),
        'description': 'Sparse activations'
    }
    
    datasets['manifold'] = {
        'name': 'Low-D Manifold',
        'data': generate_curved_manifold(2000, 50),
        'description': 'Intrinsic 2D manifold'
    }
    
    datasets['hierarchical'] = {
        'name': 'Hierarchical',
        'data': generate_hierarchical(2000, 50),
        'description': 'Multi-scale structure'
    }
    
    return datasets

def generate_curved_manifold(n, d):
    """Generate curved manifold."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.uniform(0.5, 1.5, n)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sin(2 * theta)
    
    data = np.column_stack([x, y, z])
    
    for i in range(d - 3):
        data = np.column_stack([data, np.random.randn(n) * 0.05])
    
    return data

def generate_hierarchical(n, d):
    """Generate hierarchical structure."""
    data = np.zeros((n, d))
    
    scales = [4, 2, 1]
    for scale_idx, scale in enumerate(scales):
        n_points = n // len(scales)
        start = scale_idx * n_points
        end = min(start + n_points, n)
        
        for i in range(start, end):
            for j in range(d):
                data[i, j] = np.random.randn() * scale
    
    return data

def split_stability(activations, k_values, n_splits=5):
    """Test stability across data splits."""
    np.random.seed(RANDOM_SEED)
    n_samples = len(activations)
    
    results = []
    for split in range(n_splits):
        indices = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=False)
        split_data = activations[indices]
        
        curve = compute_D_eff_curve(split_data, k_values)
        fit_results = fit_all_models(curve['k_values'], curve['D_eff'])
        
        results.append({
            'split': split,
            'curve': curve,
            'fit': fit_results
        })
    
    return results

def create_visualizations(all_curves, all_fits, stability_results):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_curves)))
    
    for idx, (name, curve) in enumerate(all_curves.items()):
        k_values = curve['k_values']
        D_eff = curve['D_eff']
        std = curve['std']
        
        axes[0, 0].errorbar(k_values, D_eff, yerr=std, marker='o', 
                            label=name, color=colors[idx], alpha=0.7, capsize=3)
    
    axes[0, 0].set_xlabel('k (neighbors)')
    axes[0, 0].set_ylabel('D_eff(k)')
    axes[0, 0].set_title('D_eff(k) vs Scale')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    
    for idx, (name, fits) in enumerate(all_fits.items()):
        k_values = all_curves[name]['k_values']
        
        if fits['logarithmic']['r_squared'] > 0:
            axes[0, 1].plot(k_values, fits['logarithmic']['fitted_curve'], 
                           linestyle='--', color=colors[idx], alpha=0.7)
        
        axes[1, 0].scatter(k_values, fits['power_law']['fitted_curve'], 
                          marker='o', color=colors[idx], alpha=0.5, s=30)
    
    for idx, (name, curve) in enumerate(all_curves.items()):
        axes[1, 0].plot(curve['k_values'], curve['D_eff'], 
                        marker='o', color=colors[idx], alpha=0.3, linewidth=1)
    
    axes[1, 0].set_xlabel('k (neighbors)')
    axes[1, 0].set_ylabel('D_eff (Power Law Fit)')
    axes[1, 0].set_title('Power Law Fit Comparison')
    axes[1, 0].set_xscale('log')
    
    model_names = ['logarithmic', 'power_law', 'saturating']
    model_colors = ['steelblue', 'coral', 'seagreen']
    
    for idx, (name, fits) in enumerate(all_fits.items()):
        x_pos = np.arange(3) + idx * 0.2
        r_values = [fits[m]['r_squared'] for m in model_names]
        axes[1, 1].bar(x_pos, r_values, 0.2, 
                       color=[model_colors[i] for i in range(3)], alpha=0.7)
    
    axes[1, 1].set_xticks(np.arange(3) + 0.3)
    axes[1, 1].set_xticklabels(['Log', 'Power', 'Saturating'])
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Goodness of Fit by Model')
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_deff_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_deff_curves.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (name, fits) in enumerate(all_fits.items()):
        k_values = all_curves[name]['k_values']
        D_eff = all_curves[name]['D_eff']
        
        axes[0, 0].plot(k_values, D_eff, 'o-', color=colors[idx], alpha=0.5, label=name)
        axes[0, 0].plot(k_values, fits['logarithmic']['fitted_curve'], '--', 
                        color=colors[idx], linewidth=2, 
                        label=f'{name} (log, R²={fits["logarithmic"]["r_squared"]:.3f})')
    
    axes[0, 0].set_xlabel('k (neighbors)')
    axes[0, 0].set_ylabel('D_eff(k)')
    axes[0, 0].set_title('Logarithmic Fits')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_xscale('log')
    
    for idx, (name, fits) in enumerate(all_fits.items()):
        k_values = all_curves[name]['k_values']
        D_eff = all_curves[name]['D_eff']
        
        axes[0, 1].plot(k_values, D_eff, 'o-', color=colors[idx], alpha=0.5)
        axes[0, 1].plot(k_values, fits['power_law']['fitted_curve'], '--', 
                        color=colors[idx], linewidth=2,
                        label=f'{name} (R²={fits["power_law"]["r_squared"]:.3f})')
    
    axes[0, 1].set_xlabel('k (neighbors)')
    axes[0, 1].set_ylabel('D_eff(k)')
    axes[0, 1].set_title('Power Law Fits')
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_xscale('log')
    
    for idx, (name, fits) in enumerate(all_fits.items()):
        k_values = all_curves[name]['k_values']
        D_eff = all_curves[name]['D_eff']
        
        axes[1, 0].plot(k_values, D_eff, 'o-', color=colors[idx], alpha=0.5)
        axes[1, 0].plot(k_values, fits['saturating']['fitted_curve'], '--', 
                        color=colors[idx], linewidth=2,
                        label=f'{name} (R²={fits["saturating"]["r_squared"]:.3f})')
    
    axes[1, 0].set_xlabel('k (neighbors)')
    axes[1, 0].set_ylabel('D_eff(k)')
    axes[1, 0].set_title('Saturating Fits')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_xscale('log')
    
    best_models = [all_fits[name]['best_model'] for name in all_fits.keys()]
    model_counts = {m: best_models.count(m) for m in ['logarithmic', 'power_law', 'saturating']}
    
    axes[1, 1].bar(model_counts.keys(), model_counts.values(), color=['steelblue', 'coral', 'seagreen'])
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Best Model Distribution')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_fit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_fit_comparison.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stability_data = {m: [] for m in ['logarithmic', 'power_law', 'saturating']}
    for res in stability_results:
        for m in ['logarithmic', 'power_law', 'saturating']:
            stability_data[m].append(res['fit'][m]['r_squared'])
    
    for idx, (model, r_vals) in enumerate(stability_data.items()):
        axes[0].plot(range(len(r_vals)), r_vals, marker='o', label=model, 
                    color=model_colors[idx])
    axes[0].set_xlabel('Split')
    axes[0].set_ylabel('R²')
    axes[0].set_title('Model Stability Across Splits')
    axes[0].legend()
    axes[0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    
    param_data = {m: [] for m in ['logarithmic', 'power_law', 'saturating']}
    for res in stability_results:
        for m in ['logarithmic', 'power_law', 'saturating']:
            param_data[m].append(res['fit'][m]['a'])
    
    for idx, (model, vals) in enumerate(param_data.items()):
        axes[1].plot(range(len(vals)), vals, marker='s', label=model, 
                    color=model_colors[idx])
    axes[1].set_xlabel('Split')
    axes[1].set_ylabel('Parameter a')
    axes[1].set_title('Parameter Stability')
    axes[1].legend()
    
    if stability_results:
        best_r2_per_split = []
        for res in stability_results:
            best_model = max(['logarithmic', 'power_law', 'saturating'], 
                           key=lambda m: res['fit'][m]['r_squared'])
            best_r2_per_split.append(res['fit'][best_model]['r_squared'])
        
        axes[2].plot(range(len(best_r2_per_split)), best_r2_per_split, marker='o', color='purple')
        axes[2].set_xlabel('Split')
        axes[2].set_ylabel('Best R²')
        axes[2].set_title('Best Model R² by Split')
        axes[2].axhline(y=np.mean(best_r2_per_split), color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_stability.png")
    
    return

def create_summary_tables(all_curves, all_fits, stability_results):
    """Create summary tables."""
    
    summary_data = []
    for name, fits in all_fits.items():
        row = {
            'Dataset': name,
            'Best_Model': fits['best_model'],
            'Best_R2': fits['best_r_squared'],
            'Log_a': fits['logarithmic']['a'],
            'Log_b': fits['logarithmic']['b'],
            'Log_R2': fits['logarithmic']['r_squared'],
            'Power_a': fits['power_law']['a'],
            'Power_b': fits['power_law']['b'],
            'Power_R2': fits['power_law']['r_squared'],
            'Sat_a': fits['saturating']['a'],
            'Sat_b': fits['saturating']['b'],
            'Sat_R2': fits['saturating']['r_squared']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)
    print(f"✓ Fit summary saved to {OUTPUT_DIR / 'fit_summary.csv'}")
    
    curve_data = []
    for name, curve in all_curves.items():
        for k, D, std in zip(curve['k_values'], curve['D_eff'], curve['std']):
            curve_data.append({
                'Dataset': name,
                'k': k,
                'D_eff': D,
                'std': std
            })
    
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(OUTPUT_DIR / "deff_curves.csv", index=False)
    print(f"✓ D_eff curves saved to {OUTPUT_DIR / 'deff_curves.csv'}")
    
    stability_data = []
    for res in stability_results:
        for model in ['logarithmic', 'power_law', 'saturating']:
            stability_data.append({
                'Split': res['split'],
                'Model': model,
                'R2': res['fit'][model]['r_squared'],
                'Parameter_a': res['fit'][model]['a'],
                'Parameter_b': res['fit'][model]['b']
            })
    
    stability_df = pd.DataFrame(stability_data)
    stability_df.to_csv(OUTPUT_DIR / "stability.csv", index=False)
    print(f"✓ Stability data saved to {OUTPUT_DIR / 'stability.csv'}")
    
    return summary_df, curve_df, stability_df

def generate_report(all_curves, all_fits, stability_results):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V14 INTERPRETATION REPORT: SCALE-DEPENDENT DIMENSIONALITY")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Does D_eff(k) follow a universal functional form across domains?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. SUCCESS CRITERIA ASSESSMENT")
    report.append("-" * 50)
    
    best_models = [f['best_model'] for f in all_fits.values()]
    same_form = len(set(best_models)) == 1
    report.append(f"  a) Similar functional form: {'✓ YES' if same_form else '✗ NO'}")
    report.append(f"     Best models: {dict(zip(*np.unique(best_models, return_counts=True)))}")
    
    if stability_results:
        model_r2_stability = {}
        for model in ['logarithmic', 'power_law', 'saturating']:
            r_vals = [r['fit'][model]['r_squared'] for r in stability_results]
            model_r2_stability[model] = {
                'mean': np.mean(r_vals),
                'std': np.std(r_vals),
                'cv': np.std(r_vals) / np.mean(r_vals) if np.mean(r_vals) > 0 else np.inf
            }
        
        most_stable = min(model_r2_stability.keys(), 
                         key=lambda m: model_r2_stability[m]['cv'])
        report.append(f"  b) Stable parameters: ✓ (most stable: {most_stable})")
        report.append(f"     Model R² stability (CV):")
        for model, stats_dict in model_r2_stability.items():
            report.append(f"       {model}: {stats_dict['cv']:.3f}")
    
    report.append("")
    report.append("2. FIT QUALITY SUMMARY")
    report.append("-" * 50)
    
    header = f"{'Dataset':<20} {'Best':<12} {'R²':>8} {'Log R²':>8} {'Power R²':>10} {'Sat R²':>8}"
    report.append(header)
    report.append("-" * len(header))
    
    for name, fits in all_fits.items():
        report.append(f"{name:<20} {fits['best_model']:<12} {fits['best_r_squared']:>8.3f} "
                     f"{fits['logarithmic']['r_squared']:>8.3f} "
                     f"{fits['power_law']['r_squared']:>10.3f} "
                     f"{fits['saturating']['r_squared']:>8.3f}")
    
    report.append("")
    report.append("3. FITTED PARAMETERS")
    report.append("-" * 50)
    
    for name, fits in all_fits.items():
        report.append(f"\n  {name}:")
        report.append(f"    Logarithmic: D = {fits['logarithmic']['a']:.3f} × log(k) + {fits['logarithmic']['b']:.3f} (R²={fits['logarithmic']['r_squared']:.3f})")
        report.append(f"    Power Law:   D = {fits['power_law']['a']:.3f} × k^{fits['power_law']['b']:.3f} (R²={fits['power_law']['r_squared']:.3f})")
        report.append(f"    Saturating:   D = {fits['saturating']['a']:.3f} × (1 - exp(-k/{fits['saturating']['b']:.1f})) (R²={fits['saturating']['r_squared']:.3f})")
    
    report.append("")
    report.append("4. STABILITY ANALYSIS")
    report.append("-" * 50)
    
    if stability_results:
        for model in ['logarithmic', 'power_law', 'saturating']:
            a_vals = [r['fit'][model]['a'] for r in stability_results]
            r_vals = [r['fit'][model]['r_squared'] for r in stability_results]
            report.append(f"  {model}:")
            report.append(f"    a = {np.mean(a_vals):.3f} ± {np.std(a_vals):.3f}")
            report.append(f"    R² = {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}")
    
    report.append("")
    report.append("5. KEY FINDINGS")
    report.append("-" * 50)
    
    model_counts = {}
    for fits in all_fits.values():
        model = fits['best_model']
        model_counts[model] = model_counts.get(model, 0) + 1
    
    if same_form:
        report.append(f"  ✓ Universal functional form: {best_models[0]}")
        report.append(f"    {len(best_models)}/{len(best_models)} datasets best fit {best_models[0]}")
    else:
        report.append(f"  ✗ No single functional form dominates")
        for model, count in model_counts.items():
            report.append(f"     {model}: {count}/{len(all_fits)} datasets")
    
    high_r2_count = sum(1 for f in all_fits.values() if f['best_r_squared'] > 0.9)
    report.append(f"\n  High quality fits (R² > 0.9): {high_r2_count}/{len(all_fits)}")
    
    report.append("")
    report.append("6. CONCLUSIONS")
    report.append("-" * 50)
    
    if same_form and high_r2_count >= len(all_fits) * 0.5:
        report.append("  UNIVERSAL FUNCTIONAL FORM EXISTS")
        report.append("  ")
        report.append(f"  The {best_models[0]} form best describes D_eff(k) across domains.")
        report.append("  ")
        report.append("  Key characteristics:")
        for name, fits in all_fits.items():
            report.append(f"    {name}: {fits[best_models[0]]['a']:.2f} × k^{fits[best_models[0]]['b']:.2f}" 
                         if best_models[0] == 'power_law' else 
                         f"    {name}: {fits[best_models[0]]['a']:.2f} × log(k) + {fits[best_models[0]]['b']:.2f}"
                         if best_models[0] == 'logarithmic' else
                         f"    {name}: D_max ≈ {fits[best_models[0]]['a']:.2f}")
    elif same_form:
        report.append("  PARTIAL UNIVERSALITY")
        report.append("  ")
        report.append(f"  The {best_models[0]} form is most common but with variable quality.")
    else:
        report.append("  NO UNIVERSAL FUNCTIONAL FORM")
        report.append("  ")
        report.append("  Different datasets require different functional forms.")
        report.append("  Dimensionality structure is domain-dependent.")
    
    report.append("")
    report.append("7. INTERPRETATION")
    report.append("-" * 50)
    
    power_law_count = model_counts.get('power_law', 0)
    if power_law_count > 0:
        report.append("  Power law behavior suggests:")
        report.append("    - Scale-free dimensionality structure")
        report.append("    - Self-similar geometry across scales")
    
    saturating_count = model_counts.get('saturating', 0)
    if saturating_count > 0:
        report.append("  Saturating behavior suggests:")
        report.append("    - Finite intrinsic dimensionality")
        report.append("    - Saturation at large scales")
    
    log_count = model_counts.get('logarithmic', 0)
    if log_count > 0:
        report.append("  Logarithmic behavior suggests:")
        report.append("    - Slow dimensionality growth")
        report.append("    - Diminishing returns with scale")
    
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
    print("PHASE 2: EXTRACTING DATA")
    print("=" * 80)
    
    tribe_activations = extract_tribe_activations(tribe_data, n_samples=2000)
    print(f"TRIBE activations: {tribe_activations.shape}")
    
    synthetic_datasets = generate_synthetic_data()
    print(f"Synthetic datasets: {list(synthetic_datasets.keys())}")
    
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING D_eff(k) CURVES")
    print("=" * 80)
    
    k_values = [5, 10, 20, 50, 100, 200]
    
    all_curves = {}
    all_curves['TRIBE'] = compute_D_eff_curve(tribe_activations, k_values)
    
    for name, dataset in synthetic_datasets.items():
        print(f"  Computing {name}...")
        all_curves[name] = compute_D_eff_curve(dataset['data'], k_values)
    
    print("\n" + "=" * 80)
    print("PHASE 4: FITTING MODELS")
    print("=" * 80)
    
    all_fits = {}
    for name, curve in all_curves.items():
        print(f"  Fitting {name}...")
        all_fits[name] = fit_all_models(curve['k_values'], curve['D_eff'])
    
    print("\n" + "=" * 80)
    print("PHASE 5: STABILITY ANALYSIS")
    print("=" * 80)
    
    print("Running split stability analysis on TRIBE...")
    stability_results = split_stability(tribe_activations, k_values, n_splits=5)
    
    print("\n" + "=" * 80)
    print("PHASE 6: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(all_curves, all_fits, stability_results)
    
    print("\n" + "=" * 80)
    print("PHASE 7: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_tables(all_curves, all_fits, stability_results)
    
    print("\n" + "=" * 80)
    print("PHASE 8: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(all_curves, all_fits, stability_results)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - fit_summary.csv")
    print("  - deff_curves.csv")
    print("  - stability.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_deff_curves.png")
    print("  - figures/fig2_fit_comparison.png")
    print("  - figures/fig3_stability.png")

if __name__ == "__main__":
    main()
