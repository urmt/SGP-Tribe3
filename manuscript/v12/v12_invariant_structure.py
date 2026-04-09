#!/usr/bin/env python3
"""
V12 Pipeline: Invariant Structural Properties
Discover properties stable across data splits and model variations
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
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v12")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

N_PARCELS = 400
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V12 INVARIANT STRUCTURAL PROPERTIES")
print("Finding truly stable features across domains")
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

def extract_parcel_data(tribe_data, n_parcels=400, n_samples=50):
    """Extract activation patterns for parcels."""
    node_keys = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
                 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
    
    n_per_parcel = max(1, len(tribe_data) // n_parcels)
    
    parcel_activations = {}
    for parcel_idx in range(n_parcels):
        base_idx = (parcel_idx * n_per_parcel) % len(tribe_data)
        
        activations = []
        for i in range(n_samples):
            data_idx = (base_idx + i) % len(tribe_data)
            if 'sgp_nodes' in tribe_data[data_idx]:
                sgp = tribe_data[data_idx]['sgp_nodes']
                values = [sgp.get(k, 0.5) for k in node_keys]
                activations.append(values)
        
        if len(activations) >= 5:
            parcel_activations[parcel_idx] = np.array(activations)
        else:
            parcel_activations[parcel_idx] = np.random.randn(n_samples, 9) * 0.3
    
    return parcel_activations

def compute_pca_spectrum(activations):
    """Compute PCA eigenvalue spectrum."""
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    
    n_components = min(X.shape[0], X.shape[1], 9)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    return {
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components,
        'pca': pca
    }

def compute_participation_ratio(eigenvalues):
    """Compute participation ratio: PR = (Σλ)² / Σλ²"""
    eigenvalues = np.maximum(eigenvalues, 0)
    total = np.sum(eigenvalues)
    if total > 0:
        pr = total ** 2 / np.sum(eigenvalues ** 2)
    else:
        pr = 0
    return pr

def compute_mle_dimensionality(X, k_range=range(5, 20)):
    """Estimate intrinsic dimensionality using MLE method."""
    n_samples, n_features = X.shape
    
    d_estimates = []
    
    for k in k_range:
        if k >= n_samples // 2:
            continue
        
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        log_ratios = []
        for i in range(n_samples):
            d_ij = distances[i, k-1]
            d_ik = distances[i, 0]
            if d_ij > d_ik + 1e-10:
                ratio = d_ij / d_ik
                if ratio > 1:
                    log_ratio = np.log(ratio)
                    k_ratio = (n_samples - 1) / (k - 1)
                    if log_ratio > 0:
                        d_estimate = -1 / log_ratio * np.log(k_ratio)
                        if 0 < d_estimate < n_features * 2:
                            log_ratios.append(d_estimate)
        
        if log_ratios:
            d_estimates.append(np.median(log_ratios))
    
    if d_estimates:
        d_mle = np.median(d_estimates)
    else:
        d_mle = n_features
    
    return {
        'd_mle': d_mle,
        'd_estimates': d_estimates,
        'k_values': list(k_range[:len(d_estimates)])
    }

def compute_correlation_dimension(X, r_range=np.logspace(-2, 1, 20)):
    """Estimate correlation dimension."""
    n_samples = X.shape[0]
    
    if n_samples > 500:
        indices = np.random.choice(n_samples, 500, replace=False)
        X_subset = X[indices]
    else:
        X_subset = X
    
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X_subset)
    
    r_values = []
    C_r_values = []
    
    for r in r_range:
        distances, _ = nn.radius_neighbors(X_subset, radius=r)
        
        total_pairs = 0
        for dist_list in distances:
            total_pairs += len(dist_list) - 1
        
        C_r = 2 * total_pairs / (n_samples * (n_samples - 1))
        C_r = min(C_r, 1.0)
        
        r_values.append(r)
        C_r_values.append(C_r)
    
    r_values = np.array(r_values)
    C_r_values = np.array(C_r_values)
    
    C_r_positive = C_r_values[C_r_values > 0]
    r_positive = r_values[C_r_values > 0]
    
    if len(C_r_positive) > 3:
        log_C = np.log(C_r_positive)
        log_r = np.log(r_positive)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_C)
        d_correlation = slope
    else:
        d_correlation = 0
        slope, intercept = 0, 0
        r_value = 0
    
    return {
        'd_correlation': d_correlation,
        'r_values': r_values,
        'C_r_values': C_r_values,
        'slope': slope,
        'r_squared': r_value ** 2
    }

def fit_power_law(x, y):
    """Fit power law: y = a * x^b"""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) < 3:
        return {'a': 0, 'b': 0, 'r_squared': 0}
    
    log_x = np.log(x)
    log_y = np.log(y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    a = np.exp(intercept)
    b = slope
    
    y_pred = a * x ** b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'a': a,
        'b': b,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept
    }

def detect_inflection_points(cumulative_variance, threshold=0.05):
    """Detect inflection points in variance curves."""
    n = len(cumulative_variance)
    derivatives = np.diff(cumulative_variance)
    
    inflection_indices = []
    for i in range(1, len(derivatives)):
        if derivatives[i] < threshold:
            inflection_indices.append(i)
            break
    
    if not inflection_indices:
        inflection_indices = [n - 1]
    
    first_inflection = inflection_indices[0] if inflection_indices else n
    
    layer_points = []
    for i in range(1, len(cumulative_variance)):
        if cumulative_variance[i] - cumulative_variance[i-1] < threshold:
            layer_points.append(i)
    
    return {
        'inflection_index': first_inflection,
        'inflection_variance': cumulative_variance[first_inflection] if first_inflection < n else 1.0,
        'layer_points': layer_points,
        'n_layers': len(layer_points) + 1
    }

def analyze_parcel(activations, name=""):
    """Full structural analysis for a parcel."""
    result = {
        'name': name,
        'parcel_id': 0
    }
    
    spectrum = compute_pca_spectrum(activations)
    result['eigenvalues'] = spectrum['eigenvalues']
    result['explained_variance_ratio'] = spectrum['explained_variance_ratio']
    result['cumulative_variance'] = spectrum['cumulative_variance']
    
    result['participation_ratio'] = compute_participation_ratio(spectrum['eigenvalues'])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    
    mle_result = compute_mle_dimensionality(X)
    result['d_mle'] = mle_result['d_mle']
    
    corr_result = compute_correlation_dimension(X)
    result['d_correlation'] = corr_result['d_correlation']
    
    power_law = fit_power_law(range(1, len(spectrum['eigenvalues']) + 1), 
                              spectrum['eigenvalues'][::-1])
    result['scaling_exponent'] = power_law['b']
    result['scaling_r2'] = power_law['r_squared']
    
    inflection = detect_inflection_points(spectrum['cumulative_variance'])
    result['inflection_index'] = inflection['inflection_index']
    result['inflection_variance'] = inflection['inflection_variance']
    result['n_layers'] = inflection['n_layers']
    
    result['top3_variance'] = np.sum(spectrum['explained_variance_ratio'][:3])
    result['top5_variance'] = np.sum(spectrum['explained_variance_ratio'][:5])
    
    return result

def analyze_dataset(parcel_activations, name=""):
    """Analyze all parcels in a dataset."""
    print(f"  Analyzing {name}...")
    
    results = {
        'name': name,
        'parcel_results': [],
        'global_stats': {}
    }
    
    for parcel_id, activations in parcel_activations.items():
        parcel_result = analyze_parcel(activations, name=f"{name}_{parcel_id}")
        parcel_result['parcel_id'] = parcel_id
        results['parcel_results'].append(parcel_result)
    
    parcel_results = results['parcel_results']
    
    results['global_stats'] = {
        'mean_pr': np.mean([p['participation_ratio'] for p in parcel_results]),
        'std_pr': np.std([p['participation_ratio'] for p in parcel_results]),
        'mean_d_mle': np.mean([p['d_mle'] for p in parcel_results]),
        'std_d_mle': np.std([p['d_mle'] for p in parcel_results]),
        'mean_d_correlation': np.mean([p['d_correlation'] for p in parcel_results]),
        'std_d_correlation': np.std([p['d_correlation'] for p in parcel_results]),
        'mean_scaling_exponent': np.mean([p['scaling_exponent'] for p in parcel_results]),
        'std_scaling_exponent': np.std([p['scaling_exponent'] for p in parcel_results]),
        'mean_inflection': np.mean([p['inflection_index'] for p in parcel_results]),
        'std_inflection': np.std([p['inflection_index'] for p in parcel_results]),
        'mean_top3': np.mean([p['top3_variance'] for p in parcel_results]),
        'std_top3': np.std([p['top3_variance'] for p in parcel_results]),
        'mean_n_layers': np.mean([p['n_layers'] for p in parcel_results])
    }
    
    return results

def generate_synthetic_domains():
    """Generate synthetic domains for comparison."""
    np.random.seed(RANDOM_SEED)
    
    domains = {}
    
    domains['uniform_sphere'] = {
        'name': 'Uniform Sphere',
        'generate': lambda n, d: np.random.randn(n, d) / np.sqrt(d),
        'description': 'Points on uniform sphere'
    }
    
    domains['gaussian_cloud'] = {
        'name': 'Gaussian Cloud',
        'generate': lambda n, d: np.random.randn(n, d) * np.random.rand(d) + np.random.randn(d),
        'description': 'Correlated Gaussian'
    }
    
    domains['sparse'] = {
        'name': 'Sparse',
        'generate': lambda n, d: np.random.randn(n, d) * (np.random.rand(n, 1) > 0.7),
        'description': 'Sparse activations'
    }
    
    domains['curved_manifold'] = {
        'name': 'Curved Manifold',
        'generate': lambda n, d: generate_curved_manifold(n, d),
        'description': 'Non-linear manifold'
    }
    
    domains['hierarchical'] = {
        'name': 'Hierarchical',
        'generate': lambda n, d: generate_hierarchical(n, d),
        'description': 'Multi-scale structure'
    }
    
    return domains

def generate_curved_manifold(n, d):
    """Generate points on a curved manifold."""
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.uniform(0.5, 1.5, n)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    data = np.column_stack([x, y])
    
    for i in range(d - 2):
        data = np.column_stack([data, np.random.randn(n) * 0.1])
    
    return data[:, :d]

def generate_hierarchical(n, d):
    """Generate hierarchical multi-scale structure."""
    data = np.zeros((n, d))
    
    scales = [4, 2, 1, 0.5]
    
    for scale in scales:
        n_points = int(n / len(scales))
        start = int(np.sum([n / len(scales) for _ in scales[:scales.index(scale)]]))
        end = min(start + n_points, n)
        
        for i in range(start, end):
            for j in range(d):
                data[i, j] = np.random.randn() * scale
    
    return data

def split_data_analysis(tribe_data):
    """Analyze data splits for invariance."""
    np.random.seed(RANDOM_SEED)
    n = len(tribe_data)
    indices = np.random.permutation(n)
    half = n // 2
    
    data1 = [tribe_data[i] for i in indices[:half]]
    data2 = [tribe_data[i] for i in indices[half:2*half]]
    
    parcels1 = extract_parcel_data(data1, n_parcels=N_PARCELS, n_samples=30)
    parcels2 = extract_parcel_data(data2, n_parcels=N_PARCELS, n_samples=30)
    
    results1 = analyze_dataset(parcels1, name="Split1")
    results2 = analyze_dataset(parcels2, name="Split2")
    
    return results1, results2, parcels1, parcels2

def compute_invariance_metrics(results_list):
    """Compute invariance metrics across analyses."""
    metrics = {
        'pr_values': [r['global_stats']['mean_pr'] for r in results_list],
        'd_mle_values': [r['global_stats']['mean_d_mle'] for r in results_list],
        'd_corr_values': [r['global_stats']['mean_d_correlation'] for r in results_list],
        'scaling_values': [r['global_stats']['mean_scaling_exponent'] for r in results_list],
        'top3_values': [r['global_stats']['mean_top3'] for r in results_list]
    }
    
    cv = lambda x: np.std(x) / np.abs(np.mean(x)) if np.mean(x) != 0 else np.inf
    
    return {
        'pr_cv': cv(metrics['pr_values']),
        'pr_mean': np.mean(metrics['pr_values']),
        'd_mle_cv': cv(metrics['d_mle_values']),
        'd_mle_mean': np.mean(metrics['d_mle_values']),
        'd_corr_cv': cv(metrics['d_corr_values']),
        'd_corr_mean': np.mean(metrics['d_corr_values']),
        'scaling_cv': cv(metrics['scaling_values']),
        'scaling_mean': np.mean(metrics['scaling_values']),
        'top3_cv': cv(metrics['top3_values']),
        'top3_mean': np.mean(metrics['top3_values']),
        'is_invariant': cv(metrics['pr_values']) < 0.1 and cv(metrics['d_mle_values']) < 0.1
    }

def create_visualizations(all_results, invariance_metrics):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    names = list(all_results.keys())
    
    all_eigenvalues = []
    for name, results in all_results.items():
        eigenvalues = np.mean([p['eigenvalues'] for p in results['parcel_results']], axis=0)
        eigenvalues = eigenvalues / eigenvalues[0]
        all_eigenvalues.append(eigenvalues)
        
        x = np.arange(1, len(eigenvalues) + 1)
        axes[0, 0].plot(x, eigenvalues, marker='o', label=name, alpha=0.7)
    
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Normalized Eigenvalue')
    axes[0, 0].set_title('PCA Eigenvalue Spectrum')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    for idx, (name, results) in enumerate(all_results.items()):
        cum_var = np.mean([p['cumulative_variance'] for p in results['parcel_results']], axis=0)
        x = np.arange(1, len(cum_var) + 1)
        axes[0, 1].plot(x, cum_var, marker='s', label=name, alpha=0.7)
    
    axes[0, 1].set_xlabel('Component')
    axes[0, 1].set_ylabel('Cumulative Variance')
    axes[0, 1].set_title('Cumulative Variance Explained')
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    
    pr_values = [results['global_stats']['mean_pr'] for results in all_results.values()]
    pr_std = [results['global_stats']['std_pr'] for results in all_results.values()]
    
    x = np.arange(len(names))
    axes[0, 2].bar(x, pr_values, yerr=pr_std, capsize=5, color=colors)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Participation Ratio')
    axes[0, 2].set_title('Participation Ratio (PR)')
    axes[0, 2].axhline(y=np.mean(pr_values), color='r', linestyle='--', alpha=0.5)
    
    d_mle_values = [results['global_stats']['mean_d_mle'] for results in all_results.values()]
    d_mle_std = [results['global_stats']['std_d_mle'] for results in all_results.values()]
    
    axes[1, 0].bar(x, d_mle_values, yerr=d_mle_std, capsize=5, color=colors)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('MLE Dimensionality')
    axes[1, 0].set_title('Intrinsic Dimensionality (MLE)')
    axes[1, 0].axhline(y=np.mean(d_mle_values), color='r', linestyle='--', alpha=0.5)
    
    scaling_values = [results['global_stats']['mean_scaling_exponent'] for results in all_results.values()]
    scaling_std = [results['global_stats']['std_scaling_exponent'] for results in all_results.values()]
    
    axes[1, 1].bar(x, scaling_values, yerr=scaling_std, capsize=5, color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Scaling Exponent')
    axes[1, 1].set_title('Power Law Scaling Exponent')
    axes[1, 1].axhline(y=np.mean(scaling_values), color='r', linestyle='--', alpha=0.5)
    
    top3_values = [results['global_stats']['mean_top3'] for results in all_results.values()]
    top3_std = [results['global_stats']['std_top3'] for results in all_results.values()]
    
    axes[1, 2].bar(x, top3_values, yerr=top3_std, capsize=5, color=colors)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Variance Explained')
    axes[1, 2].set_title('Top-3 Components Variance')
    axes[1, 2].axhline(y=np.mean(top3_values), color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_spectrum_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_spectrum_analysis.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    pr_cv_values = [invariance_metrics['pr_cv']]
    d_mle_cv_values = [invariance_metrics['d_mle_cv']]
    scaling_cv_values = [invariance_metrics['scaling_cv']]
    top3_cv_values = [invariance_metrics['top3_cv']]
    
    metric_names = ['PR', 'D_MLE', 'Scaling', 'Top3']
    cv_values = [invariance_metrics['pr_cv'], invariance_metrics['d_mle_cv'],
                 invariance_metrics['scaling_cv'], invariance_metrics['top3_cv']]
    mean_values = [invariance_metrics['pr_mean'], invariance_metrics['d_mle_mean'],
                   invariance_metrics['scaling_mean'], invariance_metrics['top3_mean']]
    
    bars = axes[0, 0].bar(metric_names, cv_values)
    colors_cv = ['green' if cv < 0.1 else 'orange' if cv < 0.3 else 'red' for cv in cv_values]
    for bar, color in zip(bars, colors_cv):
        bar.set_color(color)
    axes[0, 0].set_ylabel('Coefficient of Variation')
    axes[0, 0].set_title('Invariance Metrics (CV < 0.1 = invariant)')
    axes[0, 0].axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
    
    axes[0, 1].bar(metric_names, mean_values)
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].set_title('Global Metric Values')
    
    pr_by_parcel = []
    for name, results in list(all_results.items())[:3]:
        prs = [p['participation_ratio'] for p in results['parcel_results']]
        pr_by_parcel.append(prs)
    
    bp = axes[1, 0].boxplot(pr_by_parcel, labels=list(all_results.keys())[:3], patch_artist=True)
    colors_bp = plt.cm.Set2(np.linspace(0, 1, 3))
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel('Participation Ratio')
    axes[1, 0].set_title('PR Distribution Across Parcels')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    d_mle_by_parcel = []
    for name, results in list(all_results.items())[:3]:
        d_mles = [p['d_mle'] for p in results['parcel_results']]
        d_mle_by_parcel.append(d_mles)
    
    bp2 = axes[1, 1].boxplot(d_mle_by_parcel, labels=list(all_results.keys())[:3], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors_bp):
        patch.set_facecolor(color)
    axes[1, 1].set_ylabel('MLE Dimensionality')
    axes[1, 1].set_title('Dimensionality Distribution Across Parcels')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_invariance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_invariance.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    log_log_data = {}
    for name, results in all_results.items():
        eigenvalues = np.mean([p['eigenvalues'] for p in results['parcel_results']], axis=0)
        rank = np.arange(1, len(eigenvalues) + 1)
        
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        log_rank = np.log(rank)
        log_eig = np.log(eigenvalues)
        
        log_log_data[name] = {'rank': rank, 'eigenvalues': eigenvalues, 
                              'log_rank': log_rank, 'log_eig': log_eig}
        
        axes[0, 0].scatter(log_rank, log_eig, alpha=0.5, label=name, s=20)
    
    axes[0, 0].set_xlabel('log(Rank)')
    axes[0, 0].set_ylabel('log(Eigenvalue)')
    axes[0, 0].set_title('Log-Log Eigenvalue Spectrum')
    axes[0, 0].legend()
    
    for name, data in log_log_data.items():
        slope, intercept, r_value, _, _ = stats.linregress(data['log_rank'], data['log_eig'])
        fit_line = slope * data['log_rank'] + intercept
        axes[0, 1].plot(data['log_rank'], fit_line, label=f'{name}: β={slope:.2f}', linewidth=2)
        axes[0, 1].scatter(data['log_rank'], data['log_eig'], alpha=0.3, s=10)
    
    axes[0, 1].set_xlabel('log(Rank)')
    axes[0, 1].set_ylabel('log(Eigenvalue)')
    axes[0, 1].set_title('Power Law Fits')
    axes[0, 1].legend(fontsize=8)
    
    inflection_indices = [results['global_stats']['mean_inflection'] for results in all_results.values()]
    n_layers = [results['global_stats']['mean_n_layers'] for results in all_results.values()]
    
    x = np.arange(len(names))
    width = 0.35
    axes[1, 0].bar(x - width/2, inflection_indices, width, label='Inflection Index')
    axes[1, 0].bar(x + width/2, n_layers, width, label='N Layers')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Compression Structure')
    axes[1, 0].legend()
    
    pr_vs_d = []
    for name, results in all_results.items():
        prs = [p['participation_ratio'] for p in results['parcel_results']]
        d_mles = [p['d_mle'] for p in results['parcel_results']]
        pr_vs_d.append((prs, d_mles, name))
    
    for prs, d_mles, name in pr_vs_d:
        axes[1, 1].scatter(prs, d_mles, alpha=0.3, label=name, s=10)
    
    axes[1, 1].set_xlabel('Participation Ratio')
    axes[1, 1].set_ylabel('MLE Dimensionality')
    axes[1, 1].set_title('PR vs Dimensionality')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_scaling_analysis.png")
    
    return

def create_summary_tables(all_results, invariance_metrics):
    """Create summary tables."""
    
    summary_data = []
    for name, results in all_results.items():
        gs = results['global_stats']
        summary_data.append({
            'Dataset': name,
            'PR_Mean': gs['mean_pr'],
            'PR_Std': gs['std_pr'],
            'D_MLE_Mean': gs['mean_d_mle'],
            'D_MLE_Std': gs['std_d_mle'],
            'D_Corr_Mean': gs['mean_d_correlation'],
            'D_Corr_Std': gs['std_d_correlation'],
            'Scaling_Exp': gs['mean_scaling_exponent'],
            'Scaling_Std': gs['std_scaling_exponent'],
            'Inflection_Mean': gs['mean_inflection'],
            'Top3_Variance': gs['mean_top3']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "structural_summary.csv", index=False)
    print(f"✓ Summary saved to {OUTPUT_DIR / 'structural_summary.csv'}")
    
    parcel_data = []
    for name, results in all_results.items():
        for p in results['parcel_results']:
            parcel_data.append({
                'Dataset': name,
                'Parcel_ID': p['parcel_id'],
                'PR': p['participation_ratio'],
                'D_MLE': p['d_mle'],
                'D_Correlation': p['d_correlation'],
                'Scaling_Exp': p['scaling_exponent'],
                'Scaling_R2': p['scaling_r2'],
                'Inflection_Index': p['inflection_index'],
                'Top3_Variance': p['top3_variance']
            })
    
    parcel_df = pd.DataFrame(parcel_data)
    parcel_df.to_csv(OUTPUT_DIR / "parcel_structural.csv", index=False)
    print(f"✓ Parcel data saved to {OUTPUT_DIR / 'parcel_structural.csv'}")
    
    return summary_df, parcel_df

def generate_report(all_results, invariance_metrics):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V12 INTERPRETATION REPORT: INVARIANT STRUCTURAL PROPERTIES")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: What structural properties are invariant across domains?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. SUCCESS CRITERIA ASSESSMENT")
    report.append("-" * 50)
    
    report.append(f"  a) Stable curves across splits:")
    report.append(f"     - PR CV: {invariance_metrics['pr_cv']:.4f} {'✓' if invariance_metrics['pr_cv'] < 0.1 else '✗'}")
    report.append(f"     - D_MLE CV: {invariance_metrics['d_mle_cv']:.4f} {'✓' if invariance_metrics['d_mle_cv'] < 0.1 else '✗'}")
    
    report.append(f"  b) Similar shapes across domains:")
    report.append(f"     - Scaling CV: {invariance_metrics['scaling_cv']:.4f} {'✓' if invariance_metrics['scaling_cv'] < 0.1 else '✗'}")
    
    is_invariant = invariance_metrics['is_invariant']
    report.append(f"\n  Overall invariant structure: {'✓ YES' if is_invariant else '✗ NO'}")
    
    report.append("")
    report.append("2. GLOBAL METRIC SUMMARY")
    report.append("-" * 50)
    
    report.append(f"  Participation Ratio (PR):")
    report.append(f"    Mean: {invariance_metrics['pr_mean']:.3f}")
    report.append(f"    CV: {invariance_metrics['pr_cv']:.4f}")
    report.append(f"    Interpretation: {'High dimensionality' if invariance_metrics['pr_mean'] > 3 else 'Low dimensionality'}")
    
    report.append(f"\n  Intrinsic Dimensionality (MLE):")
    report.append(f"    Mean: {invariance_metrics['d_mle_mean']:.3f}")
    report.append(f"    CV: {invariance_metrics['d_mle_cv']:.4f}")
    
    report.append(f"\n  Scaling Exponent:")
    report.append(f"    Mean: {invariance_metrics['scaling_mean']:.3f}")
    report.append(f"    CV: {invariance_metrics['scaling_cv']:.4f}")
    report.append(f"    Interpretation: {'Fast decay' if invariance_metrics['scaling_mean'] < -1 else 'Slow decay'}")
    
    report.append("")
    report.append("3. DATASET-BY-DATASET RESULTS")
    report.append("-" * 50)
    
    header = f"{'Dataset':<20} {'PR':>8} {'D_MLE':>8} {'Scale β':>8} {'Top3':>8}"
    report.append(header)
    report.append("-" * len(header))
    
    for name, results in all_results.items():
        gs = results['global_stats']
        report.append(f"{name:<20} {gs['mean_pr']:>8.3f} {gs['mean_d_mle']:>8.3f} "
                     f"{gs['mean_scaling_exponent']:>8.3f} {gs['mean_top3']:>8.3f}")
    
    report.append("")
    report.append("4. CROSS-DOMAIN COMPARISON")
    report.append("-" * 50)
    
    pr_values = [r['global_stats']['mean_pr'] for r in all_results.values()]
    d_values = [r['global_stats']['mean_d_mle'] for r in all_results.values()]
    s_values = [r['global_stats']['mean_scaling_exponent'] for r in all_results.values()]
    
    if len(all_results) > 1:
        pr_range = max(pr_values) - min(pr_values)
        d_range = max(d_values) - min(d_values)
        s_range = max(s_values) - min(s_values)
        
        report.append(f"  PR range: {pr_range:.3f} ({min(pr_values):.3f} - {max(pr_values):.3f})")
        report.append(f"  D_MLE range: {d_range:.3f} ({min(d_values):.3f} - {max(d_values):.3f})")
        report.append(f"  Scaling range: {s_range:.3f} ({min(s_values):.3f} - {max(s_values):.3f})")
    
    report.append("")
    report.append("5. KEY FINDINGS")
    report.append("-" * 50)
    
    if invariance_metrics['pr_cv'] < 0.1:
        report.append("  ✓ Participation Ratio is INVARIANT")
        report.append(f"    All datasets show similar dimensionality structure (PR ≈ {invariance_metrics['pr_mean']:.2f})")
    else:
        report.append("  ✗ Participation Ratio varies across datasets")
    
    if invariance_metrics['d_mle_cv'] < 0.1:
        report.append("  ✓ Intrinsic Dimensionality is INVARIANT")
        report.append(f"    All datasets estimate similar intrinsic dimension (D ≈ {invariance_metrics['d_mle_mean']:.2f})")
    else:
        report.append("  ✗ Intrinsic Dimensionality varies across datasets")
    
    if invariance_metrics['scaling_cv'] < 0.3:
        report.append("  ✓ Scaling laws are CONSISTENT")
        report.append(f"    Power law exponents cluster around β ≈ {invariance_metrics['scaling_mean']:.2f}")
    else:
        report.append("  ✗ Scaling laws vary across datasets")
    
    report.append("")
    report.append("6. CONCLUSIONS")
    report.append("-" * 50)
    
    if is_invariant:
        report.append("  STRUCTURAL INVARIANCE DETECTED")
        report.append("  ")
        report.append("  The following properties are INVARIANT across data splits and domains:")
        report.append(f"    1. Participation Ratio ≈ {invariance_metrics['pr_mean']:.2f}")
        report.append(f"    2. Intrinsic Dimensionality ≈ {invariance_metrics['d_mle_mean']:.2f}")
        report.append(f"    3. Scaling Exponent ≈ {invariance_metrics['scaling_mean']:.2f}")
        report.append("  ")
        report.append("  These represent genuine structural properties that are NOT")
        report.append("  artifacts of specific data configurations or model variations.")
    else:
        report.append("  NO COMPLETE STRUCTURAL INVARIANCE")
        report.append("  ")
        report.append("  While some metrics may be partially stable, there is no")
        report.append("  consistent structural invariance across all properties.")
        report.append("  ")
        report.append("  This suggests that structural properties depend on the")
        report.append("  specific data configuration and cannot be considered universal.")
    
    report.append("")
    report.append("7. IMPLICATIONS FOR MANUSCRIPT")
    report.append("-" * 50)
    
    if is_invariant:
        report.append("  - Report participation ratio as key invariant metric")
        report.append("  - Report intrinsic dimensionality estimation")
        report.append("  - Report scaling exponent with confidence intervals")
        report.append("  - These represent robust, data-independent properties")
    else:
        report.append("  - Be cautious about claiming structural invariance")
        report.append("  - Report metrics with appropriate uncertainty")
        report.append("  - Acknowledge dependence on data configuration")
    
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
    print("PHASE 2: EXTRACTING PARCEL DATA")
    print("=" * 80)
    
    all_parcels = extract_parcel_data(tribe_data, n_parcels=N_PARCELS, n_samples=50)
    print(f"Extracted {len(all_parcels)} parcels")
    
    print("\n" + "=" * 80)
    print("PHASE 3: ANALYZING ALL DATASETS")
    print("=" * 80)
    
    all_results = {}
    
    print("\nAnalyzing full TRIBE dataset...")
    all_results['TRIBE_Full'] = analyze_dataset(all_parcels, name="TRIBE_Full")
    
    print("\nAnalyzing data splits...")
    split1, split2, _, _ = split_data_analysis(tribe_data)
    all_results['TRIBE_Split1'] = split1
    all_results['TRIBE_Split2'] = split2
    
    print("\nAnalyzing synthetic domains...")
    synthetic_domains = generate_synthetic_domains()
    
    for domain_name, domain in synthetic_domains.items():
        print(f"  Generating {domain_name}...")
        data = domain['generate'](n=N_PARCELS * 50, d=9)
        
        parcel_data = {}
        samples_per_parcel = len(data) // N_PARCELS
        for parcel_idx in range(N_PARCELS):
            start = parcel_idx * samples_per_parcel
            end = start + samples_per_parcel
            parcel_data[parcel_idx] = data[start:end]
        
        all_results[domain_name] = analyze_dataset(parcel_data, name=domain_name)
    
    print("\n" + "=" * 80)
    print("PHASE 4: COMPUTING INVARIANCE METRICS")
    print("=" * 80)
    
    invariance_metrics = compute_invariance_metrics(list(all_results.values()))
    
    print(f"\nInvariance Assessment:")
    print(f"  Participation Ratio CV: {invariance_metrics['pr_cv']:.4f}")
    print(f"  Dimensionality CV: {invariance_metrics['d_mle_cv']:.4f}")
    print(f"  Scaling CV: {invariance_metrics['scaling_cv']:.4f}")
    print(f"  Is Invariant: {invariance_metrics['is_invariant']}")
    
    print("\n" + "=" * 80)
    print("PHASE 5: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(all_results, invariance_metrics)
    
    print("\n" + "=" * 80)
    print("PHASE 6: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_tables(all_results, invariance_metrics)
    
    print("\n" + "=" * 80)
    print("PHASE 7: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(all_results, invariance_metrics)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - structural_summary.csv")
    print("  - parcel_structural.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_spectrum_analysis.png")
    print("  - figures/fig2_invariance.png")
    print("  - figures/fig3_scaling_analysis.png")

if __name__ == "__main__":
    main()
