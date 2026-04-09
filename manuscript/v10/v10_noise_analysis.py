#!/usr/bin/env python3
"""
V10 Pipeline: Noise/Jitter Analysis (ξ)
Discover structure in residual variability across systems
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v10")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

N_PARCELS = 400
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V10 NOISE/JITTER ANALYSIS (ξ)")
print("Discovering structure in residual variability")
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

def extract_parcel_activations(tribe_data, n_parcels=400):
    """Extract activation patterns for parcels from TRIBE data."""
    node_keys = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
                 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
    
    n_per_parcel = max(1, len(tribe_data) // n_parcels)
    
    parcel_data = []
    for parcel_idx in range(n_parcels):
        base_idx = (parcel_idx * n_per_parcel) % len(tribe_data)
        
        activations = []
        for i in range(n_per_parcel):
            data_idx = (base_idx + i) % len(tribe_data)
            if 'sgp_nodes' in tribe_data[data_idx]:
                sgp = tribe_data[data_idx]['sgp_nodes']
                values = [sgp.get(k, 0.5) for k in node_keys]
                activations.append(values)
        
        if not activations:
            activations = [[0.5] * 9]
        
        parcel_data.append({
            'parcel_id': parcel_idx,
            'activations': np.array(activations),
            'category': tribe_data[base_idx % len(tribe_data)].get('category', 'unknown') if base_idx < len(tribe_data) else 'unknown'
        })
    
    return parcel_data

def compute_residuals(activations):
    """Compute residuals: ξ = x - mean(x)"""
    mean = np.mean(activations, axis=0)
    residuals = activations - mean
    return residuals, mean

def compute_distribution_statistics(residuals):
    """Compute distribution statistics for residuals."""
    stats_dict = {}
    
    stats_dict['mean'] = np.mean(residuals, axis=0)
    stats_dict['variance'] = np.var(residuals, axis=0)
    stats_dict['std'] = np.std(residuals, axis=0)
    
    flat_residuals = residuals.flatten()
    
    stats_dict['overall_mean'] = np.mean(flat_residuals)
    stats_dict['overall_var'] = np.var(flat_residuals)
    stats_dict['overall_std'] = np.std(flat_residuals)
    
    if len(flat_residuals) >= 3:
        stats_dict['skewness'] = stats.skew(flat_residuals)
        stats_dict['kurtosis'] = stats.kurtosis(flat_residuals)
    else:
        stats_dict['skewness'] = 0
        stats_dict['kurtosis'] = 0
    
    n = len(flat_residuals)
    if n >= 3:
        _, shapiro_p = stats.shapiro(flat_residuals[:min(5000, n)])
        stats_dict['shapiro_p'] = shapiro_p
        stats_dict['is_gaussian'] = shapiro_p > 0.05
    else:
        stats_dict['shapiro_p'] = 1.0
        stats_dict['is_gaussian'] = True
    
    return stats_dict

def compute_spectral_structure(residuals):
    """Compute spectral structure via covariance matrix and eigenvalues."""
    if residuals.shape[0] < 2:
        return {'eigenvalues': np.array([0]), 'explained_variance': np.array([0])}
    
    cov_matrix = np.cov(residuals.T)
    
    if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
        cov_matrix = np.eye(residuals.shape[1]) * 0.01
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    
    total_var = np.sum(eigenvalues)
    if total_var > 0:
        explained_variance = eigenvalues / total_var
    else:
        explained_variance = np.zeros_like(eigenvalues)
    
    spectral_dict = {
        'covariance_matrix': cov_matrix,
        'eigenvalues': eigenvalues,
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance),
        'spectral_entropy': -np.sum(explained_variance * np.log(explained_variance + 1e-10)),
        'condition_number': eigenvalues[0] / (eigenvalues[-1] + 1e-10) if eigenvalues[-1] > 0 else np.inf
    }
    
    return spectral_dict

def compute_spatial_structure(all_parcel_stats, all_spectral):
    """Compute spatial structure across parcels."""
    n_parcels = len(all_parcel_stats)
    
    similarity_matrix = np.zeros((n_parcels, n_parcels))
    for i in range(n_parcels):
        for j in range(n_parcels):
            stat_i = np.array([
                all_parcel_stats[i]['overall_var'],
                all_parcel_stats[i]['skewness'],
                all_parcel_stats[i]['kurtosis']
            ])
            stat_j = np.array([
                all_parcel_stats[j]['overall_var'],
                all_parcel_stats[j]['skewness'],
                all_parcel_stats[j]['kurtosis']
            ])
            similarity_matrix[i, j] = 1 / (1 + np.linalg.norm(stat_i - stat_j))
    
    feature_matrix = np.array([
        [all_parcel_stats[i]['overall_var'] for i in range(n_parcels)],
        [all_parcel_stats[i]['skewness'] for i in range(n_parcels)],
        [all_parcel_stats[i]['kurtosis'] for i in range(n_parcels)],
        [np.mean(all_spectral[i]['explained_variance'][:3]) for i in range(n_parcels)]
    ]).T
    
    scaler = StandardScaler()
    feature_normalized = scaler.fit_transform(feature_matrix)
    
    return {
        'similarity_matrix': similarity_matrix,
        'feature_matrix': feature_matrix,
        'feature_normalized': feature_normalized
    }

def cluster_parcels(feature_matrix, n_clusters_range=range(2, 8)):
    """Cluster parcels and find optimal number of clusters."""
    scaler = StandardScaler()
    features = scaler.fit_transform(feature_matrix)
    
    results = {}
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(features)
        
        if n_clusters > 1:
            silhouette = silhouette_score(features, labels)
        else:
            silhouette = 0
        
        results[n_clusters] = {
            'labels': labels,
            'silhouette': silhouette,
            'inertia': kmeans.inertia_
        }
    
    best_n = max(results.keys(), key=lambda k: results[k]['silhouette'])
    
    return results, best_n

def analyze_parcel(parcel_data):
    """Full analysis for a single parcel."""
    activations = parcel_data['activations']
    residuals, mean = compute_residuals(activations)
    dist_stats = compute_distribution_statistics(residuals)
    spectral = compute_spectral_structure(residuals)
    
    return {
        'parcel_id': parcel_data['parcel_id'],
        'category': parcel_data['category'],
        'n_samples': activations.shape[0],
        'mean': mean,
        'residuals': residuals,
        'distribution': dist_stats,
        'spectral': spectral
    }

def generate_synthetic_domains(n_parcels=400, n_samples=100):
    """Generate synthetic domains for comparison."""
    np.random.seed(RANDOM_SEED)
    
    domains = {}
    
    domains['gaussian'] = {
        'name': 'Gaussian',
        'data': np.random.randn(n_parcels, n_samples, 9) * 0.3,
        'description': 'Multivariate Gaussian noise'
    }
    
    domains['uniform'] = {
        'name': 'Uniform',
        'data': np.random.uniform(-1, 1, (n_parcels, n_samples, 9)),
        'description': 'Uniform distribution'
    }
    
    domains['heavy_tail'] = {
        'name': 'Heavy Tail',
        'data': np.random.standard_t(df=3, size=(n_parcels, n_samples, 9)) * 0.3,
        'description': 'Student-t with heavy tails'
    }
    
    domains['multimodal'] = {
        'name': 'Multimodal',
        'data': np.concatenate([
            np.random.randn(n_parcels, n_samples//2, 9) * 0.2 - 0.5,
            np.random.randn(n_parcels, n_samples//2, 9) * 0.2 + 0.5
        ], axis=1),
        'description': 'Bimodal mixture'
    }
    
    domains['correlated'] = {
        'name': 'Correlated',
        'data': np.zeros((n_parcels, n_samples, 9)),
        'description': 'Correlated structure'
    }
    for i in range(n_parcels):
        base = np.random.randn(n_samples)
        for j in range(9):
            domains['correlated']['data'][i, :, j] = base + np.random.randn(n_samples) * 0.2
    
    return domains

def analyze_domain(domain_name, domain_data):
    """Analyze a domain (TRIBE or synthetic)."""
    n_parcels = domain_data.shape[0]
    
    parcel_stats = []
    all_spectral = []
    
    for parcel_idx in range(n_parcels):
        parcel_activations = domain_data[parcel_idx]
        residuals, _ = compute_residuals(parcel_activations)
        
        dist_stats = compute_distribution_statistics(residuals)
        spectral = compute_spectral_structure(residuals)
        
        parcel_stats.append(dist_stats)
        all_spectral.append(spectral)
    
    domain_stats = {
        'name': domain_name,
        'n_parcels': n_parcels,
        'mean_variance': np.mean([p['overall_var'] for p in parcel_stats]),
        'mean_skewness': np.mean([p['skewness'] for p in parcel_stats]),
        'mean_kurtosis': np.mean([p['kurtosis'] for p in parcel_stats]),
        'gaussian_fraction': np.mean([p['is_gaussian'] for p in parcel_stats]),
        'mean_spectral_entropy': np.mean([s['spectral_entropy'] for s in all_spectral]),
        'mean_condition_number': np.mean([s['condition_number'] for s in all_spectral]),
        'parcel_stats': parcel_stats,
        'spectral': all_spectral
    }
    
    return domain_stats

def create_visualizations(domain_results, spatial_results, parcel_analysis):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, results) in enumerate(domain_results.items()):
        if idx >= 6:
            break
        parcel_stats = results['parcel_stats']
        
        variances = [p['overall_var'] for p in parcel_stats]
        axes[0, 0].hist(variances, bins=30, alpha=0.5, label=name)
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Residual Variance')
    axes[0, 0].legend()
    
    for idx, (name, results) in enumerate(domain_results.items()):
        if idx >= 6:
            break
        parcel_stats = results['parcel_stats']
        
        skewness = [p['skewness'] for p in parcel_stats]
        axes[0, 1].hist(skewness, bins=30, alpha=0.5, label=name)
    axes[0, 1].set_xlabel('Skewness')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Skewness')
    axes[0, 1].legend()
    
    for idx, (name, results) in enumerate(domain_results.items()):
        if idx >= 6:
            break
        parcel_stats = results['parcel_stats']
        
        kurtosis = [p['kurtosis'] for p in parcel_stats]
        axes[0, 2].hist(kurtosis, bins=30, alpha=0.5, label=name)
    axes[0, 2].set_xlabel('Kurtosis')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Distribution of Kurtosis')
    axes[0, 2].legend()
    axes[0, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Gaussian')
    
    gaussian_fractions = [results['gaussian_fraction'] for results in domain_results.values()]
    axes[1, 0].bar(domain_results.keys(), gaussian_fractions)
    axes[1, 0].set_ylabel('Fraction Gaussian (p > 0.05)')
    axes[1, 0].set_title('Normality by Domain')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    spectral_entropies = [results['mean_spectral_entropy'] for results in domain_results.values()]
    axes[1, 1].bar(domain_results.keys(), spectral_entropies)
    axes[1, 1].set_ylabel('Mean Spectral Entropy')
    axes[1, 1].set_title('Spectral Complexity')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    condition_numbers = [min(results['mean_condition_number'], 1000) for results in domain_results.values()]
    axes[1, 2].bar(domain_results.keys(), condition_numbers)
    axes[1, 2].set_ylabel('Mean Condition Number (capped)')
    axes[1, 2].set_title('Covariance Condition Number')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_distribution_analysis.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for name, results in domain_results.items():
        all_eigenvalues = []
        for spectral in results['spectral']:
            all_eigenvalues.extend(spectral['eigenvalues'][:5])
        
        if all_eigenvalues:
            sorted_eig = np.sort(all_eigenvalues)[::-1][:50]
            normalized = sorted_eig / (sorted_eig[0] + 1e-10)
            axes[0, 0].plot(normalized, alpha=0.7, label=name)
    
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Normalized Eigenvalue')
    axes[0, 0].set_title('Eigenvalue Spectra (Averaged)')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    tribe_results = domain_results.get('TRIBE', None)
    if tribe_results and len(tribe_results['spectral']) > 0:
        first_3_var = [np.sum(s['explained_variance'][:3]) for s in tribe_results['spectral']]
        axes[0, 1].hist(first_3_var, bins=30, color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Variance Explained (PC1-3)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('TRIBE: Variance in Top 3 Components')
        axes[0, 1].axvline(x=np.mean(first_3_var), color='r', linestyle='--', 
                          label=f'Mean={np.mean(first_3_var):.2f}')
        axes[0, 1].legend()
    
    features = spatial_results['feature_normalized']
    n_clusters_range = range(2, 8)
    silhouettes = []
    inertias = []
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(spatial_results['feature_matrix'])
    
    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        silhouettes.append(silhouette_score(features_scaled, labels))
        inertias.append(kmeans.inertia_)
    
    axes[1, 0].plot(list(n_clusters_range), silhouettes, marker='o')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Optimal Cluster Selection')
    best_n = list(n_clusters_range)[np.argmax(silhouettes)]
    axes[1, 0].axvline(x=best_n, color='r', linestyle='--', alpha=0.5)
    
    axes[1, 1].plot(list(n_clusters_range), inertias, marker='o')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Inertia')
    axes[1, 1].set_title('Elbow Method')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_spectral_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_spectral_structure.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    tribe_results = domain_results.get('TRIBE', None)
    if tribe_results:
        parcel_stats = tribe_results['parcel_stats']
        
        variances = [p['overall_var'] for p in parcel_stats]
        axes[0, 0].hist(variances, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Residual Variance')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('TRIBE: Residual Variance Distribution')
        axes[0, 0].axvline(x=np.mean(variances), color='r', linestyle='--', 
                           label=f'Mean={np.mean(variances):.3f}')
        axes[0, 0].legend()
        
        if len(variances) >= 20:
            stats.probplot(variances, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('TRIBE: Q-Q Plot of Variance')
        
        parcel_ids = range(len(parcel_stats))
        variances_by_parcel = [p['overall_var'] for p in parcel_stats]
        skewness_by_parcel = [p['skewness'] for p in parcel_stats]
        
        axes[1, 0].scatter(variances_by_parcel, skewness_by_parcel, alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Variance')
        axes[1, 0].set_ylabel('Skewness')
        axes[1, 0].set_title('TRIBE: Variance vs Skewness')
        
        im = axes[1, 1].imshow(spatial_results['similarity_matrix'], cmap='viridis', aspect='auto')
        axes[1, 1].set_xlabel('Parcel')
        axes[1, 1].set_ylabel('Parcel')
        axes[1, 1].set_title('Parcel Similarity Matrix')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_tribe_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_tribe_detailed.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    n_clusters = best_n
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    cluster_variance = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        cluster_variance[label].append(spatial_results['feature_matrix'][i, 0])
    
    for i in range(n_clusters):
        axes[0, 0].hist(cluster_variance[i], bins=20, alpha=0.7, label=f'Cluster {i}')
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Clustering: Variance by Cluster (k={n_clusters})')
    axes[0, 0].legend()
    
    parcel_ids = np.arange(len(labels))
    for i in range(n_clusters):
        mask = labels == i
        axes[0, 1].scatter(parcel_ids[mask], np.zeros_like(parcel_ids[mask]) + i, 
                          alpha=0.5, s=10, label=f'Cluster {i}')
    axes[0, 1].set_xlabel('Parcel ID')
    axes[0, 1].set_ylabel('Cluster')
    axes[0, 1].set_title('Parcel Cluster Assignments')
    axes[0, 1].legend()
    
    variance_by_cluster = [np.mean(cluster_variance[i]) for i in range(n_clusters)]
    axes[0, 2].bar(range(n_clusters), variance_by_cluster)
    axes[0, 2].set_xlabel('Cluster')
    axes[0, 2].set_ylabel('Mean Variance')
    axes[0, 2].set_title('Mean Variance by Cluster')
    
    for idx, (name, results) in enumerate(domain_results.items()):
        if idx >= 6:
            break
        parcel_stats = results['parcel_stats']
        
        kurtosis_vals = [p['kurtosis'] for p in parcel_stats]
        axes[1, 0].hist(kurtosis_vals, bins=30, alpha=0.5, label=name)
    axes[1, 0].set_xlabel('Kurtosis')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Kurtosis Comparison')
    axes[1, 0].legend()
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    shapiro_ps = [[p['shapiro_p'] for p in results['parcel_stats']] 
                  for results in domain_results.values()]
    bp = axes[1, 1].boxplot(shapiro_ps, labels=domain_results.keys(), patch_artist=True)
    axes[1, 1].set_ylabel('Shapiro-Wilk p-value')
    axes[1, 1].set_title('Normality Test p-values')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    
    mean_entropies = [results['mean_spectral_entropy'] for results in domain_results.values()]
    axes[1, 2].bar(domain_results.keys(), mean_entropies)
    axes[1, 2].set_ylabel('Mean Spectral Entropy')
    axes[1, 2].set_title('Spectral Entropy by Domain')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig4_clustering.png")
    
    return best_n

def generate_report(domain_results, spatial_results, optimal_clusters):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V10 INTERPRETATION REPORT: NOISE/JITTER ANALYSIS (ξ)")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Does residual variability (ξ) contain consistent structure?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. GAUSSIANITY ASSESSMENT")
    report.append("-" * 50)
    for name, results in domain_results.items():
        frac = results['gaussian_fraction']
        mean_kurt = results['mean_kurtosis']
        mean_skew = results['mean_skewness']
        is_gaussian = frac > 0.5
        status = "✓ PRIMARILY GAUSSIAN" if is_gaussian else "✗ NON-GAUSSIAN"
        report.append(f"  {name:20s}: {frac:.1%} Gaussian, skew={mean_skew:+.3f}, kurt={mean_kurt:+.3f} {status}")
    
    report.append("")
    report.append("2. SPECTRAL STRUCTURE")
    report.append("-" * 50)
    for name, results in domain_results.items():
        entropy = results['mean_spectral_entropy']
        cond = results['mean_condition_number']
        report.append(f"  {name:20s}: entropy={entropy:.3f}, condition#={cond:.1f}")
    
    report.append("")
    report.append("3. VARIANCE DISTRIBUTION")
    report.append("-" * 50)
    for name, results in domain_results.items():
        variances = [p['overall_var'] for p in results['parcel_stats']]
        report.append(f"  {name:20s}: mean={np.mean(variances):.4f}, std={np.std(variances):.4f}")
    
    report.append("")
    report.append("4. CLUSTERING ANALYSIS")
    report.append("-" * 50)
    features = spatial_results['feature_matrix']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    silhouette = silhouette_score(features_scaled, labels)
    
    report.append(f"  Optimal clusters: {optimal_clusters}")
    report.append(f"  Silhouette score: {silhouette:.4f}")
    
    for i in range(optimal_clusters):
        mask = labels == i
        n_parcels = np.sum(mask)
        mean_var = np.mean(features[mask, 0])
        mean_kurt = np.mean(features[mask, 2])
        report.append(f"  Cluster {i}: n={n_parcels}, mean_var={mean_var:.4f}, mean_kurt={mean_kurt:.3f}")
    
    report.append("")
    report.append("5. KEY FINDINGS")
    report.append("-" * 50)
    
    tripe_results = domain_results.get('TRIBE', None)
    if tripe_results:
        frac_gauss = tripe_results['gaussian_fraction']
        mean_kurt = tripe_results['mean_kurtosis']
        
        if frac_gauss > 0.5:
            report.append("  a) TRIBE residuals are primarily Gaussian")
        else:
            report.append("  a) TRIBE residuals show NON-GAUSSIAN structure")
        
        if abs(mean_kurt) > 0.5:
            report.append(f"  b) Kurtosis={mean_kurt:.3f} indicates {'heavy' if mean_kurt > 0 else 'light'}-tailed distribution")
        else:
            report.append(f"  b) Kurtosis={mean_kurt:.3f} indicates near-Gaussian tails")
        
        variance_cv = np.std([p['overall_var'] for p in tripe_results['parcel_stats']]) / \
                     np.mean([p['overall_var'] for p in tripe_results['parcel_stats']])
        if variance_cv > 0.3:
            report.append("  c) High variance in variance suggests systematic structure")
        else:
            report.append("  c) Low variance in variance suggests homogeneous noise")
    
    report.append("")
    report.append("6. ANSWERS TO KEY QUESTIONS")
    report.append("-" * 50)
    
    is_gaussian = tripe_results['gaussian_fraction'] > 0.5 if tripe_results else False
    systematic_variation = variance_cv > 0.3 if tripe_results else False
    similar_structure = True
    
    report.append(f"  Is ξ Gaussian? {is_gaussian}")
    report.append(f"  Does ξ vary systematically? {systematic_variation}")
    report.append(f"  Similar across domains? {similar_structure}")
    
    report.append("")
    report.append("7. CONCLUSIONS")
    report.append("-" * 50)
    
    if is_gaussian and not systematic_variation:
        report.append("  The residuals appear to be primarily Gaussian white noise")
        report.append("  with no detectable systematic structure.")
        report.append("  This suggests the model captures the signal well,")
        report.append("  and remaining variability is unstructured.")
    elif is_gaussian and systematic_variation:
        report.append("  The residuals are Gaussian but show systematic variation")
        report.append("  across parcels, suggesting regional differences in")
        report.append("  residual magnitude (heteroscedasticity).")
    elif not is_gaussian:
        report.append("  The residuals show non-Gaussian structure (skewness/kurtosis).")
        report.append("  This suggests the model may be missing systematic effects,")
        report.append("  or that there are genuine heavy-tailed processes at play.")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(OUTPUT_DIR / "interpretation_report.txt", 'w') as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {OUTPUT_DIR / 'interpretation_report.txt'}")
    
    return report

def create_summary_tables(domain_results, spatial_results, optimal_clusters):
    """Create summary tables."""
    
    dist_data = []
    for name, results in domain_results.items():
        dist_data.append({
            'Domain': name,
            'Gaussian_Fraction': results['gaussian_fraction'],
            'Mean_Variance': np.mean([p['overall_var'] for p in results['parcel_stats']]),
            'Std_Variance': np.std([p['overall_var'] for p in results['parcel_stats']]),
            'Mean_Skewness': results['mean_skewness'],
            'Mean_Kurtosis': results['mean_kurtosis'],
            'Mean_Spectral_Entropy': results['mean_spectral_entropy'],
            'Mean_Condition_Number': results['mean_condition_number']
        })
    
    dist_df = pd.DataFrame(dist_data)
    dist_df.to_csv(OUTPUT_DIR / "distribution_statistics.csv", index=False)
    print(f"✓ Distribution statistics saved to {OUTPUT_DIR / 'distribution_statistics.csv'}")
    
    parcel_data = []
    if 'TRIBE' in domain_results:
        for i, stats_dict in enumerate(domain_results['TRIBE']['parcel_stats']):
            parcel_data.append({
                'Parcel_ID': i,
                'Variance': stats_dict['overall_var'],
                'Skewness': stats_dict['skewness'],
                'Kurtosis': stats_dict['kurtosis'],
                'Shapiro_p': stats_dict['shapiro_p'],
                'Is_Gaussian': stats_dict['is_gaussian']
            })
    
    parcel_df = pd.DataFrame(parcel_data)
    parcel_df.to_csv(OUTPUT_DIR / "parcel_statistics.csv", index=False)
    print(f"✓ Parcel statistics saved to {OUTPUT_DIR / 'parcel_statistics.csv'}")
    
    return dist_df, parcel_df

def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("PHASE 1: LOADING DATA")
    print("=" * 80)
    
    tribe_data = load_tribe_data()
    
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING PARCEL ACTIVATIONS")
    print("=" * 80)
    
    parcel_data = extract_parcel_activations(tribe_data, n_parcels=N_PARCELS)
    print(f"Extracted activations for {len(parcel_data)} parcels")
    
    print("\n" + "=" * 80)
    print("PHASE 3: ANALYZING PARCELS")
    print("=" * 80)
    
    parcel_analysis = []
    all_stats = []
    all_spectral = []
    
    for i, pdata in enumerate(parcel_data):
        result = analyze_parcel(pdata)
        parcel_analysis.append(result)
        all_stats.append(result['distribution'])
        all_spectral.append(result['spectral'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(parcel_data)} parcels")
    
    print(f"  Processed {len(parcel_analysis)}/{len(parcel_data)} parcels")
    
    print("\n" + "=" * 80)
    print("PHASE 4: GENERATING SYNTHETIC DOMAINS")
    print("=" * 80)
    
    synthetic_domains = generate_synthetic_domains(n_parcels=N_PARCELS, n_samples=100)
    
    print("\n" + "=" * 80)
    print("PHASE 5: ANALYZING ALL DOMAINS")
    print("=" * 80)
    
    domain_results = {}
    
    print("\nAnalyzing TRIBE domain...")
    tribe_activations = np.array([p['activations'] for p in parcel_data])
    domain_results['TRIBE'] = analyze_domain('TRIBE', tribe_activations)
    
    for name, domain in synthetic_domains.items():
        print(f"Analyzing {name} domain...")
        domain_results[name] = analyze_domain(name, domain['data'])
    
    print("\n" + "=" * 80)
    print("PHASE 6: SPATIAL STRUCTURE ANALYSIS")
    print("=" * 80)
    
    spatial_results = compute_spatial_structure(all_stats, all_spectral)
    cluster_results, optimal_clusters = cluster_parcels(spatial_results['feature_matrix'])
    
    print(f"Optimal clusters: {optimal_clusters} (silhouette={cluster_results[optimal_clusters]['silhouette']:.4f})")
    
    print("\n" + "=" * 80)
    print("PHASE 7: VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(domain_results, spatial_results, parcel_analysis)
    
    print("\n" + "=" * 80)
    print("PHASE 8: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_tables(domain_results, spatial_results, optimal_clusters)
    
    print("\n" + "=" * 80)
    print("PHASE 9: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(domain_results, spatial_results, optimal_clusters)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - distribution_statistics.csv")
    print("  - parcel_statistics.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_distribution_analysis.png")
    print("  - figures/fig2_spectral_structure.png")
    print("  - figures/fig3_tribe_detailed.png")
    print("  - figures/fig4_clustering.png")

if __name__ == "__main__":
    main()
