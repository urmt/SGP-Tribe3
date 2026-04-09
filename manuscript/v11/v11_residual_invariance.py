#!/usr/bin/env python3
"""
V11 Pipeline: Residual Invariance Test
Determine if residual structure is intrinsic or model-dependent
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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/student/sgp-tribe3/manuscript/v11")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.joinpath("figures").mkdir(exist_ok=True)

N_PARCELS = 400
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("V11 RESIDUAL INVARIANCE TEST")
print("Is residual structure intrinsic or model-dependent?")
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

def extract_parcel_activations_varied(tribe_data, n_parcels=400, n_samples_per_parcel=30):
    """Extract varied activation patterns with different configurations."""
    node_keys = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
                 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
    
    n_per_parcel = max(1, len(tribe_data) // n_parcels)
    
    all_activations = {k: [] for k in node_keys}
    
    for item in tribe_data:
        if 'sgp_nodes' in item:
            sgp = item['sgp_nodes']
            for k in node_keys:
                all_activations[k].append(sgp.get(k, 0.5))
    
    parcel_data = []
    for parcel_idx in range(n_parcels):
        base_idx = (parcel_idx * n_per_parcel) % len(tribe_data)
        
        parcel_activations = []
        for i in range(min(n_samples_per_parcel, n_per_parcel)):
            data_idx = (base_idx + i) % len(tribe_data)
            if 'sgp_nodes' in tribe_data[data_idx]:
                sgp = tribe_data[data_idx]['sgp_nodes']
                values = [sgp.get(k, 0.5) for k in node_keys]
                parcel_activations.append(values)
        
        if not parcel_activations:
            parcel_activations = [[0.5] * 9]
        
        parcel_data.append({
            'parcel_id': parcel_idx,
            'activations': np.array(parcel_activations)
        })
    
    return parcel_data, all_activations

def compute_residuals(activations):
    """Compute residuals: ξ = x - mean(x)"""
    mean = np.mean(activations, axis=0)
    residuals = activations - mean
    return residuals

def compute_distribution_stats(residuals):
    """Compute distribution statistics."""
    flat = residuals.flatten()
    
    stats_dict = {
        'mean': np.mean(flat),
        'variance': np.var(flat),
        'std': np.std(flat),
        'skewness': stats.skew(flat) if len(flat) >= 3 else 0,
        'kurtosis': stats.kurtosis(flat) if len(flat) >= 3 else 0,
    }
    
    n = len(flat)
    if n >= 3:
        _, shapiro_p = stats.shapiro(flat[:min(5000, n)])
        stats_dict['shapiro_p'] = shapiro_p
        stats_dict['is_gaussian'] = shapiro_p > 0.05
    else:
        stats_dict['shapiro_p'] = 1.0
        stats_dict['is_gaussian'] = True
    
    return stats_dict

def compute_spectral_stats(residuals):
    """Compute spectral statistics."""
    if residuals.shape[0] < 2:
        return {'entropy': 0, 'top3_variance': 0}
    
    cov = np.cov(residuals.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    
    total = np.sum(eigenvalues)
    if total > 0:
        explained = eigenvalues / total
    else:
        explained = np.zeros_like(eigenvalues)
    
    entropy = -np.sum(explained * np.log(explained + 1e-10))
    top3 = np.sum(explained[:3]) if len(explained) >= 3 else np.sum(explained)
    
    return {
        'entropy': entropy,
        'top3_variance': top3,
        'eigenvalues': eigenvalues
    }

def analyze_parcels(parcel_data, name=""):
    """Full analysis for all parcels."""
    results = {
        'name': name,
        'parcel_stats': [],
        'outlier_parcels': [],
        'global_stats': {}
    }
    
    all_variances = []
    all_kurtosis = []
    all_skewness = []
    
    for pdata in parcel_data:
        residuals = compute_residuals(pdata['activations'])
        dist_stats = compute_distribution_stats(residuals)
        spectral_stats = compute_spectral_stats(residuals)
        
        parcel_result = {
            'parcel_id': pdata['parcel_id'],
            **dist_stats,
            **spectral_stats
        }
        
        results['parcel_stats'].append(parcel_result)
        all_variances.append(dist_stats['variance'])
        all_kurtosis.append(dist_stats['kurtosis'])
        all_skewness.append(dist_stats['skewness'])
        
        kurtosis_threshold = 5
        if dist_stats['kurtosis'] > kurtosis_threshold:
            results['outlier_parcels'].append(pdata['parcel_id'])
    
    results['global_stats'] = {
        'mean_kurtosis': np.mean(all_kurtosis),
        'std_kurtosis': np.std(all_kurtosis),
        'mean_skewness': np.mean(all_skewness),
        'std_skewness': np.std(all_skewness),
        'gaussian_fraction': np.mean([p['is_gaussian'] for p in results['parcel_stats']]),
        'outlier_fraction': len(results['outlier_parcels']) / len(parcel_data)
    }
    
    return results

def split_data_analysis(tribe_data, split_name=""):
    """Split data into halves and analyze separately."""
    np.random.seed(RANDOM_SEED)
    n = len(tribe_data)
    indices = np.random.permutation(n)
    half = n // 2
    
    data1 = [tribe_data[i] for i in indices[:half]]
    data2 = [tribe_data[i] for i in indices[half:2*half]]
    
    print(f"  Split {split_name}: n1={len(data1)}, n2={len(data2)}")
    
    parcels1, _ = extract_parcel_activations_varied(data1, n_parcels=N_PARCELS, n_samples_per_parcel=20)
    parcels2, _ = extract_parcel_activations_varied(data2, n_parcels=N_PARCELS, n_samples_per_parcel=20)
    
    results1 = analyze_parcels(parcels1, name=f"{split_name}_Half1")
    results2 = analyze_parcels(parcels2, name=f"{split_name}_Half2")
    
    return results1, results2, parcels1, parcels2

def model_variation_analysis(all_activations):
    """Analyze with different node combinations."""
    node_keys = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
                 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
    
    configs = {
        'all_nodes': node_keys,
        'sensory_focus': ['G7_sensory', 'G2_wernicke', 'G1_broca'],
        'cognitive_focus': ['G4_pfc', 'G5_dmn', 'G3_tpj'],
        'limbic_focus': ['G6_limbic', 'G5_dmn', 'G8_atl'],
    }
    
    results = {}
    
    for config_name, selected_nodes in configs.items():
        print(f"  Analyzing config: {config_name}")
        
        activations_subset = {k: v for k, v in all_activations.items() if k in selected_nodes}
        
        parcel_data = []
        for parcel_idx in range(N_PARCELS):
            n_samples = len(list(activations_subset.values())[0]) // N_PARCELS
            parcel_activations = []
            for sample_idx in range(n_samples):
                idx = parcel_idx * n_samples + sample_idx
                if idx < len(list(activations_subset.values())[0]):
                    values = [activations_subset[k][idx] if idx < len(activations_subset[k]) else 0.5 
                             for k in selected_nodes]
                    parcel_activations.append(values)
            
            if not parcel_activations:
                parcel_activations = [[0.5] * len(selected_nodes)]
            
            parcel_data.append({
                'parcel_id': parcel_idx,
                'activations': np.array(parcel_activations)
            })
        
        results[config_name] = analyze_parcels(parcel_data, name=config_name)
    
    return results

def generate_language_embeddings(n_samples=1000, n_dims=9):
    """Generate synthetic language embedding data."""
    np.random.seed(RANDOM_SEED + 1)
    
    embeddings = {
        'gaussian': np.random.randn(n_samples, n_dims) * 0.3,
        'uniform': np.random.uniform(-1, 1, (n_samples, n_dims)),
        'heavy_tail': np.random.standard_t(df=3, size=(n_samples, n_dims)) * 0.3,
        'bimodal': np.concatenate([
            np.random.randn(n_samples//2, n_dims) * 0.2 - 0.5,
            np.random.randn(n_samples//2, n_dims) * 0.2 + 0.5
        ], axis=0)
    }
    
    return embeddings

def cross_domain_analysis(tribe_data, embeddings):
    """Cross-domain residual analysis."""
    print("\nCross-domain analysis...")
    
    tribe_parcels, _ = extract_parcel_activations_varied(tribe_data, n_parcels=N_PARCELS, n_samples_per_parcel=30)
    tribe_results = analyze_parcels(tribe_parcels, name="TRIBE")
    
    results = {'TRIBE': tribe_results}
    
    for emb_name, emb_data in embeddings.items():
        print(f"  Analyzing: {emb_name}")
        
        parcel_data = []
        samples_per_parcel = len(emb_data) // N_PARCELS
        
        for parcel_idx in range(N_PARCELS):
            start_idx = parcel_idx * samples_per_parcel
            end_idx = start_idx + samples_per_parcel
            parcel_activations = emb_data[start_idx:end_idx]
            
            if len(parcel_activations) < 2:
                parcel_activations = np.random.randn(10, emb_data.shape[1]) * 0.3
            
            parcel_data.append({
                'parcel_id': parcel_idx,
                'activations': parcel_activations
            })
        
        results[emb_name] = analyze_parcels(parcel_data, name=emb_name)
    
    return results

def compute_outlier_overlap(results1, results2):
    """Compute overlap between outlier parcels."""
    outliers1 = set(results1['outlier_parcels'])
    outliers2 = set(results2['outlier_parcels'])
    
    overlap = outliers1 & outliers2
    union = outliers1 | outliers2
    
    if len(union) > 0:
        jaccard = len(overlap) / len(union)
    else:
        jaccard = 0
    
    return {
        'overlap_count': len(overlap),
        'overlap_fraction_1': len(overlap) / max(1, len(outliers1)),
        'overlap_fraction_2': len(overlap) / max(1, len(outliers2)),
        'jaccard': jaccard,
        'outliers1': outliers1,
        'outliers2': outliers2,
        'common_outliers': overlap
    }

def compute_stability_metrics(all_results):
    """Compute stability metrics across analyses."""
    metrics = {
        'kurtosis_values': [],
        'outlier_parcel_sets': [],
        'gaussian_fractions': []
    }
    
    for name, res in all_results.items():
        metrics['kurtosis_values'].append(res['global_stats']['mean_kurtosis'])
        metrics['outlier_parcel_sets'].append(set(res['outlier_parcels']))
        metrics['gaussian_fractions'].append(res['global_stats']['gaussian_fraction'])
    
    all_outlier_union = set()
    for s in metrics['outlier_parcel_sets']:
        all_outlier_union |= s
    
    common_outliers = set.intersection(*metrics['outlier_parcel_sets']) if metrics['outlier_parcel_sets'] else set()
    
    return {
        'kurtosis_mean': np.mean(metrics['kurtosis_values']),
        'kurtosis_std': np.std(metrics['kurtosis_values']),
        'kurtosis_cv': np.std(metrics['kurtosis_values']) / max(1, np.mean(metrics['kurtosis_values'])),
        'gaussian_fraction_mean': np.mean(metrics['gaussian_fractions']),
        'gaussian_fraction_std': np.std(metrics['gaussian_fractions']),
        'total_unique_outliers': len(all_outlier_union),
        'common_outliers': common_outliers,
        'common_outlier_fraction': len(common_outliers) / max(1, N_PARCELS)
    }

def create_visualizations(all_results, split_results, model_variation_results, cross_domain_results, stability):
    """Create all visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig_dir = OUTPUT_DIR / "figures"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    all_kurtosis = []
    all_names = []
    for name, res in all_results.items():
        kurt_vals = [p['kurtosis'] for p in res['parcel_stats']]
        all_kurtosis.append(kurt_vals)
        all_names.append(name)
    
    bp = axes[0, 0].boxplot(all_kurtosis, labels=all_names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_kurtosis)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 0].set_ylabel('Kurtosis')
    axes[0, 0].set_title('Kurtosis Distribution by Analysis')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Outlier threshold')
    axes[0, 0].legend()
    
    gaussian_fracs = [res['global_stats']['gaussian_fraction'] for res in all_results.values()]
    axes[0, 1].bar(all_names, gaussian_fracs)
    axes[0, 1].set_ylabel('Gaussian Fraction')
    axes[0, 1].set_title('Normality by Analysis')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    outlier_counts = [len(res['outlier_parcels']) for res in all_results.values()]
    axes[0, 2].bar(all_names, outlier_counts)
    axes[0, 2].set_ylabel('Number of Outlier Parcels')
    axes[0, 2].set_title('Outlier Parcels by Analysis')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    split_names = list(split_results.keys())
    kurtosis_by_split = [all_results.get(name, {}).get('global_stats', {}).get('mean_kurtosis', 0) 
                         for name in split_names]
    
    if len(split_names) >= 2:
        overlap_data = compute_outlier_overlap(all_results.get(split_names[0], {'outlier_parcels': []}),
                                              all_results.get(split_names[1], {'outlier_parcels': []}))
        
        axes[1, 0].bar(['Overlap Count', 'Outliers 1', 'Outliers 2'], 
                      [overlap_data['overlap_count'], 
                       len(overlap_data['outliers1']),
                       len(overlap_data['outliers2'])])
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f"Outlier Overlap (Jaccard={overlap_data['jaccard']:.3f})")
    
    axes[1, 1].bar(['Mean Kurtosis', 'Kurt CV', 'Common Outliers %'],
                   [stability['kurtosis_mean'], stability['kurtosis_cv'], 
                    stability['common_outlier_fraction'] * 100])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Stability Metrics')
    
    if stability['common_outliers']:
        common_list = list(stability['common_outliers'])[:20]
        axes[1, 2].hist(common_list, bins=20, color='steelblue', alpha=0.7)
        axes[1, 2].set_xlabel('Parcel ID')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title(f'Common Outlier Parcel IDs (n={len(stability["common_outliers"])})')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_invariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_invariance_analysis.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    all_parcels = []
    for name, res in all_results.items():
        for p in res['parcel_stats']:
            p['analysis'] = name
            all_parcels.append(p)
    
    df = pd.DataFrame(all_parcels)
    
    pivot_kurt = df.pivot(index='parcel_id', columns='analysis', values='kurtosis')
    
    sns.heatmap(pivot_kurt, cmap='RdBu_r', center=0, ax=axes[0, 0], 
                vmin=-5, vmax=15, cbar_kws={'label': 'Kurtosis'})
    axes[0, 0].set_title('Kurtosis Heatmap (Parcels × Analyses)')
    axes[0, 0].set_xlabel('Analysis')
    axes[0, 0].set_ylabel('Parcel ID')
    
    pivot_skew = df.pivot(index='parcel_id', columns='analysis', values='skewness')
    
    sns.heatmap(pivot_skew, cmap='RdBu_r', center=0, ax=axes[0, 1], 
                vmin=-3, vmax=3, cbar_kws={'label': 'Skewness'})
    axes[0, 1].set_title('Skewness Heatmap (Parcels × Analyses)')
    axes[0, 1].set_xlabel('Analysis')
    axes[0, 1].set_ylabel('Parcel ID')
    
    parcel_ids = pivot_kurt.index[:100]
    for col in pivot_kurt.columns:
        axes[1, 0].plot(pivot_kurt.loc[parcel_ids, col].values, alpha=0.5, label=col)
    axes[1, 0].set_xlabel('Parcel Index (first 100)')
    axes[1, 0].set_ylabel('Kurtosis')
    axes[1, 0].set_title('Kurtosis Trajectories')
    axes[1, 0].legend(fontsize=8)
    
    correlations = pivot_kurt.corr()
    sns.heatmap(correlations, annot=True, fmt='.3f', ax=axes[1, 1], cmap='coolwarm', center=0)
    axes[1, 1].set_title('Kurtosis Correlation Across Analyses')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_heatmap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_heatmap_analysis.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    split1_outliers = set(all_results.get('Half1', {}).get('outlier_parcels', []))
    split2_outliers = set(all_results.get('Half2', {}).get('outlier_parcels', []))
    
    venn_data = [
        len(split1_outliers - split2_outliers),
        len(split1_outliers & split2_outliers),
        len(split2_outliers - split1_outliers)
    ]
    
    axes[0, 0].bar(['Half1 Only', 'Both', 'Half2 Only'], venn_data, color=['steelblue', 'coral', 'seagreen'])
    axes[0, 0].set_ylabel('Number of Outlier Parcels')
    axes[0, 0].set_title('Outlier Parcel Overlap Between Data Halves')
    
    kurt1 = [all_results.get('Half1', {}).get('global_stats', {}).get('mean_kurtosis', 0)]
    kurt2 = [all_results.get('Half2', {}).get('global_stats', {}).get('mean_kurtosis', 0)]
    
    x = np.arange(2)
    axes[0, 1].bar(x, [kurt1[0], kurt2[0]], width=0.5, color=['steelblue', 'coral'])
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Half1', 'Half2'])
    axes[0, 1].set_ylabel('Mean Kurtosis')
    axes[0, 1].set_title('Kurtosis Stability Across Data Splits')
    axes[0, 1].axhline(y=np.mean([kurt1[0], kurt2[0]]), color='k', linestyle='--', alpha=0.5)
    
    cross_domain_kurt = {name: res['global_stats']['mean_kurtosis'] 
                         for name, res in cross_domain_results.items()}
    cross_domain_names = list(cross_domain_kurt.keys())
    cross_domain_vals = list(cross_domain_kurt.values())
    
    colors_cd = plt.cm.Set3(np.linspace(0, 1, len(cross_domain_names)))
    axes[1, 0].bar(cross_domain_names, cross_domain_vals, color=colors_cd)
    axes[1, 0].set_ylabel('Mean Kurtosis')
    axes[1, 0].set_title('Cross-Domain Kurtosis Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    if 'TRIBE' in cross_domain_results:
        tribe_kurt = [p['kurtosis'] for p in cross_domain_results['TRIBE']['parcel_stats']]
    else:
        tribe_kurt = []
    
    axes[1, 1].hist(tribe_kurt, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(tribe_kurt), color='r', linestyle='--', 
                       label=f'Mean={np.mean(tribe_kurt):.2f}')
    axes[1, 1].axvline(x=np.median(tribe_kurt), color='g', linestyle='--', 
                       label=f'Median={np.median(tribe_kurt):.2f}')
    axes[1, 1].set_xlabel('Kurtosis')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('TRIBE: Kurtosis Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_split_domain_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_split_domain_analysis.png")
    
    return

def create_summary_tables(all_results, split_results, cross_domain_results, stability):
    """Create summary tables."""
    
    summary_data = []
    for name, res in all_results.items():
        summary_data.append({
            'Analysis': name,
            'Mean_Kurtosis': res['global_stats']['mean_kurtosis'],
            'Std_Kurtosis': res['global_stats']['std_kurtosis'],
            'Mean_Skewness': res['global_stats']['mean_skewness'],
            'Gaussian_Fraction': res['global_stats']['gaussian_fraction'],
            'N_Outliers': len(res['outlier_parcels']),
            'Outlier_Fraction': res['global_stats']['outlier_fraction']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "invariance_summary.csv", index=False)
    print(f"✓ Summary saved to {OUTPUT_DIR / 'invariance_summary.csv'}")
    
    parcel_data = []
    for name, res in all_results.items():
        for p in res['parcel_stats']:
            parcel_data.append({
                'Parcel_ID': p['parcel_id'],
                'Analysis': name,
                'Kurtosis': p['kurtosis'],
                'Skewness': p['skewness'],
                'Variance': p['variance'],
                'Is_Outlier': p['parcel_id'] in res['outlier_parcels']
            })
    
    parcel_df = pd.DataFrame(parcel_data)
    parcel_df.to_csv(OUTPUT_DIR / "parcel_comparison.csv", index=False)
    print(f"✓ Parcel comparison saved to {OUTPUT_DIR / 'parcel_comparison.csv'}")
    
    return summary_df, parcel_df

def generate_report(all_results, split_results, model_variation_results, cross_domain_results, stability):
    """Generate interpretation report."""
    print("\n" + "=" * 80)
    print("INTERPRETATION REPORT")
    print("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("V11 INTERPRETATION REPORT: RESIDUAL INVARIANCE TEST")
    report.append("=" * 80)
    report.append("")
    
    report.append("QUESTION: Is residual structure intrinsic or model-dependent?")
    report.append("-" * 70)
    
    report.append("")
    report.append("1. SUCCESS CRITERIA ASSESSMENT")
    report.append("-" * 50)
    
    report.append(f"  a) Same parcels remain outliers: {len(stability['common_outliers'])} parcels")
    report.append(f"     - Common outliers: {len(stability['common_outliers'])}/{N_PARCELS} ({stability['common_outlier_fraction']*100:.1f}%)")
    
    kurt_cv_threshold = 0.3
    kurt_stable = stability['kurtosis_cv'] < kurt_cv_threshold
    report.append(f"  b) Similar kurtosis across runs: {'YES' if kurt_stable else 'NO'}")
    report.append(f"     - Mean kurtosis: {stability['kurtosis_mean']:.3f}")
    report.append(f"     - CV (stability): {stability['kurtosis_cv']:.3f}")
    
    report.append("")
    report.append("2. OUTLIER PARCEL ANALYSIS")
    report.append("-" * 50)
    
    all_outlier_counts = {name: len(res['outlier_parcels']) for name, res in all_results.items()}
    report.append(f"  Outlier counts by analysis:")
    for name, count in all_outlier_counts.items():
        report.append(f"    {name}: {count} parcels ({count/N_PARCELS*100:.1f}%)")
    
    if len(stability['common_outliers']) > 0:
        report.append(f"\n  Consistent outliers (appear in multiple analyses):")
        for pid in sorted(list(stability['common_outliers'])[:10]):
            report.append(f"    Parcel {pid}")
        if len(stability['common_outliers']) > 10:
            report.append(f"    ... and {len(stability['common_outliers']) - 10} more")
    
    report.append("")
    report.append("3. STABILITY METRICS")
    report.append("-" * 50)
    report.append(f"  Kurtosis mean: {stability['kurtosis_mean']:.3f}")
    report.append(f"  Kurtosis std: {stability['kurtosis_std']:.3f}")
    report.append(f"  Kurtosis CV: {stability['kurtosis_cv']:.3f}")
    report.append(f"  Gaussian fraction mean: {stability['gaussian_fraction_mean']:.3f}")
    report.append(f"  Gaussian fraction std: {stability['gaussian_fraction_std']:.3f}")
    report.append(f"  Unique outlier parcels: {stability['total_unique_outliers']}")
    
    report.append("")
    report.append("4. CROSS-DOMAIN COMPARISON")
    report.append("-" * 50)
    for name, res in cross_domain_results.items():
        report.append(f"  {name}: kurtosis={res['global_stats']['mean_kurtosis']:.3f}, "
                     f"gaussian={res['global_stats']['gaussian_fraction']:.1%}")
    
    report.append("")
    report.append("5. KEY FINDINGS")
    report.append("-" * 50)
    
    if len(stability['common_outliers']) > N_PARCELS * 0.1:
        report.append("  ✓ Strong intrinsic structure detected")
        report.append("    Many parcels consistently show heavy-tailed residuals")
        intrinsic = True
    elif len(stability['common_outliers']) > 0:
        report.append("  ~ Partial intrinsic structure")
        report.append("    Some parcels consistently show heavy-tailed residuals")
        intrinsic = "partial"
    else:
        report.append("  ✗ No consistent intrinsic structure")
        report.append("    Heavy-tailed residuals are model-dependent")
        intrinsic = False
    
    if stability['kurtosis_cv'] < 0.3:
        report.append("  ✓ Kurtosis is stable across analyses")
    else:
        report.append("  ✗ Kurtosis varies significantly across analyses")
    
    report.append("")
    report.append("6. CONCLUSIONS")
    report.append("-" * 50)
    
    if intrinsic and stability['kurtosis_cv'] < 0.3:
        report.append("  The residual structure appears to be INTRINSIC.")
        report.append("  Heavy-tailed residuals in specific parcels are consistent")
        report.append("  across different data splits and model configurations,")
        report.append("  suggesting this is a genuine property of those regions,")
        report.append("  not an artifact of the analysis method.")
    elif intrinsic:
        report.append("  The residual structure is PARTIALLY INTRINSIC.")
        report.append("  While some parcels consistently show heavy tails,")
        report.append("  the magnitude varies across analyses, suggesting")
        report.append("  both intrinsic and model-dependent factors.")
    else:
        report.append("  The residual structure appears to be MODEL-DEPENDENT.")
        report.append("  Heavy-tailed residuals are not consistent across analyses,")
        report.append("  suggesting they are artifacts of the specific")
        report.append("  model or data configuration rather than intrinsic.")
    
    report.append("")
    report.append("7. IMPLICATIONS")
    report.append("-" * 50)
    
    if intrinsic:
        report.append("  - These parcels may require different modeling approaches")
        report.append("  - The heavy tails could reflect genuine extreme events")
        report.append("  - Consider robust statistical methods for these regions")
    else:
        report.append("  - Heavy tails are likely data/model artifacts")
        report.append("  - Standard modeling approaches should suffice")
        report.append("  - Focus on improving model fit rather than adaptation")
    
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
    print("PHASE 2: SPLIT DATA ANALYSIS")
    print("=" * 80)
    
    all_results = {}
    
    print("Analyzing full dataset...")
    full_parcels, all_activations = extract_parcel_activations_varied(
        tribe_data, n_parcels=N_PARCELS, n_samples_per_parcel=30)
    all_results['Full'] = analyze_parcels(full_parcels, name="Full")
    
    print("\nSplitting data into halves...")
    results1, results2, parcels1, parcels2 = split_data_analysis(tribe_data, split_name="Half")
    all_results['Half1'] = results1
    all_results['Half2'] = results2
    
    print("\n" + "=" * 80)
    print("PHASE 3: MODEL VARIATION ANALYSIS")
    print("=" * 80)
    
    model_variation_results = model_variation_analysis(all_activations)
    for name, res in model_variation_results.items():
        all_results[f'Model_{name}'] = res
    
    print("\n" + "=" * 80)
    print("PHASE 4: CROSS-DOMAIN ANALYSIS")
    print("=" * 80)
    
    embeddings = generate_language_embeddings(n_samples=10000, n_dims=9)
    cross_domain_results = cross_domain_analysis(tribe_data, embeddings)
    
    print("\n" + "=" * 80)
    print("PHASE 5: COMPUTING STABILITY METRICS")
    print("=" * 80)
    
    stability = compute_stability_metrics(all_results)
    print(f"Common outliers: {len(stability['common_outliers'])}")
    print(f"Kurtosis CV: {stability['kurtosis_cv']:.3f}")
    
    print("\n" + "=" * 80)
    print("PHASE 6: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(all_results, results1, model_variation_results, cross_domain_results, stability)
    
    print("\n" + "=" * 80)
    print("PHASE 7: SUMMARY TABLES")
    print("=" * 80)
    
    create_summary_tables(all_results, results1, cross_domain_results, stability)
    
    print("\n" + "=" * 80)
    print("PHASE 8: INTERPRETATION REPORT")
    print("=" * 80)
    
    generate_report(all_results, results1, model_variation_results, cross_domain_results, stability)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - invariance_summary.csv")
    print("  - parcel_comparison.csv")
    print("  - interpretation_report.txt")
    print("  - figures/fig1_invariance_analysis.png")
    print("  - figures/fig2_heatmap_analysis.png")
    print("  - figures/fig3_split_domain_analysis.png")

if __name__ == "__main__":
    main()
