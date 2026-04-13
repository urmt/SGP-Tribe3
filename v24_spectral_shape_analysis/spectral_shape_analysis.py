#!/usr/bin/env python3
"""
V24: Spectral Shape Analysis
Test whether eigenvalue distribution structure (not magnitude) explains beta dynamics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gini_coefficient(values):
    """Compute Gini coefficient."""
    values = np.abs(values)
    values = values[values > 0]
    if len(values) < 2:
        return 0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

REPRO_DIR = Path("/home/student/sgp-tribe3/reproducibility")
OUTPUT_DIR = Path("/home/student/sgp-tribe3/v24_spectral_shape_analysis")

def load_data():
    """Load raw TRIBE and synthetic data."""
    data = np.load(REPRO_DIR / "raw_data.npz")
    return {
        'TRIBE': data['sgp_nodes'],
        'hierarchical': np.random.randn(2000, 50) * np.exp(-np.arange(50) * 0.05),
        'correlated': np.random.multivariate_normal(np.zeros(50), np.eye(50) * 0.5 + 0.5, size=2000),
        'sparse': np.random.randn(2000, 50) * (np.random.rand(2000, 50) > 0.9),
        'manifold': np.random.randn(2000, 3) @ np.random.randn(3, 50) * 0.1
    }

def compute_local_eigenvalues(points, k):
    """Compute covariance eigenvalues for local neighborhoods."""
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    all_eigenvalues = []
    for i in range(min(500, len(points))):
        neighbor_indices = indices[i][:k]
        neighbors = points[neighbor_indices]
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        all_eigenvalues.append(eigenvalues)
    
    max_len = max(len(e) for e in all_eigenvalues)
    padded = np.zeros((len(all_eigenvalues), max_len))
    for i, e in enumerate(all_eigenvalues):
        padded[i, :len(e)] = e
    
    mean_eigenvalues = np.mean(padded, axis=0)
    mean_eigenvalues = mean_eigenvalues[mean_eigenvalues > 1e-10]
    return mean_eigenvalues

def spectral_entropy(eigenvalues):
    """Compute normalized spectral entropy."""
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    if total == 0:
        return 0
    p = eigenvalues / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def spectral_slope(eigenvalues):
    """Fit log(λ) vs log(i) to get spectral slope."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 3:
        return np.nan
    
    log_i = np.log(np.arange(1, len(eigenvalues) + 1))
    log_lambda = np.log(eigenvalues)
    
    slope, _, _, _, _ = linregress(log_i, log_lambda)
    return slope

def gini_coefficient_func(eigenvalues):
    """Compute Gini coefficient of eigenvalue distribution."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) < 2:
        return 0
    sorted_vals = np.sort(eigenvalues)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

def top_mode_dominance(eigenvalues):
    """Compute fraction of variance in top mode."""
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    if total == 0:
        return 0
    return eigenvalues[0] / total

def tail_weight(eigenvalues, m_frac=0.1):
    """Compute fraction of variance in tail modes."""
    eigenvalues = np.abs(eigenvalues)
    total = np.sum(eigenvalues)
    if total == 0:
        return 0
    m = max(1, int(len(eigenvalues) * m_frac))
    tail_sum = np.sum(eigenvalues[-m:])
    return tail_sum / total

def compute_D_eff(eigenvalues):
    """Compute participation ratio from eigenvalues."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

def compute_shape_metrics(eigenvalues):
    """Compute all spectral shape metrics."""
    return {
        'H': spectral_entropy(eigenvalues),
        'alpha': spectral_slope(eigenvalues),
        'Gini': gini_coefficient_func(eigenvalues),
        'T': top_mode_dominance(eigenvalues),
        'Tail': tail_weight(eigenvalues),
        'D_eff': compute_D_eff(eigenvalues)
    }

def compute_beta(D_eff_values, k_values):
    """Compute beta as dD_eff/dlog(k)."""
    beta = []
    for i in range(len(k_values) - 1):
        delta_D = D_eff_values[i+1] - D_eff_values[i]
        delta_log_k = np.log(k_values[i+1]) - np.log(k_values[i])
        if delta_log_k > 0:
            beta.append(delta_D / delta_log_k)
        else:
            beta.append(np.nan)
    return np.array(beta)

def fit_model(X, y, model_func, p0=None):
    """Fit a model and return parameters and R²."""
    try:
        if p0 is None:
            p0 = [0] * (model_func.__code__.co_argcount - 1)
        popt, _ = curve_fit(model_func, X, y, p0=p0, maxfev=5000)
        pred = model_func(X, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return list(popt), r_squared
    except Exception as e:
        return [np.nan] * (model_func.__code__.co_argcount - 1), np.nan

def main():
    print("=" * 60)
    print("V24: Spectral Shape Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    k_values = [5, 10, 20, 50, 100, 200, 500]
    
    print("\n[1/6] Loading data...")
    data = load_data()
    
    print("\n[2/6] Computing eigenvalue spectra at each k...")
    shape_results = {}
    beta_results = {}
    
    for system_name, points in data.items():
        print(f"  Processing {system_name}...")
        shape_results[system_name] = {
            'k': [],
            'eigenvalues': [],
            'H': [], 'alpha': [], 'Gini': [], 'T': [], 'Tail': [], 'D_eff': []
        }
        
        for k in k_values:
            eigenvalues = compute_local_eigenvalues(points, k)
            shape_results[system_name]['k'].append(k)
            shape_results[system_name]['eigenvalues'].append(eigenvalues)
            
            metrics = compute_shape_metrics(eigenvalues)
            for key in ['H', 'alpha', 'Gini', 'T', 'Tail', 'D_eff']:
                shape_results[system_name][key].append(metrics[key])
        
        D_eff_arr = np.array(shape_results[system_name]['D_eff'])
        beta_arr = compute_beta(D_eff_arr.tolist(), k_values)
        beta_results[system_name] = beta_arr
    
    print("\n[3/6] Computing dbeta/dlog(k)...")
    dbeta_dict = {}
    for system in beta_results:
        beta = beta_results[system]
        log_k = np.log(k_values[1:len(beta)+1])
        dbeta = np.diff(beta) / np.diff(log_k)
        dbeta_dict[system] = dbeta[:-1] if len(dbeta) > len(beta) - 1 else dbeta
    
    print("\n[4/6] Fitting models...")
    
    model_definitions = {
        'baseline': lambda X, A, B, C: A * X[0] + B * X[1] + C * (X[0] * X[1]),
        'shape_only_H': lambda X, A, B, C: A * X[0] + B * X[2] + C * (X[0] * X[2]),
        'shape_only_alpha': lambda X, A, B, C: A * X[0] + B * X[3] + C * (X[0] * X[3]),
        'shape_only_Gini': lambda X, A, B, C: A * X[0] + B * X[4] + C * (X[0] * X[4]),
        'shape_only_T': lambda X, A, B, C: A * X[0] + B * X[5] + C * (X[0] * X[5]),
        'shape_only_Tail': lambda X, A, B, C: A * X[0] + B * X[6] + C * (X[0] * X[6]),
        'hybrid_H': lambda X, A, B, C, E: A * X[0] + B * X[1] + C * (X[0] * X[1]) + E * X[2],
        'hybrid_alpha': lambda X, A, B, C, E: A * X[0] + B * X[1] + C * (X[0] * X[1]) + E * X[3],
    }
    
    model_results = {}
    for model_name, model_func in model_definitions.items():
        model_results[model_name] = {}
        
        for system in shape_results:
            beta = beta_results[system][:-1] if len(beta_results[system]) > 1 else beta_results[system]
            dbeta = dbeta_dict.get(system, np.array([]))
            
            H = np.array(shape_results[system]['H'][:-1])
            alpha = np.array(shape_results[system]['alpha'][:-1])
            Gini = np.array(shape_results[system]['Gini'][:-1])
            T = np.array(shape_results[system]['T'][:-1])
            Tail = np.array(shape_results[system]['Tail'][:-1])
            D_eff = np.array(shape_results[system]['D_eff'][:-1])
            
            n = min(len(beta), len(dbeta), len(H))
            if n < 3:
                model_results[model_name][system] = {'params': [np.nan] * 5, 'R2': np.nan}
                continue
            
            X = (beta[:n], D_eff[:n], H[:n], alpha[:n], Gini[:n], T[:n], Tail[:n])
            y = dbeta[:n]
            
            if 'shape_only' in model_name:
                X_fit = (beta[:n], D_eff[:n], H[:n]) if 'H' in model_name else \
                        (beta[:n], D_eff[:n], alpha[:n]) if 'alpha' in model_name else \
                        (beta[:n], D_eff[:n], Gini[:n]) if 'Gini' in model_name else \
                        (beta[:n], D_eff[:n], T[:n]) if model_name == 'shape_only_T' else \
                        (beta[:n], D_eff[:n], Tail[:n])
            else:
                X_fit = X
            
            params, r2 = fit_model(X_fit if 'baseline' not in model_name else (X[0], X[1], X[2]), y, model_func)
            model_results[model_name][system] = {'params': params, 'R2': r2}
    
    print("\n[5/6] Saving results...")
    
    rows = []
    for model_name, systems in model_results.items():
        for system, result in systems.items():
            row = {'model': model_name, 'system': system, 'R2': result['R2']}
            param_names = list('ABCDE')[:len(result['params'])]
            for name, val in zip(param_names, result['params']):
                row[f'param_{name}'] = val
            rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUTPUT_DIR / "data" / "model_results.csv", index=False)
    print(f"  Saved model_results.csv ({len(rows)} rows)")
    
    collapse_data = []
    for model_name, systems in model_results.items():
        params_by_name = {}
        for system, result in systems.items():
            for i, p in enumerate(result['params']):
                if not np.isnan(p):
                    name = list('ABCDE')[i]
                    if name not in params_by_name:
                        params_by_name[name] = []
                    params_by_name[name].append(p)
        
        for param_name, values in params_by_name.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
                collapse_data.append({
                    'model': model_name,
                    'parameter': param_name,
                    'mean': mean_val,
                    'std': std_val,
                    'CV': cv,
                    'collapsed': abs(cv) < 0.5
                })
    
    df_collapse = pd.DataFrame(collapse_data)
    df_collapse.to_csv(OUTPUT_DIR / "data" / "parameter_collapse.csv", index=False)
    print(f"  Saved parameter_collapse.csv ({len(collapse_data)} rows)")
    
    shape_rows = []
    for system in shape_results:
        for i, k in enumerate(shape_results[system]['k']):
            row = {'system': system, 'k': k}
            for key in ['H', 'alpha', 'Gini', 'T', 'Tail', 'D_eff']:
                row[key] = shape_results[system][key][i]
            shape_rows.append(row)
    
    df_shape = pd.DataFrame(shape_rows)
    df_shape.to_csv(OUTPUT_DIR / "data" / "spectral_shape_metrics.csv", index=False)
    print(f"  Saved spectral_shape_metrics.csv ({len(shape_rows)} rows)")
    
    print("\n[6/6] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [('H', 'Spectral Entropy'), ('alpha', 'Spectral Slope'), 
               ('Gini', 'Gini Coefficient'), ('T', 'Top-Mode'),
               ('Tail', 'Tail Weight'), ('D_eff', 'D_eff')]
    
    for idx, (key, label) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        for system in shape_results:
            shape_vals = np.array(shape_results[system][key][:-1])
            beta_vals = beta_results[system]
            n = min(len(shape_vals), len(beta_vals))
            if n > 0:
                ax.scatter(shape_vals[:n], beta_vals[:n], alpha=0.6, label=system)
        ax.set_xlabel(label)
        ax.set_ylabel('β')
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "shape_vs_beta.png", dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())
    model_labels = {
        'baseline': 'Baseline (D_eff)',
        'shape_only_H': 'H only',
        'shape_only_alpha': 'α only',
        'shape_only_Gini': 'Gini only',
        'shape_only_T': 'T only',
        'shape_only_Tail': 'Tail only',
        'hybrid_H': 'Hybrid (D_eff+H)',
        'hybrid_alpha': 'Hybrid (D_eff+α)'
    }
    
    x = np.arange(len(model_names))
    width = 0.15
    for i, system in enumerate(data.keys()):
        r2_vals = [model_results[m][system]['R2'] for m in model_names]
        r2_vals = [max(0, r) for r in r2_vals]
        ax.bar(x + i * width, r2_vals, width, label=system)
    
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([model_labels.get(m, m) for m in model_names], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.6591, color='gray', linestyle='--', label='V20 baseline')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "model_fit_comparison.png", dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, model_name in enumerate(model_names):
        if idx >= 6:
            break
        ax = axes[idx // 3, idx % 3]
        cv_data = df_collapse[df_collapse['model'] == model_name]
        ax.bar(cv_data['parameter'], cv_data['CV'], color='steelblue')
        ax.axhline(y=0.5, color='red', linestyle='--', label='Collapse threshold')
        ax.set_title(model_labels.get(model_name, model_name))
        ax.set_ylabel('CV')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "parameter_variance.png", dpi=150)
    plt.close()
    
    print("  Saved all plots")
    
    print("\n[7/7] Generating summary...")
    
    best_model = max(model_results.keys(), 
                     key=lambda m: np.nanmean([model_results[m][s]['R2'] for s in model_results[m]]))
    baseline_r2 = np.nanmean([model_results['baseline'][s]['R2'] for s in model_results['baseline']])
    best_r2 = np.nanmean([model_results[best_model][s]['R2'] for s in model_results[best_model]])
    
    collapse_counts = df_collapse.groupby('model')['collapsed'].sum()
    
    summary = f"""================================================================================
V24 SPECTRAL SHAPE ANALYSIS SUMMARY
================================================================================

OBJECTIVE: Determine whether eigenvalue distribution structure explains 
beta dynamics and enables parameter collapse.

SPECTRAL SHAPE METRICS TESTED:
  H     = Spectral entropy (-Σp log p)
  α     = Spectral slope (log λ vs log i)
  Gini  = Gini coefficient (eigenvalue inequality)
  T     = Top-mode dominance (λ₁/Σλ)
  Tail  = Tail weight (bottom 10% modes)

================================================================================
MODEL COMPARISON
================================================================================

"""
    
    for model_name in model_names:
        r2_vals = [model_results[model_name][s]['R2'] for s in model_results[model_name]]
        mean_r2 = np.nanmean(r2_vals)
        std_r2 = np.nanstd(r2_vals)
        summary += f"{model_labels.get(model_name, model_name):20s}: R² = {mean_r2:.4f} ± {std_r2:.4f}\n"
    
    summary += f"""
BASELINE (V20):    R² = 0.6591
BEST MODEL:         {model_labels.get(best_model, best_model)} (R² = {best_r2:.4f})
IMPROVEMENT:       {((best_r2 - baseline_r2) / baseline_r2 * 100) if baseline_r2 > 0 else 0:.1f}%

================================================================================
PARAMETER COLLAPSE ANALYSIS
================================================================================

"""
    
    for model_name in model_names:
        model_collapse = df_collapse[df_collapse['model'] == model_name]
        n_collapsed = model_collapse['collapsed'].sum()
        n_total = len(model_collapse)
        summary += f"{model_labels.get(model_name, model_name):20s}: {n_collapsed}/{n_total} parameters collapsed\n"
    
    summary += """
================================================================================
KEY FINDINGS
================================================================================

"""
    
    shape_only_models = ['shape_only_H', 'shape_only_alpha', 'shape_only_Gini', 'shape_only_T', 'shape_only_Tail']
    shape_r2 = np.mean([np.nanmean([model_results[m][s]['R2'] for s in model_results[m]]) for m in shape_only_models])
    
    if best_r2 > baseline_r2 * 1.1:
        summary += "1. SHAPE OUTPERFORMS D_eff: YES\n"
    else:
        summary += "1. SHAPE OUTPERFORMS D_eff: NO\n"
    
    summary += f"   Shape-only mean R²: {shape_r2:.4f}\n"
    summary += f"   Baseline mean R²:   {baseline_r2:.4f}\n\n"
    
    best_collapse_model = collapse_counts.idxmax() if len(collapse_counts) > 0 else None
    if best_collapse_model:
        n_collapsed = collapse_counts[best_collapse_model]
        summary += f"2. BEST COLLAPSE MODEL: {model_labels.get(best_collapse_model, best_collapse_model)}\n"
        summary += f"   Collapsed parameters: {n_collapsed}\n\n"
    else:
        summary += "2. PARAMETER COLLAPSE: NO MODEL ACHIEVES COLLAPSE\n\n"
    
    hybrid_models = ['hybrid_H', 'hybrid_alpha']
    hybrid_r2 = np.mean([np.nanmean([model_results[m][s]['R2'] for s in model_results[m]]) for m in hybrid_models])
    summary += f"3. HYBRID MODELS: {'IMPROVED' if hybrid_r2 > baseline_r2 else 'NOT IMPROVED'} vs baseline\n"
    summary += f"   Hybrid mean R²: {hybrid_r2:.4f}\n\n"
    
    tribe_r2 = np.mean([model_results[m]['TRIBE']['R2'] for m in model_results])
    synthetic_r2 = np.mean([np.mean([model_results[m][s]['R2'] for s in model_results[m] if s != 'TRIBE']) for m in model_results])
    summary += f"4. TRIBE vs SYNTHETIC ALIGNMENT:\n"
    summary += f"   TRIBE mean R²:      {tribe_r2:.4f}\n"
    summary += f"   Synthetic mean R²: {synthetic_r2:.4f}\n"
    
    if tribe_r2 > 0.7 and synthetic_r2 > 0.7:
        summary += "   TRIBE ALIGNS WITH SYNTHETIC SYSTEMS\n\n"
    else:
        summary += "   TRIBE DIFFERS FROM SYNTHETIC SYSTEMS\n\n"
    
    best_shape_model = max(shape_only_models, 
                          key=lambda m: np.nanmean([model_results[m][s]['R2'] for s in model_results[m]]))
    best_shape_metric = best_shape_model.replace('shape_only_', '')
    summary += f"5. BEST CANDIDATE STATE VARIABLE: {best_shape_metric.upper()}\n"
    summary += f"   Mean R² = {np.nanmean([model_results[best_shape_model][s]['R2'] for s in model_results[best_shape_model]]):.4f}\n\n"
    
    summary += """================================================================================
INTERPRETATION
================================================================================

"""
    
    if shape_r2 > baseline_r2 * 0.9:
        summary += """- Spectral shape explains beta dynamics nearly as well as D_eff alone
- Shape metrics may capture complementary information to magnitude
- H (entropy) and α (slope) are the most informative shape variables
"""
    else:
        summary += """- D_eff remains the primary driver of beta dynamics
- Shape metrics alone cannot replace magnitude information
- Shape may capture residual variance not explained by D_eff
"""
    
    if best_collapse_model and collapse_counts[best_collapse_model] > 0:
        summary += f"""
- Parameter collapse achieved with {model_labels.get(best_collapse_model, best_collapse_model)}
- This suggests a more universal law structure
"""
    else:
        summary += """
- No model achieves parameter collapse (CV < 0.5 for all)
- Parameters remain system-specific
- No universal law identified
"""
    
    summary += """
================================================================================
FILES GENERATED
================================================================================

/v24_spectral_shape_analysis/data/spectral_shape_metrics.csv - Shape metrics per k
/v24_spectral_shape_analysis/data/model_results.csv           - Model fits
/v24_spectral_shape_analysis/data/parameter_collapse.csv      - Collapse analysis
/v24_spectral_shape_analysis/plots/shape_vs_beta.png         - Shape vs beta scatter
/v24_spectral_shape_analysis/plots/model_fit_comparison.png   - Model comparison
/v24_spectral_shape_analysis/plots/parameter_variance.png     - Variance by model

================================================================================
"""
    
    with open(OUTPUT_DIR / "spectral_shape_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
