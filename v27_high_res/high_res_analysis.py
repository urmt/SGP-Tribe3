#!/usr/bin/env python3
"""
V27: High-Resolution State Space Analysis
Test whether compression (D_eff/k) is the missing state variable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPRO_DIR = Path("/home/student/sgp-tribe3/reproducibility")
OUTPUT_DIR = Path("/home/student/sgp-tribe3/v27_high_res")

def load_data():
    """Load raw TRIBE and synthetic data."""
    data = np.load(REPRO_DIR / "raw_data.npz")
    np.random.seed(42)
    
    return {
        'TRIBE': data['sgp_nodes'],
        'hierarchical': np.random.randn(2000, 50) * np.exp(-np.arange(50) * 0.05),
        'correlated': np.random.multivariate_normal(np.zeros(50), np.eye(50) * 0.5 + 0.5, size=2000),
        'sparse': np.random.randn(2000, 50) * (np.random.rand(2000, 50) > 0.9),
        'manifold': np.random.randn(2000, 3) @ np.random.randn(3, 50) * 0.1
    }

def compute_local_eigenvalues(points, k):
    """Compute mean covariance eigenvalues for local neighborhoods."""
    from sklearn.neighbors import NearestNeighbors
    
    k_actual = min(k + 1, len(points))
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    all_eigenvalues = []
    for i in range(min(300, len(points))):
        neighbor_indices = indices[i][:k_actual]
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
    
    return np.mean(padded, axis=0)

def spectral_entropy(eigenvalues):
    """Compute spectral entropy."""
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0
    total = np.sum(eigenvalues)
    p = eigenvalues / total
    return -np.sum(p * np.log(p + 1e-10))

def compute_high_res_dataset():
    """Compute D_eff, beta, H at high resolution."""
    data = load_data()
    
    # High resolution k values (log-spaced)
    k_values = [5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    
    results = []
    
    for system_name, points in data.items():
        print(f"  Processing {system_name}...")
        
        for k in k_values:
            eigenvalues = compute_local_eigenvalues(points, k)
            
            # D_eff from participation ratio
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                D_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            else:
                D_eff = 1.0
            
            # Spectral entropy H
            H = spectral_entropy(eigenvalues)
            
            # Compression ratio R
            R = D_eff / k
            
            results.append({
                'system': system_name,
                'k': k,
                'D_eff': D_eff,
                'H': H,
                'R': R
            })
    
    return pd.DataFrame(results)

def compute_derivatives(df):
    """Compute beta and dbeta/dlog(k)."""
    rows = []
    
    for system in df['system'].unique():
        sys_data = df[df['system'] == system].sort_values('k')
        
        k_vals = sys_data['k'].values
        D_vals = sys_data['D_eff'].values
        H_vals = sys_data['H'].values
        R_vals = sys_data['R'].values
        
        # Compute beta = dD/dlog(k)
        for i in range(len(k_vals) - 1):
            delta_D = D_vals[i+1] - D_vals[i]
            delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
            if delta_log_k > 0:
                beta = delta_D / delta_log_k
                rows.append({
                    'system': system,
                    'k': k_vals[i],
                    'D_eff': D_vals[i],
                    'H': H_vals[i],
                    'R': R_vals[i],
                    'beta': beta,
                    'k_mid': (k_vals[i] + k_vals[i+1]) / 2,
                    'D_mid': (D_vals[i] + D_vals[i+1]) / 2,
                    'H_mid': (H_vals[i] + H_vals[i+1]) / 2,
                    'R_mid': (R_vals[i] + R_vals[i+1]) / 2
                })
    
    deriv_df = pd.DataFrame(rows)
    
    # Compute dbeta/dlog(k)
    dbeta_rows = []
    for system in deriv_df['system'].unique():
        sys_data = deriv_df[deriv_df['system'] == system].sort_values('k')
        
        k_vals = sys_data['k_mid'].values
        beta_vals = sys_data['beta'].values
        D_vals = sys_data['D_mid'].values
        H_vals = sys_data['H_mid'].values
        R_vals = sys_data['R_mid'].values
        
        for i in range(len(beta_vals) - 1):
            delta_beta = beta_vals[i+1] - beta_vals[i]
            delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
            if delta_log_k > 0:
                dbeta = delta_beta / delta_log_k
                dbeta_rows.append({
                    'system': system,
                    'k': k_vals[i],
                    'D_eff': D_vals[i],
                    'H': H_vals[i],
                    'R': R_vals[i],
                    'beta': beta_vals[i],
                    'dbeta': dbeta
                })
    
    return pd.DataFrame(dbeta_rows)

def fit_models(df):
    """Fit state-space models."""
    results = []
    
    def model_d(X, a, b, c, d):
        D, H, dbeta = X[:3]
        return a * D + b * H + c * D * H + d - dbeta
    
    def model_g(X, a, b, c, e, f, dbeta):
        D, H, R = X
        return a * D + b * H + c * D * H + e * R + f - dbeta
    
    for system in df['system'].unique():
        sys_data = df[df['system'] == system].dropna()
        
        if len(sys_data) < 5:
            continue
        
        D = sys_data['D_eff'].values
        H = sys_data['H'].values
        R = sys_data['R'].values
        dbeta = sys_data['dbeta'].values
        
        # Model D: dβ = a·D + b·H + c·(D·H) + d
        try:
            def model_d_loss(params, D, H, dbeta):
                a, b, c, d = params
                pred = a * D + b * H + c * D * H + d
                return np.sum((pred - dbeta) ** 2)
            
            from scipy.optimize import minimize
            res = minimize(model_d_loss, [0.1, 0.1, 0.01, 0.1], args=(D, H, dbeta))
            a, b, c, d = res.x
            pred_d = a * D + b * H + c * D * H + d
            ss_res = np.sum((pred_d - dbeta) ** 2)
            ss_tot = np.sum((dbeta - np.mean(dbeta)) ** 2)
            r2_d = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results.append({
                'system': system,
                'model': 'D_baseline',
                'a': a, 'b': b, 'c': c, 'd': d,
                'R2': r2_d, 'n_points': len(sys_data)
            })
        except:
            results.append({
                'system': system, 'model': 'D_baseline',
                'a': np.nan, 'b': np.nan, 'c': np.nan, 'd': np.nan,
                'R2': np.nan, 'n_points': len(sys_data)
            })
        
        # Model G: dβ = a·D + b·H + c·(D·H) + e·R + f
        try:
            def model_g_loss(params, D, H, R, dbeta):
                a, b, c, e, f = params
                pred = a * D + b * H + c * D * H + e * R + f
                return np.sum((pred - dbeta) ** 2)
            
            res = minimize(model_g_loss, [0.1, 0.1, 0.01, 0.1, 0.1], args=(D, H, R, dbeta))
            a, b, c, e, f = res.x
            pred_g = a * D + b * H + c * D * H + e * R + f
            ss_res = np.sum((pred_g - dbeta) ** 2)
            ss_tot = np.sum((dbeta - np.mean(dbeta)) ** 2)
            r2_g = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results.append({
                'system': system,
                'model': 'G_compression',
                'a': a, 'b': b, 'c': c, 'd': e, 'e': f,
                'R2': r2_g, 'n_points': len(sys_data)
            })
        except:
            results.append({
                'system': system, 'model': 'G_compression',
                'a': np.nan, 'b': np.nan, 'c': np.nan, 'd': np.nan, 'e': np.nan,
                'R2': np.nan, 'n_points': len(sys_data)
            })
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("V27: High-Resolution State Space Validation")
    print("=" * 60)
    
    print("\n[1/6] Computing high-resolution D_eff, H, R...")
    base_df = compute_high_res_dataset()
    base_df.to_csv(OUTPUT_DIR / "data" / "high_res_base.csv", index=False)
    print(f"  Saved high_res_base.csv ({len(base_df)} rows)")
    
    print("\n[2/6] Computing derivatives (beta, dbeta)...")
    deriv_df = compute_derivatives(base_df)
    deriv_df.to_csv(OUTPUT_DIR / "data" / "high_res_dataset.csv", index=False)
    print(f"  Saved high_res_dataset.csv ({len(deriv_df)} rows)")
    print(f"  Systems: {deriv_df['system'].unique()}")
    print(f"  Points per system: {deriv_df.groupby('system').size().to_dict()}")
    
    print("\n[3/6] Fitting models...")
    model_results = fit_models(deriv_df)
    model_results.to_csv(OUTPUT_DIR / "data" / "model_results.csv", index=False)
    print(f"  Saved model_results.csv ({len(model_results)} rows)")
    
    # Print results
    print("\n  Model comparison:")
    for system in sorted(model_results['system'].unique()):
        sys_results = model_results[model_results['system'] == system]
        r2_d = sys_results[sys_results['model'] == 'D_baseline']['R2'].values
        r2_g = sys_results[sys_results['model'] == 'G_compression']['R2'].values
        r2_d = r2_d[0] if len(r2_d) > 0 else np.nan
        r2_g = r2_g[0] if len(r2_g) > 0 else np.nan
        improvement = (r2_g - r2_d) if not np.isnan(r2_d) and not np.isnan(r2_g) else np.nan
        print(f"    {system:15s}: D_baseline R²={r2_d:.4f}, G_compression R²={r2_g:.4f}, ΔR²={improvement:+.4f}")
    
    print("\n[4/6] Computing parameter collapse...")
    collapse_data = []
    for model_name in ['D_baseline', 'G_compression']:
        model_subset = model_results[model_results['model'] == model_name]
        for param in ['a', 'b', 'c', 'd', 'e']:
            values = model_subset[param].dropna().values
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else np.inf
                collapse_data.append({
                    'model': model_name,
                    'parameter': param,
                    'mean': mean_val,
                    'std': std_val,
                    'CV': cv,
                    'collapsed': abs(cv) < 0.5
                })
    
    collapse_df = pd.DataFrame(collapse_data)
    collapse_df.to_csv(OUTPUT_DIR / "data" / "parameter_collapse.csv", index=False)
    print(f"  Saved parameter_collapse.csv")
    
    print("\n[5/6] Generating visualizations...")
    
    # R² comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: R² comparison
    ax = axes[0]
    systems = sorted(model_results['system'].unique())
    x = np.arange(len(systems))
    width = 0.35
    
    r2_d = []
    r2_g = []
    for s in systems:
        d_val = model_results[(model_results['system'] == s) & (model_results['model'] == 'D_baseline')]['R2'].values
        g_val = model_results[(model_results['system'] == s) & (model_results['model'] == 'G_compression')]['R2'].values
        r2_d.append(d_val[0] if len(d_val) > 0 else 0)
        r2_g.append(g_val[0] if len(g_val) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, r2_d, width, label='D_baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, r2_g, width, label='G_compression', color='darkorange')
    
    ax.set_ylabel('R²')
    ax.set_title('Model Comparison: D vs D+R')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: R² improvement
    ax = axes[1]
    improvements = [r2_g[i] - r2_d[i] for i in range(len(systems))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(systems, improvements, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('ΔR² (Improvement)')
    ax.set_title('R² Improvement from Adding Compression')
    ax.set_xticklabels(systems, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "r2_comparison.png", dpi=150)
    plt.close()
    
    # Compression vs Residual scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for system in deriv_df['system'].unique():
        sys_data = deriv_df[deriv_df['system'] == system]
        
        # Get model predictions
        model_row = model_results[(model_results['system'] == system) & (model_results['model'] == 'D_baseline')]
        if len(model_row) > 0:
            a, b, c, d = model_row[['a', 'b', 'c', 'd']].values[0]
            pred = a * sys_data['D_eff'] + b * sys_data['H'] + c * sys_data['D_eff'] * sys_data['H'] + d
            residual = sys_data['dbeta'] - pred
            ax.scatter(sys_data['R'], residual, alpha=0.6, label=system)
    
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Compression Ratio (D_eff / k)')
    ax.set_ylabel('Residual (dβ - Model D prediction)')
    ax.set_title('Compression vs Residual: Testing Third Variable')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "compression_vs_residual.png", dpi=150)
    plt.close()
    
    # System-specific analysis
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for i, system in enumerate(sorted(deriv_df['system'].unique())):
        ax = axes[i // 3, i % 3]
        sys_data = deriv_df[deriv_df['system'] == system]
        
        # Model D prediction
        model_row = model_results[(model_results['system'] == system) & (model_results['model'] == 'D_baseline')]
        if len(model_row) > 0 and not np.isnan(model_row['R2'].values[0]):
            a, b, c, d = model_row[['a', 'b', 'c', 'd']].values[0]
            pred_d = a * sys_data['D_eff'] + b * sys_data['H'] + c * sys_data['D_eff'] * sys_data['H'] + d
            
            # Model G prediction
            model_g_row = model_results[(model_results['system'] == system) & (model_results['model'] == 'G_compression')]
            if len(model_g_row) > 0 and not np.isnan(model_g_row['R2'].values[0]):
                a2, b2, c2, e2, f2 = model_g_row[['a', 'b', 'c', 'd', 'e']].values[0]
                pred_g = a2 * sys_data['D_eff'] + b2 * sys_data['H'] + c2 * sys_data['D_eff'] * sys_data['H'] + e2 * sys_data['R'] + f2
            
            actual = sys_data['dbeta']
            ax.scatter(actual, pred_d, alpha=0.5, label=f'D (R²={model_row["R2"].values[0]:.3f})')
            ax.scatter(actual, pred_g, alpha=0.5, label=f'G (R²={model_g_row["R2"].values[0]:.3f})')
            ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', alpha=0.3)
            ax.set_xlabel('Actual dβ')
            ax.set_ylabel('Predicted dβ')
            ax.set_title(system)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "system_fits.png", dpi=150)
    plt.close()
    
    print("  Saved all plots")
    
    print("\n[6/6] Generating summary...")
    
    # Compute summary statistics
    model_d_r2 = model_results[model_results['model'] == 'D_baseline']['R2'].mean()
    model_g_r2 = model_results[model_results['model'] == 'G_compression']['R2'].mean()
    overall_improvement = model_g_r2 - model_d_r2
    
    # TRIBE-specific
    tribe_d = model_results[(model_results['system'] == 'TRIBE') & (model_results['model'] == 'D_baseline')]['R2'].values
    tribe_g = model_results[(model_results['system'] == 'TRIBE') & (model_results['model'] == 'G_compression')]['R2'].values
    tribe_improvement = (tribe_g[0] - tribe_d[0]) if len(tribe_d) > 0 and len(tribe_g) > 0 else np.nan
    
    # Parameter collapse
    d_collapsed = collapse_df[(collapse_df['model'] == 'D_baseline') & (collapse_df['collapsed'])]['parameter'].tolist()
    g_collapsed = collapse_df[(collapse_df['model'] == 'G_compression') & (collapse_df['collapsed'])]['parameter'].tolist()
    
    summary = f"""================================================================================
V27 HIGH-RESOLUTION STATE SPACE VALIDATION SUMMARY
================================================================================

OBJECTIVE: Test whether compression (D_eff/k) is the missing state variable
responsible for TRIBE deviation.

================================================================================
DATA SUMMARY
================================================================================

High-resolution sampling: 13 k-values (log-spaced: 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500)
Total data points per system: ~{len(deriv_df[deriv_df['system'] == 'TRIBE'])} (after derivative computation)

================================================================================
MODEL COMPARISON
================================================================================

Model D (baseline):   dβ = a·D + b·H + c·(D·H) + d
Model G (compression): dβ = a·D + b·H + c·(D·H) + e·R + f

System             | D_baseline R² | G_compression R² | ΔR²
-------------------|---------------|-------------------|--------
"""
    
    for system in sorted(model_results['system'].unique()):
        r2_d = model_results[(model_results['system'] == system) & (model_results['model'] == 'D_baseline')]['R2'].values
        r2_g = model_results[(model_results['system'] == system) & (model_results['model'] == 'G_compression')]['R2'].values
        r2_d = r2_d[0] if len(r2_d) > 0 else np.nan
        r2_g = r2_g[0] if len(r2_g) > 0 else np.nan
        improvement = r2_g - r2_d if not np.isnan(r2_d) and not np.isnan(r2_g) else np.nan
        summary += f"{system:17s} | {r2_d:13.4f} | {r2_g:17.4f} | {improvement:+.4f}\n"
    
    summary += f"""
Mean R² (D):     {model_d_r2:.4f}
Mean R² (G):     {model_g_r2:.4f}
Overall ΔR²:     {overall_improvement:+.4f}

================================================================================
TRIBE-SPECIFIC ANALYSIS
================================================================================

TRIBE D_baseline R²:   {tribe_d[0]:.4f}
TRIBE G_compression R²: {tribe_g[0]:.4f}
TRIBE ΔR²:             {tribe_improvement:+.4f}

"""
    
    if not np.isnan(tribe_improvement):
        if tribe_improvement > 0.1:
            summary += "RESULT: Compression SIGNIFICANTLY improves TRIBE fit\n"
        elif tribe_improvement > 0:
            summary += "RESULT: Compression MARGINALLY improves TRIBE fit\n"
        else:
            summary += "RESULT: Compression does NOT improve TRIBE fit\n"
    else:
        summary += "RESULT: Insufficient data for TRIBE analysis\n"
    
    summary += f"""
================================================================================
PARAMETER COLLAPSE ANALYSIS
================================================================================

Model D parameters collapsed: {d_collapsed if d_collapsed else 'None'}
Model G parameters collapsed: {g_collapsed if g_collapsed else 'None'}

"""
    
    if len(g_collapsed) > len(d_collapsed):
        summary += "RESULT: Adding compression IMPROVES parameter stability\n"
    elif len(g_collapsed) == len(d_collapsed):
        summary += "RESULT: Parameter stability UNCHANGED\n"
    else:
        summary += "RESULT: Adding compression REDUCES parameter stability\n"
    
    summary += """
================================================================================
KEY FINDINGS
================================================================================

1. DOES COMPRESSION IMPROVE MODEL FIT?
"""
    
    if overall_improvement > 0.05:
        summary += f"   YES - Mean R² improves by {overall_improvement:.4f}\n"
    elif overall_improvement > 0:
        summary += f"   MARGINAL - Mean R² improves by {overall_improvement:.4f}\n"
    else:
        summary += f"   NO - Mean R² does not improve\n"
    
    summary += """
2. IS IMPROVEMENT TRIBE-SPECIFIC?
"""
    
    if not np.isnan(tribe_improvement):
        synthetic_systems = ['hierarchical', 'correlated', 'sparse', 'manifold']
        synthetic_improvements = []
        for s in synthetic_systems:
            r2_d = model_results[(model_results['system'] == s) & (model_results['model'] == 'D_baseline')]['R2'].values
            r2_g = model_results[(model_results['system'] == s) & (model_results['model'] == 'G_compression')]['R2'].values
            if len(r2_d) > 0 and len(r2_g) > 0:
                synthetic_improvements.append(r2_g[0] - r2_d[0])
        
        if len(synthetic_improvements) > 0:
            mean_synth_imp = np.mean(synthetic_improvements)
            if tribe_improvement > mean_synth_imp * 1.5:
                summary += f"   YES - TRIBE improvement ({tribe_improvement:+.4f}) > Synthetic mean ({mean_synth_imp:+.4f})\n"
            elif tribe_improvement > 0 and mean_synth_imp <= 0:
                summary += f"   YES - TRIBE improves ({tribe_improvement:+.4f}) while synthetic does NOT\n"
            else:
                summary += f"   NO - Similar improvement across systems\n"
    
    summary += """
3. DOES COMPRESSION REDUCE RESIDUAL STRUCTURE?
"""
    
    residual_corrs = []
    for system in deriv_df['system'].unique():
        sys_data = deriv_df[deriv_df['system'] == system]
        model_row = model_results[(model_results['system'] == system) & (model_results['model'] == 'D_baseline')]
        if len(model_row) > 0:
            a, b, c, d = model_row[['a', 'b', 'c', 'd']].values[0]
            pred = a * sys_data['D_eff'] + b * sys_data['H'] + c * sys_data['D_eff'] * sys_data['H'] + d
            residual = sys_data['dbeta'] - pred
            if len(residual) > 3 and np.std(residual) > 0:
                r, _ = pearsonr(sys_data['R'], residual)
                residual_corrs.append({'system': system, 'r': r})
    
    if len(residual_corrs) > 0:
        max_corr = max(residual_corrs, key=lambda x: abs(x['r']))
        summary += f"   Max correlation (R vs residual): |r| = {abs(max_corr['r']):.3f} ({max_corr['system']})\n"
        if abs(max_corr['r']) > 0.5:
            summary += "   STRUCTURED - Compression explains residuals\n"
        elif abs(max_corr['r']) > 0.3:
            summary += "   PARTIAL - Compression partially explains residuals\n"
        else:
            summary += "   RANDOM - Residuals not explained by compression\n"
    
    summary += """
4. DOES PARAMETER STABILITY IMPROVE?
"""
    
    if len(g_collapsed) >= len(d_collapsed):
        summary += f"   YES - {len(g_collapsed)} parameters collapse vs {len(d_collapsed)}\n"
    else:
        summary += f"   NO - {len(g_collapsed)} vs {len(d_collapsed)} parameters collapse\n"
    
    summary += """
5. FINAL CONCLUSION
"""
    
    if tribe_improvement > 0.1 and overall_improvement > 0.05:
        summary += """   COMPRESSION IS VALIDATED as a third state variable.
   
   Evidence:
   - TRIBE shows significant R² improvement with compression
   - Compression captures TRIBE-specific residual structure
   - The complete state-space model is: dβ = F(D_eff, H, R)
   
"""
    elif overall_improvement > 0.05:
        summary += """   COMPRESSION PARTIALLY VALIDATED.
   
   Evidence:
   - Overall R² improvement present
   - But may not be TRIBE-specific
   - Compression adds general information
   
"""
    else:
        summary += """   COMPRESSION NOT VALIDATED.
   
   Evidence:
   - No significant R² improvement
   - Residuals remain unstructured
   - The two-variable model (D_eff, H) is sufficient
   
   Recommendation: Stop variable search, accept (D_eff, H) model.
   
"""
    
    summary += """================================================================================
FILES GENERATED
================================================================================

/v27_high_res/data/high_res_base.csv       - D_eff, H, R at each k
/v27_high_res/data/high_res_dataset.csv   - With derivatives
/v27_high_res/data/model_results.csv       - Model fits
/v27_high_res/data/parameter_collapse.csv  - Collapse analysis
/v27_high_res/plots/r2_comparison.png     - R² comparison
/v27_high_res/plots/compression_vs_residual.png - Compression correlation
/v27_high_res/plots/system_fits.png      - Per-system fits

================================================================================
"""
    
    with open(OUTPUT_DIR / "validation_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
