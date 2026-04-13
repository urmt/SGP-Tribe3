#!/usr/bin/env python3
"""
V23 Xi Analysis - Construct and test xi variables for beta dynamics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path("/home/student/sgp-tribe3/reproducibility")
OUTPUT_DIR = Path("/home/student/sgp-tribe3/v23_xi_analysis")

def load_data():
    """Load raw data."""
    data = np.load(DATA_DIR / "raw_data.npz")
    
    # Generate synthetic data with fixed seed
    np.random.seed(42)
    
    # Hierarchical: exponentially decaying variance
    hierarchical = np.random.randn(2000, 50) * np.exp(-np.arange(50) * 0.05)
    
    # Correlated: full covariance
    cov = np.eye(50) * 0.5 + 0.5
    correlated = np.random.multivariate_normal(np.zeros(50), cov, size=2000)
    
    # Sparse: 90% zeros
    sparse = np.random.randn(2000, 50) * (np.random.rand(2000, 50) > 0.9)
    
    # Manifold: embedded curve
    t = np.random.randn(2000) * np.pi
    basis = np.vstack([np.sin(np.linspace(0, np.pi, 50)), np.cos(np.linspace(0, np.pi, 50))])
    manifold = np.outer(np.sin(t), basis[0]) + np.outer(np.cos(t), basis[1]) + np.random.randn(2000, 50) * 0.1
    
    return {
        'TRIBE': data['sgp_nodes'],
        'hierarchical': hierarchical,
        'correlated': correlated,
        'sparse': sparse,
        'manifold': manifold
    }

def compute_D_eff_k(points, k_values=[5, 10, 20, 50, 100, 200, 500]):
    """Compute D_eff(k) using local covariance."""
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=max(k_values)+1, metric='euclidean').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    D_eff_dict = {}
    for k in k_values:
        D_eff_values = []
        for i in range(len(points)):
            neighbor_indices = indices[i][:k]
            neighbors = points[neighbor_indices]
            cov = np.cov(neighbors.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                D_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            else:
                D_eff = 1.0
            D_eff_values.append(D_eff)
        D_eff_dict[k] = np.mean(D_eff_values)
    
    return D_eff_dict

def compute_beta_k(D_eff_dict, k_values):
    """Compute beta(k) via sliding window."""
    beta_dict = {}
    log_k = np.log(k_values)
    k_arr = np.array(k_values)
    
    for system, D_eff_values in D_eff_dict.items():
        beta_values = []
        for i in range(len(k_values)):
            if i < len(k_values) - 1:
                delta_D = D_eff_values[k_arr[i+1]] - D_eff_values[k_arr[i]]
                delta_log_k = log_k[i+1] - log_k[i]
                if delta_log_k > 0:
                    beta_values.append(delta_D / delta_log_k)
                else:
                    beta_values.append(np.nan)
            else:
                beta_values.append(np.nan)
        beta_dict[system] = np.array(beta_values[:-1])
    
    return beta_dict

def compute_xi_variables(D_eff_dict, k_values, ambient_dims):
    """Compute xi variables for each system."""
    xi_dict = {}
    k_arr = np.array(k_values)
    log_k = np.log(k_arr)
    
    for system, D_eff_values in D_eff_dict.items():
        D_arr = np.array([D_eff_values[k] for k in k_values])
        D_max = np.max(D_arr)
        D_ambient = ambient_dims[system]
        
        xi = {
            'system': system,
            'xi1_curvature': [],   # d²D_eff / d(log k)²
            'xi2_variance': [],    # std of D_eff (placeholder - use curvature variance)
            'xi3_compression': [], # 1 - D_eff / D_ambient
            'xi4_norm_gradient': [], # d(D_eff/D_max) / d(log k)
            'xi5_slope_change': []  # Δβ
        }
        
        # Compute first derivative
        dD = np.diff(D_arr)
        d_log_k = np.diff(log_k)
        first_deriv = dD / d_log_k
        
        # Compute second derivative (curvature)
        d_first = np.diff(first_deriv)
        second_deriv = d_first / d_log_k[1:]
        
        n_segments = len(second_deriv)
        for i in range(n_segments):
            xi['xi1_curvature'].append(second_deriv[i] if i < len(second_deriv) else 0)
            xi['xi2_variance'].append(second_deriv[i]**2 if i < len(second_deriv) else 0)
            xi['xi3_compression'].append(1 - D_arr[min(i+2, len(D_arr)-1)] / D_ambient)
            xi['xi4_norm_gradient'].append(first_deriv[min(i+1, len(first_deriv)-1)] / D_max if D_max > 0 else 0)
            xi['xi5_slope_change'].append(second_deriv[i] if i < len(second_deriv) else 0)
        
        xi_dict[system] = xi
    
    return xi_dict

def fit_xi_model(beta_dict, D_eff_dict, xi_dict, k_values, xi_name):
    """Fit model with xi: dβ = a·β + b·D + c·(β·D) + e·ξ"""
    results = {}
    k_arr = np.array(k_values)
    
    for system in beta_dict.keys():
        beta = beta_dict[system]
        D_eff = np.array([D_eff_dict[system][k] for k in k_arr[1:len(beta)+1]])
        xi = np.array(xi_dict[system][xi_name])
        
        # Compute dbeta
        log_k = np.log(k_arr[1:len(beta)+1])
        dbeta = np.diff(beta) / np.diff(log_k)
        
        # Truncate to same length
        n = min(len(dbeta), len(beta)-1, len(D_eff)-1, len(xi)-1)
        dbeta = dbeta[:n]
        beta_use = beta[:n]
        D_use = D_eff[:n]
        xi_use = xi[:n]
        
        if n < 3:
            results[system] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 'e': np.nan, 'R_squared': np.nan}
            continue
        
        def model_with_xi(X, a, b, c, e):
            beta_i, D_i, xi_i = X
            return a * beta_i + b * D_i + c * (beta_i * D_i) + e * xi_i
        
        try:
            popt, _ = curve_fit(model_with_xi, (beta_use, D_use, xi_use), dbeta, p0=[0, 0, 0, 0], maxfev=5000)
            pred = model_with_xi((beta_use, D_use, xi_use), *popt)
            ss_res = np.sum((dbeta - pred) ** 2)
            ss_tot = np.sum((dbeta - np.mean(dbeta)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results[system] = {
                'a': popt[0], 'b': popt[1], 'c': popt[2], 'e': popt[3],
                'R_squared': r_squared
            }
        except:
            results[system] = {'a': np.nan, 'b': np.nan, 'c': np.nan, 'e': np.nan, 'R_squared': np.nan}
    
    return results

def main():
    print("=" * 60)
    print("V23 Xi Analysis - Testing Xi Variables")
    print("=" * 60)
    
    ambient_dims = {
        'TRIBE': 9,
        'hierarchical': 50,
        'correlated': 50,
        'sparse': 50,
        'manifold': 50
    }
    k_values = [5, 10, 20, 50, 100, 200, 500]
    
    print("\n[1/6] Loading data...")
    data = load_data()
    
    print("\n[2/6] Computing D_eff(k) curves...")
    D_eff_results = {}
    for system, points in data.items():
        D_eff_results[system] = compute_D_eff_k(points, k_values)
        print(f"  {system}: D_eff(k=500) = {D_eff_results[system][500]:.2f}")
    
    print("\n[3/6] Computing beta(k) curves...")
    beta_results = compute_beta_k(D_eff_results, k_values)
    
    print("\n[4/6] Computing xi variables...")
    xi_results = compute_xi_variables(D_eff_results, k_values, ambient_dims)
    
    # Save xi variables
    xi_rows = []
    for system, xi in xi_results.items():
        for i in range(len(xi['xi1_curvature'])):
            xi_rows.append({
                'system': system,
                'segment': i,
                'xi1_curvature': xi['xi1_curvature'][i],
                'xi2_variance': xi['xi2_variance'][i],
                'xi3_compression': xi['xi3_compression'][i],
                'xi4_norm_gradient': xi['xi4_norm_gradient'][i],
                'xi5_slope_change': xi['xi5_slope_change'][i]
            })
    df_xi = pd.DataFrame(xi_rows)
    df_xi.to_csv(OUTPUT_DIR / "data" / "xi_variables.csv", index=False)
    print(f"  Saved xi_variables.csv ({len(xi_rows)} rows)")
    
    print("\n[5/6] Fitting xi-integrated models...")
    xi_names = ['xi1_curvature', 'xi2_variance', 'xi3_compression', 'xi4_norm_gradient', 'xi5_slope_change']
    model_fits = {}
    
    for xi_name in xi_names:
        fits = fit_xi_model(beta_results, D_eff_results, xi_results, k_values, xi_name)
        model_fits[xi_name] = fits
        mean_r2 = np.nanmean([f['R_squared'] for f in fits.values()])
        print(f"  {xi_name}: mean R² = {mean_r2:.4f}")
    
    # Save model fits
    fit_rows = []
    for xi_name, fits in model_fits.items():
        for system, params in fits.items():
            fit_rows.append({
                'xi_variable': xi_name,
                'system': system,
                'coefficient_a': params['a'],
                'coefficient_b': params['b'],
                'coefficient_c': params['c'],
                'coefficient_e': params['e'],
                'R_squared': params['R_squared']
            })
    df_fits = pd.DataFrame(fit_rows)
    df_fits.to_csv(OUTPUT_DIR / "data" / "model_fits.csv", index=False)
    print(f"  Saved model_fits.csv ({len(fit_rows)} rows)")
    
    print("\n[6/6] Computing parameter collapse...")
    collapse_rows = []
    for xi_name, fits in model_fits.items():
        params = {k: v for k, v in fits.items() if not np.isnan(v['a'])}
        if len(params) > 0:
            for param_name in ['a', 'b', 'c', 'e']:
                values = [p[param_name] for p in params.values()]
                values = [v for v in values if not np.isnan(v)]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
                    collapse_rows.append({
                        'xi_variable': xi_name,
                        'parameter': param_name,
                        'mean': mean_val,
                        'std': std_val,
                        'CV': cv,
                        'collapsed': abs(cv) < 0.5
                    })
    
    df_collapse = pd.DataFrame(collapse_rows)
    df_collapse.to_csv(OUTPUT_DIR / "data" / "parameter_collapse.csv", index=False)
    print(f"  Saved parameter_collapse.csv ({len(collapse_rows)} rows)")
    
    # Generate visualizations
    print("\n[7/6] Generating visualizations...")
    
    # Xi vs Beta
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, xi_name in enumerate(xi_names):
        ax = axes[i // 3, i % 3]
        for system, xi in xi_results.items():
            xi_arr = np.array(xi[xi_name])
            beta_arr = beta_results[system]
            n = min(len(xi_arr), len(beta_arr))
            if n > 0:
                ax.scatter(xi_arr[:n], beta_arr[:n], alpha=0.5, label=system)
        ax.set_xlabel(xi_name)
        ax.set_ylabel('beta')
        ax.legend(fontsize=8)
        ax.set_title(f'{xi_name} vs beta')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "xi_vs_beta.png", dpi=150)
    plt.close()
    
    # Xi vs D_eff
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, xi_name in enumerate(xi_names):
        ax = axes[i // 3, i % 3]
        for system, xi in xi_results.items():
            xi_arr = np.array(xi[xi_name])
            D_vals = [D_eff_results[system][k] for k in k_values[1:len(xi_arr)+1]]
            n = min(len(xi_arr), len(D_vals))
            if n > 0:
                ax.scatter(xi_arr[:n], D_vals[:n], alpha=0.5, label=system)
        ax.set_xlabel(xi_name)
        ax.set_ylabel('D_eff')
        ax.legend(fontsize=8)
        ax.set_title(f'{xi_name} vs D_eff')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "xi_vs_D.png", dpi=150)
    plt.close()
    
    # Model R² comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    xi_labels = [x.replace('xi', 'ξ') for x in xi_names]
    mean_r2s = [np.nanmean([f['R_squared'] for f in model_fits[x].values()]) for x in xi_names]
    bars = ax.bar(xi_labels, mean_r2s, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'])
    ax.axhline(y=0.6591, color='gray', linestyle='--', label='V20 baseline')
    ax.set_ylabel('Mean R²')
    ax.set_title('Model Fit Comparison: Xi-Integrated vs Baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "model_fit_with_xi.png", dpi=150)
    plt.close()
    
    # Parameter variance reduction
    fig, ax = plt.subplots(figsize=(8, 5))
    v20_params = {'a': 1.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}  # simplified
    for i, xi_name in enumerate(xi_names):
        params = list(model_fits[xi_name].values())
        a_vals = [p['a'] for p in params if not np.isnan(p['a'])]
        if a_vals:
            cv = np.std(a_vals) / abs(np.mean(a_vals)) if np.mean(a_vals) != 0 else 0
            ax.bar(i, cv, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'][i])
    ax.set_xticks(range(len(xi_names)))
    ax.set_xticklabels([x.replace('xi', 'ξ') for x in xi_names])
    ax.set_ylabel('Coefficient of Variation (A)')
    ax.set_title('Parameter Variance by Xi Variable')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "parameter_variance_reduction.png", dpi=150)
    plt.close()
    
    print("  Saved all plots")
    
    # Generate summary
    best_xi = max(xi_names, key=lambda x: np.nanmean([f['R_squared'] for f in model_fits[x].values()]))
    best_r2 = np.nanmean([f['R_squared'] for f in model_fits[best_xi].values()])
    baseline_r2 = 0.6591
    
    summary = f"""================================================================================
V23 XI ANALYSIS SUMMARY
================================================================================

OBJECTIVE: Test whether adding xi (ξ) variables improves beta dynamics model

XI VARIABLES TESTED:
  ξ₁ = d²D_eff / d(log k)²     (Curvature)
  ξ₂ = Var(D_eff)              (Variance proxy)
  ξ₃ = 1 - D_eff / D_ambient   (Compression)
  ξ₄ = d(D_eff/D_max) / d(log k) (Normalized gradient)
  ξ₅ = Δβ                      (Slope change)

MODEL: dβ/dlog(k) = a·β + b·D_eff + c·(β·D_eff) + e·ξ

================================================================================
RESULTS
================================================================================

MODEL FIT COMPARISON:
  Baseline (V20):     R² = 0.6591
  With ξ₁ (curvature):  R² = {np.nanmean([f['R_squared'] for f in model_fits['xi1_curvature'].values()]):.4f}
  With ξ₂ (variance):   R² = {np.nanmean([f['R_squared'] for f in model_fits['xi2_variance'].values()]):.4f}
  With ξ₃ (compression): R² = {np.nanmean([f['R_squared'] for f in model_fits['xi3_compression'].values()]):.4f}
  With ξ₄ (norm gradient): R² = {np.nanmean([f['R_squared'] for f in model_fits['xi4_norm_gradient'].values()]):.4f}
  With ξ₅ (slope change): R² = {np.nanmean([f['R_squared'] for f in model_fits['xi5_slope_change'].values()]):.4f}

BEST XI VARIABLE: {best_xi} (R² = {best_r2:.4f})
IMPROVEMENT vs V20: {((best_r2 - baseline_r2) / baseline_r2 * 100):.1f}%

================================================================================
PARAMETER COLLAPSE ANALYSIS
================================================================================

"""
    
    for xi_name in xi_names:
        summary += f"\n{xi_name}:\n"
        params = model_fits[xi_name]
        for param in ['a', 'b', 'c', 'e']:
            values = [p[param] for p in params.values() if not np.isnan(p[param])]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                collapsed = "COLLAPSED" if abs(cv) < 0.5 else "NOT COLLAPSED"
                summary += f"  {param}: mean={mean_val:.4f}, CV={cv:.2f} [{collapsed}]\n"
    
    summary += f"""
================================================================================
INTERPRETATION
================================================================================

1. BEST XI: {best_xi}
   - R² = {best_r2:.4f} (vs baseline 0.6591)
   - {'Improved fit' if best_r2 > baseline_r2 else 'No improvement'} over V20 model

2. PARAMETER COLLAPSE:
"""
    
    collapse_count = sum(1 for _, row in df_collapse.iterrows() if row['collapsed'])
    total_params = len(df_collapse)
    summary += f"   - {collapse_count}/{total_params} parameters show collapse (CV < 0.5)\n"
    
    summary += """
3. XI BEHAVIOR:
"""
    
    if best_xi in ['xi1_curvature', 'xi4_norm_gradient']:
        summary += "   - Xi captures local dynamics (derivative-based)\n"
        summary += "   - Interpretation: Xi behaves as STATE VARIABLE\n"
    else:
        summary += "   - Xi captures global properties (static)\n"
        summary += "   - Interpretation: Xi may be noise or confounded\n"
    
    summary += """
4. CONCLUSION:
"""
    
    if best_r2 > baseline_r2 * 1.1 and collapse_count > total_params * 0.5:
        summary += "   - Xi improves model AND reduces parameter variance\n"
        summary += "   - Xi is a meaningful state variable\n"
    elif best_r2 > baseline_r2 * 1.1:
        summary += "   - Xi improves model but parameters do not collapse\n"
        summary += "   - Xi is informative but not universal\n"
    else:
        summary += "   - Xi does not improve model fit\n"
        summary += "   - Xi is not a meaningful addition to the model\n"
    
    summary += """
================================================================================
FILES GENERATED
================================================================================

/v23_xi_analysis/data/xi_variables.csv    - Raw xi values per system
/v23_xi_analysis/data/model_fits.csv      - Model parameters and R²
/v23_xi_analysis/data/parameter_collapse.csv - Collapse analysis
/v23_xi_analysis/plots/xi_vs_beta.png    - Xi vs beta scatter
/v23_xi_analysis/plots/xi_vs_D.png       - Xi vs D_eff scatter
/v23_xi_analysis/plots/model_fit_with_xi.png - R² comparison
/v23_xi_analysis/plots/parameter_variance_reduction.png - Variance analysis

================================================================================
"""
    
    with open(OUTPUT_DIR / "xi_analysis_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
