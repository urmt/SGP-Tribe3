#!/usr/bin/env python3
"""
V25: State Space Closure Test
Test whether beta dynamics can be expressed as F(D_eff, H) without explicit k dependence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = Path("/home/student/sgp-tribe3/v25_state_space")

def load_data():
    """Load D_eff from V16 and H from V24."""
    # Load D_eff from V16
    deff_df = pd.read_csv("/home/student/sgp-tribe3/manuscript/v16/all_curves.csv")
    deff_df = deff_df.rename(columns={'System': 'system'})
    
    # Load H from V24
    h_df = pd.read_csv("/home/student/sgp-tribe3/v24_spectral_shape_analysis/data/spectral_shape_metrics.csv")
    h_df = h_df[['system', 'k', 'H']]
    
    return deff_df, h_df

def build_state_dataset():
    """Build aligned state dataset."""
    deff_df, h_df = load_data()
    
    # Merge datasets
    merged = pd.merge(deff_df, h_df, on=['system', 'k'], how='inner')
    
    # Compute beta from D_eff
    state_rows = []
    for system in merged['system'].unique():
        sys_data = merged[merged['system'] == system].sort_values('k')
        k_vals = sys_data['k'].values
        D_vals = sys_data['D_eff'].values
        H_vals = sys_data['H'].values
        
        # Compute beta = dD_eff/dlog(k)
        beta_vals = []
        for i in range(len(k_vals) - 1):
            delta_D = D_vals[i+1] - D_vals[i]
            delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
            if delta_log_k > 0:
                beta_vals.append(delta_D / delta_log_k)
            else:
                beta_vals.append(np.nan)
        beta_vals.append(np.nan)  # Last point has no beta
        
        # Compute dbeta/dlog(k)
        dbeta_vals = []
        for i in range(len(beta_vals) - 1):
            if not np.isnan(beta_vals[i]) and not np.isnan(beta_vals[i+1]):
                k_mid = (k_vals[i] + k_vals[i+1]) / 2
                delta_beta = beta_vals[i+1] - beta_vals[i]
                delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
                if delta_log_k > 0:
                    dbeta_vals.append(delta_beta / delta_log_k)
                else:
                    dbeta_vals.append(np.nan)
            else:
                dbeta_vals.append(np.nan)
        
        for i in range(len(k_vals)):
            if not np.isnan(beta_vals[i]):
                state_rows.append({
                    'system': system,
                    'k': k_vals[i],
                    'D_eff': D_vals[i],
                    'H': max(0, H_vals[i]),  # Ensure non-negative
                    'beta': beta_vals[i] if i < len(beta_vals) else np.nan
                })
    
    state_df = pd.DataFrame(state_rows)
    return state_df

def fit_models(state_df):
    """Fit all state-space models."""
    models = {}
    
    def model_a(X, a, b):
        D, H = X
        return a * D + b
    
    def model_b(X, a, b):
        D, H = X
        return a * H + b
    
    def model_c(X, a, b, c):
        D, H = X
        return a * D + b * H + c
    
    def model_d(X, a, b, c, d):
        D, H = X
        return a * D + b * H + c * D * H + d
    
    def model_e(X, a, b, c):
        D, H = X
        return a * np.log(D + 1e-10) + b * H + c
    
    def model_f(X, a, b, c, d, e):
        D, H = X
        return a * D + b * H + c * D * H + d * np.log(D + 1e-10) + e
    
    model_defs = {
        'A_baseline': (model_a, 2),
        'B_H_only': (model_b, 2),
        'C_linear': (model_c, 3),
        'D_interaction': (model_d, 4),
        'E_log': (model_e, 3),
        'F_full': (model_f, 5)
    }
    
    for system in state_df['system'].unique():
        sys_data = state_df[state_df['system'] == system]
        beta = sys_data['beta'].values
        D = sys_data['D_eff'].values
        H = sys_data['H'].values
        
        # Compute dbeta
        dbeta = np.diff(beta) / np.diff(np.log(sys_data['k'].values))
        
        # Truncate for alignment
        n = min(len(dbeta), len(beta)-1, len(D)-1)
        beta_fit = beta[:n]
        D_fit = D[:n]
        H_fit = H[:n]
        dbeta_fit = dbeta[:n]
        
        # Filter valid data
        valid = ~(np.isnan(dbeta_fit) | np.isnan(beta_fit) | np.isnan(D_fit) | np.isnan(H_fit))
        if np.sum(valid) < 3:
            continue
        
        X = (D_fit[valid], H_fit[valid])
        y = dbeta_fit[valid]
        
        for model_name, (model_func, n_params) in model_defs.items():
            try:
                p0 = [0.1] * n_params
                popt, _ = curve_fit(model_func, X, y, p0=p0, maxfev=10000)
                pred = model_func(X, *popt)
                ss_res = np.sum((y - pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                if system not in models:
                    models[system] = {}
                models[system][model_name] = {
                    'params': popt,
                    'R2': r2,
                    'n_points': np.sum(valid)
                }
            except Exception as e:
                if system not in models:
                    models[system] = {}
                models[system][model_name] = {
                    'params': [np.nan] * n_params,
                    'R2': np.nan,
                    'n_points': np.sum(valid)
                }
    
    return models

def compute_collapse(models):
    """Compute parameter collapse statistics."""
    collapse_data = []
    param_names = ['a', 'b', 'c', 'd', 'e']
    
    for model_name in models[list(models.keys())[0]].keys():
        for param_idx, param_name in enumerate(param_names):
            values = []
            for system, model_results in models.items():
                if model_name in model_results:
                    params = model_results[model_name]['params']
                    if param_idx < len(params) and not np.isnan(params[param_idx]):
                        values.append(params[param_idx])
            
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
    
    return pd.DataFrame(collapse_data)

def main():
    print("=" * 60)
    print("V25: State Space Closure Test")
    print("=" * 60)
    
    print("\n[1/5] Building state dataset...")
    state_df = build_state_dataset()
    state_df.to_csv(OUTPUT_DIR / "data" / "state_dataset.csv", index=False)
    print(f"  Built state dataset: {len(state_df)} rows")
    print(f"  Systems: {state_df['system'].unique()}")
    
    print("\n[2/5] Fitting models...")
    models = fit_models(state_df)
    
    # Save model results
    model_rows = []
    for system, model_results in models.items():
        for model_name, results in model_results.items():
            row = {'system': system, 'model': model_name, 'R2': results['R2'], 'n_points': results['n_points']}
            for i, p in enumerate(results['params']):
                row[f'param_{chr(97+i)}'] = p
            model_rows.append(row)
    
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(OUTPUT_DIR / "data" / "model_results.csv", index=False)
    print(f"  Saved model_results.csv ({len(model_rows)} rows)")
    
    print("\n[3/5] Computing parameter collapse...")
    collapse_df = compute_collapse(models)
    collapse_df.to_csv(OUTPUT_DIR / "data" / "parameter_collapse.csv", index=False)
    print(f"  Saved parameter_collapse.csv ({len(collapse_df)} rows)")
    
    print("\n[4/5] Generating visualizations...")
    
    # Predicted vs Actual
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    model_names = list(models[list(models.keys())[0]].keys())
    model_labels = {
        'A_baseline': 'A: D_eff only',
        'B_H_only': 'B: H only',
        'C_linear': 'C: D_eff + H',
        'D_interaction': 'D: D_eff×H',
        'E_log': 'E: log(D)+H',
        'F_full': 'F: Full'
    }
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx // 3, idx % 3]
        for system in models.keys():
            if model_name in models[system]:
                results = models[system][model_name]
                ax.scatter([results['R2']], [system], alpha=0.7)
        
        ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.1, 1.1)
        ax.set_title(model_labels.get(model_name, model_name))
        ax.set_xlabel('R²')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "pred_vs_actual.png", dpi=150)
    plt.close()
    
    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    systems = list(models.keys())
    x = np.arange(len(systems))
    width = 0.12
    
    for i, model_name in enumerate(model_names):
        r2_vals = [models[s].get(model_name, {'R2': np.nan})['R2'] for s in systems]
        ax.bar(x + i * width, r2_vals, width, label=model_labels.get(model_name, model_name))
    
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='R²=0.9 threshold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "model_comparison.png", dpi=150)
    plt.close()
    
    # Parameter collapse heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_cv = collapse_df.pivot(index='model', columns='parameter', values='CV')
    im = ax.imshow(pivot_cv.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=5)
    ax.set_xticks(range(len(pivot_cv.columns)))
    ax.set_xticklabels(pivot_cv.columns)
    ax.set_yticks(range(len(pivot_cv.index)))
    ax.set_yticklabels([model_labels.get(m, m) for m in pivot_cv.index])
    ax.set_title('Parameter Coefficient of Variation')
    plt.colorbar(im, ax=ax, label='CV')
    
    # Add threshold line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "collapse_heatmap.png", dpi=150)
    plt.close()
    
    # 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # Get best model's parameters for surface
    best_model_name = 'F_full'
    if best_model_name in models[list(models.keys())[0]]:
        params = models[list(models.keys())[0]][best_model_name]['params']
        a, b, c, d, e = params[0], params[1], params[2], params[3], params[4]
    else:
        a, b, c, d, e = 0.1, 0.1, 0.01, 0.1, 0.1
    
    D_range = np.linspace(1, 50, 30)
    H_range = np.linspace(0, 4, 30)
    D_grid, H_grid = np.meshgrid(D_range, H_range)
    Z = a * D_grid + b * H_grid + c * D_grid * H_grid + d * np.log(D_grid + 1e-10) + e
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(D_grid, H_grid, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('D_eff')
    ax1.set_ylabel('H')
    ax1.set_zlabel('dβ/dlog(k)')
    ax1.set_title('State Space Surface')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(D_grid, H_grid, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='dβ/dlog(k)')
    ax2.set_xlabel('D_eff')
    ax2.set_ylabel('H')
    ax2.set_title('State Space Contours')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "state_surface.png", dpi=150)
    plt.close()
    
    print("  Saved all plots")
    
    print("\n[5/5] Generating summary...")
    
    # Compute summary statistics
    mean_r2_by_model = model_df.groupby('model')['R2'].mean()
    best_model = mean_r2_by_model.idxmax()
    best_r2 = mean_r2_by_model.max()
    
    # Check collapse
    collapsed_count = collapse_df['collapsed'].sum()
    total_params = len(collapse_df)
    
    # Check if H improves over D_eff alone
    baseline_r2 = model_df[model_df['model'] == 'A_baseline']['R2'].mean()
    h_contribution = model_df[model_df['model'] == 'B_H_only']['R2'].mean()
    linear_r2 = model_df[model_df['model'] == 'C_linear']['R2'].mean()
    full_r2 = model_df[model_df['model'] == 'F_full']['R2'].mean()
    
    summary = f"""================================================================================
V25 STATE SPACE CLOSURE TEST SUMMARY
================================================================================

OBJECTIVE: Test whether beta dynamics can be expressed as F(D_eff, H)
without explicit k dependence.

================================================================================
MODEL COMPARISON
================================================================================

Model                  | Formula                              | Mean R²
-----------------------|--------------------------------------|----------
A: Baseline (D_eff)   | dβ = a·D_eff + b                    | {baseline_r2:.4f}
B: H only              | dβ = a·H + b                         | {h_contribution:.4f}
C: Linear              | dβ = a·D_eff + b·H + c             | {linear_r2:.4f}
D: Interaction         | dβ = a·D + b·H + c·(D·H) + d      | {model_df[model_df['model'] == 'D_interaction']['R2'].mean():.4f}
E: Logarithmic         | dβ = a·log(D) + b·H + c            | {model_df[model_df['model'] == 'E_log']['R2'].mean():.4f}
F: Full                | dβ = a·D + b·H + c·(D·H) + d·log(D) + e | {full_r2:.4f}

BEST MODEL: {best_model} (R² = {best_r2:.4f})

================================================================================
PER-SYSTEM RESULTS
================================================================================

"""
    
    for system in sorted(models.keys()):
        summary += f"\n{system}:\n"
        for model_name in model_names:
            if model_name in models[system]:
                r2 = models[system][model_name]['R2']
                summary += f"  {model_labels.get(model_name, model_name):20s}: R² = {r2:.4f}\n"
    
    summary += f"""
================================================================================
PARAMETER COLLAPSE ANALYSIS
================================================================================

"""
    
    for model_name in model_names:
        model_collapse = collapse_df[collapse_df['model'] == model_name]
        n_collapsed = model_collapse['collapsed'].sum()
        total = len(model_collapse)
        summary += f"{model_labels.get(model_name, model_name):20s}: {n_collapsed}/{total} parameters collapsed\n"
    
    summary += f"""
================================================================================
KEY FINDINGS
================================================================================

1. DOES (D_eff, H) PREDICT dβ WELL?
"""
    
    if full_r2 > 0.9:
        summary += "   YES - Full model achieves R² > 0.9\n"
    else:
        summary += f"   PARTIAL - Full model achieves R² = {full_r2:.4f}\n"
    
    summary += f"""
2. WHICH MODEL PERFORMS BEST?
   {best_model} with mean R² = {best_r2:.4f}

3. DOES ADDING H IMPROVE OVER D_eff ALONE?
"""
    
    if linear_r2 > baseline_r2 * 1.1:
        summary += f"   YES - Adding H improves R² from {baseline_r2:.4f} to {linear_r2:.4f}\n"
    else:
        summary += f"   MARGINAL - H adds {((linear_r2/baseline_r2)-1)*100:.1f}% improvement\n"
    
    summary += f"""
4. DO PARAMETERS COLLAPSE ACROSS SYSTEMS? (CV < 0.5)
   {collapsed_count}/{total_params} parameters collapsed
"""
    
    if collapsed_count >= total_params * 0.5:
        summary += "   YES - Significant collapse observed\n"
    else:
        summary += "   NO - Limited collapse\n"
    
    summary += """
5. CAN β DYNAMICS BE DESCRIBED WITHOUT k?
"""
    
    if full_r2 > 0.9 and collapsed_count >= total_params * 0.3:
        summary += "   LIKELY - State-space closure is achievable\n"
    elif full_r2 > 0.8:
        summary += "   PARTIAL - State-space representation captures most variance\n"
    else:
        summary += "   NO - Explicit k dependence remains important\n"
    
    summary += f"""
================================================================================
INTERPRETATION
================================================================================

"""
    
    if full_r2 > 0.9:
        summary += """SUCCESS: Beta dynamics can be expressed as a function of (D_eff, H).

The state-space closure F(D_eff, H) captures beta dynamics without explicit
neighborhood size (k) dependence. This suggests:

1. The growth-saturation pattern is a property of the representational
   geometry, not of the sampling scale per se.

2. D_eff and H provide complementary information:
   - D_eff captures magnitude of dimensional expansion
   - H captures diversity/distribution of eigenvalue structure

3. The combined model achieves high R² across all systems, indicating
   robustness of the state-space representation.

"""
    elif full_r2 > 0.7:
        summary += """PARTIAL SUCCESS: State-space model captures most variance but explicit
k dependence may remain.

The (D_eff, H) representation explains most but not all beta dynamics.
Remaining variance may reflect:

1. Scale-dependent effects not captured by state variables
2. System-specific transitions not captured by D_eff/H
3. Numerical approximations in computing derivatives

"""
    else:
        summary += """LIMITED SUCCESS: Explicit k dependence remains important.

The state-space model (D_eff, H) does not fully capture beta dynamics.
This suggests:

1. k remains an essential variable
2. The relationship between state variables and beta is nonlinear
3. Additional state variables may be needed

"""
    
    summary += f"""================================================================================
FILES GENERATED
================================================================================

/v25_state_space/data/state_dataset.csv   - Combined state variables
/v25_state_space/data/model_results.csv   - Model fits per system
/v25_state_space/data/parameter_collapse.csv - Collapse statistics
/v25_state_space/plots/pred_vs_actual.png - R² by system/model
/v25_state_space/plots/model_comparison.png - Bar chart comparison
/v25_state_space/plots/collapse_heatmap.png - CV heatmap
/v25_state_space/plots/state_surface.png - 3D surface

================================================================================
"""
    
    with open(OUTPUT_DIR / "state_space_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
