#!/usr/bin/env python3
"""
V26: Residual Analysis - Finding the Missing State Variable
Analyze residuals from state-space model to identify variable explaining TRIBE deviation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("/home/student/sgp-tribe3/v26_residual_analysis")

def load_data():
    """Load state dataset."""
    state_df = pd.read_csv("/home/student/sgp-tribe3/v25_state_space/data/state_dataset.csv")
    model_df = pd.read_csv("/home/student/sgp-tribe3/v25_state_space/data/model_results.csv")
    return state_df, model_df

def compute_residuals(state_df, model_df):
    """Compute residuals from Model D (interaction model)."""
    # Get Model D parameters per system
    model_d = model_df[model_df['model'] == 'D_interaction'].copy()
    
    # Merge with state data
    residuals = []
    
    for _, row in model_d.iterrows():
        system = row['system']
        a, b, c, d = row['param_a'], row['param_b'], row['param_c'], row['param_d']
        
        sys_data = state_df[state_df['system'] == system].sort_values('k')
        k_vals = sys_data['k'].values
        D_vals = sys_data['D_eff'].values
        H_vals = sys_data['H'].values
        beta_vals = sys_data['beta'].values
        
        # Compute dbeta
        for i in range(len(k_vals) - 1):
            if i < len(k_vals) - 1:
                delta_beta = beta_vals[i+1] - beta_vals[i]
                delta_log_k = np.log(k_vals[i+1]) - np.log(k_vals[i])
                if delta_log_k > 0:
                    dbeta = delta_beta / delta_log_k
                    # Model D prediction
                    D = D_vals[i]
                    H = max(0, H_vals[i])
                    pred = a * D + b * H + c * D * H + d
                    residual = dbeta - pred
                    
                    residuals.append({
                        'system': system,
                        'k': k_vals[i],
                        'D_eff': D,
                        'H': H,
                        'beta': beta_vals[i],
                        'dbeta': dbeta,
                        'predicted': pred,
                        'residual': residual
                    })
    
    return pd.DataFrame(residuals)

def compute_candidate_variables(state_df):
    """Compute candidate third variables."""
    candidates = []
    
    for system in state_df['system'].unique():
        sys_data = state_df[state_df['system'] == system].sort_values('k')
        k_vals = sys_data['k'].values
        D_vals = sys_data['D_eff'].values
        H_vals = sys_data['H'].values
        beta_vals = sys_data['beta'].values
        
        # Compute derivatives
        dD = np.diff(D_vals)
        d_log_k = np.diff(np.log(k_vals))
        dD_dlogk = dD / d_log_k  # dD_eff/dlog(k) = beta
        
        # Second derivative of D_eff (curvature)
        d2D = np.diff(dD_dlogk)
        d2_log_k = d_log_k[1:]
        curvature = d2D / d2_log_k
        
        # Second derivative of beta (acceleration)
        d2beta = np.diff(dD_dlogk)
        d2beta_dlogk = d2beta / d2_log_k
        
        # Compression ratio: D_eff / k
        k_mid = k_vals[:-2] if len(curvature) == len(k_vals) - 2 else k_vals[1:-1]
        D_mid = D_vals[:-2] if len(curvature) == len(D_vals) - 2 else D_vals[1:-1]
        compression = D_mid / k_mid
        
        # Normalized slope: beta / D_eff
        beta_mid = beta_vals[:-2] if len(curvature) == len(beta_vals) - 2 else beta_vals[1:-1]
        norm_slope = beta_mid / (D_mid + 1e-10)
        
        # Entropy gradient: dH/dlog(k)
        dH = np.diff(H_vals)
        dH_dlogk = dH / d_log_k
        
        # Saturation distance: D_max - D_eff
        D_max = np.max(D_vals)
        saturation = D_max - D_mid
        
        for i in range(len(curvature)):
            candidates.append({
                'system': system,
                'segment': i,
                'k': k_mid[i] if i < len(k_mid) else np.nan,
                'curvature': curvature[i] if i < len(curvature) else np.nan,
                'acceleration': d2beta_dlogk[i] if i < len(d2beta_dlogk) else np.nan,
                'compression': compression[i] if i < len(compression) else np.nan,
                'norm_slope': norm_slope[i] if i < len(norm_slope) else np.nan,
                'entropy_gradient': dH_dlogk[i] if i < len(dH_dlogk) else np.nan,
                'saturation': saturation[i] if i < len(saturation) else np.nan
            })
    
    return pd.DataFrame(candidates)

def correlation_analysis(residuals_df, candidates_df):
    """Analyze correlations between residuals and candidate variables."""
    # Merge residuals with candidates
    merged = pd.merge(
        residuals_df,
        candidates_df,
        on=['system', 'k'],
        how='inner'
    )
    
    candidate_vars = ['curvature', 'acceleration', 'compression', 'norm_slope', 'entropy_gradient', 'saturation']
    
    results = []
    for var in candidate_vars:
        # Pooled correlation
        valid = merged[['residual', var]].dropna()
        if len(valid) > 3:
            r, p = pearsonr(valid['residual'], valid[var])
        else:
            r, p = np.nan, np.nan
        
        results.append({
            'variable': var,
            'pooled_r': r,
            'pooled_p': p,
            'pooled_n': len(valid)
        })
        
        # Per-system correlations
        for system in merged['system'].unique():
            sys_data = merged[merged['system'] == system][['residual', var]].dropna()
            if len(sys_data) > 2:
                r, p = pearsonr(sys_data['residual'], sys_data[var])
            else:
                r, p = np.nan, np.nan
            results.append({
                'variable': var,
                'system': system,
                'system_r': r,
                'system_p': p
            })
    
    return pd.DataFrame(results)

def augmented_model_test(residuals_df, candidates_df):
    """Test augmented models with candidate variables."""
    merged = pd.merge(
        residuals_df,
        candidates_df,
        on=['system', 'k'],
        how='inner'
    )
    
    candidate_vars = ['curvature', 'acceleration', 'compression', 'norm_slope', 'entropy_gradient', 'saturation']
    
    # Model D baseline R² per system
    baseline_r2 = {}
    for system in merged['system'].unique():
        sys_data = merged[merged['system'] == system]
        ss_res = np.sum(sys_data['residual'] ** 2)
        ss_tot = np.sum((sys_data['dbeta'] - sys_data['dbeta'].mean()) ** 2)
        baseline_r2[system] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    results = []
    for var in candidate_vars:
        for system in merged['system'].unique():
            sys_data = merged[merged['system'] == system].dropna(subset=['residual', 'D_eff', 'H', var])
            
            if len(sys_data) < 3:
                continue
            
            # Fit augmented model: dbeta = a*D + b*H + c*DH + d + e*X
            def model(X, a, b, c, d, e):
                D, H, X_var = X
                return a * D + b * H + c * D * H + d + e * X_var
            
            try:
                X = (sys_data['D_eff'].values, sys_data['H'].values, sys_data[var].values)
                y = sys_data['dbeta'].values
                popt, _ = curve_fit(model, X, y, p0=[0.1, 0.1, 0.01, 0.1, 0.1], maxfev=5000)
                pred = model(X, *popt)
                ss_res = np.sum((y - pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results.append({
                    'variable': var,
                    'system': system,
                    'augmented_R2': r2,
                    'baseline_R2': baseline_r2.get(system, 0),
                    'improvement': r2 - baseline_r2.get(system, 0),
                    'param_e': popt[4]
                })
            except Exception as e:
                results.append({
                    'variable': var,
                    'system': system,
                    'augmented_R2': np.nan,
                    'baseline_R2': baseline_r2.get(system, 0),
                    'improvement': np.nan,
                    'param_e': np.nan
                })
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("V26: Residual Analysis - Finding Missing State Variable")
    print("=" * 60)
    
    print("\n[1/5] Loading data...")
    state_df, model_df = load_data()
    print(f"  Loaded state data: {len(state_df)} rows")
    print(f"  Loaded model data: {len(model_df)} rows")
    
    print("\n[2/5] Computing residuals from Model D...")
    residuals_df = compute_residuals(state_df, model_df)
    residuals_df.to_csv(OUTPUT_DIR / "data" / "residuals.csv", index=False)
    print(f"  Saved residuals.csv ({len(residuals_df)} rows)")
    
    # Compute residual statistics
    print(f"\n  Residual statistics:")
    for system in residuals_df['system'].unique():
        sys_res = residuals_df[residuals_df['system'] == system]['residual']
        print(f"    {system}: mean={sys_res.mean():.4f}, std={sys_res.std():.4f}, max={abs(sys_res).max():.4f}")
    
    print("\n[3/5] Computing candidate variables...")
    candidates_df = compute_candidate_variables(state_df)
    candidates_df.to_csv(OUTPUT_DIR / "data" / "candidates.csv", index=False)
    print(f"  Saved candidates.csv ({len(candidates_df)} rows)")
    
    print("\n[4/5] Running correlation analysis...")
    corr_df = correlation_analysis(residuals_df, candidates_df)
    corr_df.to_csv(OUTPUT_DIR / "data" / "correlation_results.csv", index=False)
    print(f"  Saved correlation_results.csv ({len(corr_df)} rows)")
    
    # Print top correlations
    print("\n  Pooled correlations with residuals:")
    pooled = corr_df[corr_df['variable'].notna() & (corr_df['pooled_r'].notna())]
    if len(pooled) > 0:
        pooled_sorted = pooled.sort_values('pooled_r', key=abs, ascending=False)
        for _, row in pooled_sorted.iterrows():
            print(f"    {row['variable']}: r={row['pooled_r']:.3f}, p={row['pooled_p']:.4f}")
    
    print("\n[5/5] Testing augmented models...")
    aug_df = augmented_model_test(residuals_df, candidates_df)
    aug_df.to_csv(OUTPUT_DIR / "data" / "augmented_model_results.csv", index=False)
    print(f"  Saved augmented_model_results.csv ({len(aug_df)} rows)")
    
    # Print top improvements
    print("\n  R² improvements by variable:")
    for var in aug_df['variable'].unique():
        var_data = aug_df[aug_df['variable'] == var]
        mean_improvement = var_data['improvement'].mean()
        mean_r2 = var_data['augmented_R2'].mean()
        print(f"    {var}: mean R²={mean_r2:.3f}, improvement={mean_improvement:.3f}")
    
    print("\n[6/5] Generating visualizations...")
    
    # Residual vs candidate variables
    candidate_vars = ['curvature', 'acceleration', 'compression', 'norm_slope', 'entropy_gradient', 'saturation']
    var_labels = ['Curvature', 'Acceleration', 'Compression', 'Norm Slope', 'H Gradient', 'Saturation']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for i, (var, label) in enumerate(zip(candidate_vars, var_labels)):
        ax = axes[i // 3, i % 3]
        merged = pd.merge(residuals_df, candidates_df, on=['system', 'k'], how='inner')
        for system in merged['system'].unique():
            sys_data = merged[merged['system'] == system].dropna(subset=['residual', var])
            ax.scatter(sys_data[var], sys_data['residual'], alpha=0.5, label=system)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel(label)
        ax.set_ylabel('Residual')
        ax.legend(fontsize=8)
        ax.set_title(f'Residual vs {label}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "residual_vs_candidates.png", dpi=150)
    plt.close()
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    systems = residuals_df['system'].unique()
    corr_matrix = np.zeros((len(candidate_vars), len(systems)))
    
    for i, var in enumerate(candidate_vars):
        for j, system in enumerate(systems):
            merged = pd.merge(residuals_df, candidates_df, on=['system', 'k'], how='inner')
            sys_data = merged[merged['system'] == system][['residual', var]].dropna()
            if len(sys_data) > 2:
                r, _ = pearsonr(sys_data['residual'], sys_data[var])
                corr_matrix[i, j] = r
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.set_yticks(range(len(candidate_vars)))
    ax.set_yticklabels(var_labels)
    ax.set_title('Residual Correlation with Candidate Variables')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add values
    for i in range(len(candidate_vars)):
        for j in range(len(systems)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                         color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    # R² improvement comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(systems))
    width = 0.12
    
    baseline_r2 = [aug_df[(aug_df['variable'] == candidate_vars[0]) & (aug_df['system'] == s)]['baseline_R2'].values[0] 
                   if len(aug_df[(aug_df['variable'] == candidate_vars[0]) & (aug_df['system'] == s)]) > 0 else 0 
                   for s in systems]
    ax.bar(x - 2.5*width, baseline_r2, width, label='Baseline (D)', color='gray')
    
    for i, var in enumerate(candidate_vars):
        r2_vals = [aug_df[(aug_df['variable'] == var) & (aug_df['system'] == s)]['augmented_R2'].values[0] 
                   if len(aug_df[(aug_df['variable'] == var) & (aug_df['system'] == s)]) > 0 else 0 
                   for s in systems]
        ax.bar(x + (i - 1.5) * width, r2_vals, width, label=var_labels[i])
    
    ax.set_ylabel('R²')
    ax.set_title('Model R² with Augmented Variables')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=8)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "r2_improvement.png", dpi=150)
    plt.close()
    
    print("  Saved all plots")
    
    print("\n[7/5] Generating summary...")
    
    # Find best candidate
    pooled_corr = corr_df[corr_df['pooled_r'].notna()][['variable', 'pooled_r']].drop_duplicates()
    if len(pooled_corr) > 0:
        pooled_corr['abs_r'] = pooled_corr['pooled_r'].abs()
        best_candidate = pooled_corr.sort_values('abs_r', ascending=False).iloc[0]
    else:
        best_candidate = None
    
    # TRIBE-specific analysis
    tribe_corr = corr_df[(corr_df['system'] == 'TRIBE') & (corr_df['system_r'].notna())][['variable', 'system_r']]
    if len(tribe_corr) > 0:
        tribe_corr['abs_r'] = tribe_corr['system_r'].abs()
        best_tribe = tribe_corr.sort_values('abs_r', ascending=False).iloc[0]
    else:
        best_tribe = None
    
    # R² improvements
    aug_df_clean = aug_df.dropna(subset=['improvement'])
    if len(aug_df_clean) > 0:
        mean_improvements = aug_df_clean.groupby('variable')['improvement'].mean()
        valid_improvements = mean_improvements.dropna()
        if len(valid_improvements) > 0:
            best_improvement_var = valid_improvements.idxmax()
        else:
            best_improvement_var = None
    else:
        best_improvement_var = None
    
    summary = f"""================================================================================
V26 RESIDUAL ANALYSIS SUMMARY
================================================================================

OBJECTIVE: Identify the missing state variable responsible for TRIBE deviation
by analyzing residual structure from the best state-space model.

================================================================================
RESIDUAL STATISTICS (Model D)
================================================================================

"""
    
    for system in sorted(residuals_df['system'].unique()):
        sys_res = residuals_df[residuals_df['system'] == system]['residual']
        summary += f"{system:15s}: mean={sys_res.mean():+.4f}, std={sys_res.std():.4f}, |max|={abs(sys_res).max():.4f}\n"
    
    summary += f"""
================================================================================
CORRELATION ANALYSIS (Pooled)
================================================================================

"""
    
    if len(pooled_corr) > 0:
        for _, row in pooled_corr.sort_values('abs_r', ascending=False).iterrows():
            summary += f"{row['variable']:20s}: r = {row['pooled_r']:+.4f}\n"
    
    summary += f"""
================================================================================
TRIBE-SPECIFIC CORRELATIONS
================================================================================

"""
    
    if len(tribe_corr) > 0:
        for _, row in tribe_corr.sort_values('abs_r', ascending=False).iterrows():
            summary += f"{row['variable']:20s}: r = {row['system_r']:+.4f}\n"
    
    summary += f"""
================================================================================
AUGMENTED MODEL R² IMPROVEMENTS
================================================================================

Variable             | Mean R² | Mean Improvement
---------------------|---------|-----------------
"""
    
    for var in candidate_vars:
        var_data = aug_df[aug_df['variable'] == var]
        mean_r2 = var_data['augmented_R2'].mean()
        mean_imp = var_data['improvement'].mean()
        summary += f"{var:20s} | {mean_r2:7.4f} | {mean_imp:+.4f}\n"
    
    summary += f"""
================================================================================
KEY FINDINGS
================================================================================

1. ARE RESIDUALS STRUCTURED OR RANDOM?
"""
    
    max_residual = residuals_df.groupby('system')['residual'].apply(lambda x: abs(x).max()).max()
    if max_residual > 0.5:
        summary += "   STRUCTURED - Significant residuals remain after Model D\n"
    else:
        summary += "   PARTIALLY STRUCTURED - Some residual structure exists\n"
    
    summary += f"""
2. WHICH VARIABLE BEST EXPLAINS RESIDUALS?
"""
    
    if best_candidate is not None:
        summary += f"   Pooled: {best_candidate['variable']} (r = {best_candidate['pooled_r']:.4f})\n"
    if best_tribe is not None:
        summary += f"   TRIBE: {best_tribe['variable']} (r = {best_tribe['system_r']:.4f})\n"
    
    summary += """
3. DOES ADDING X IMPROVE R² SIGNIFICANTLY?
"""
    
    if best_improvement_var is not None and len(valid_improvements) > 0:
        imp = valid_improvements.get(best_improvement_var, 0)
        if imp > 0.1:
            summary += f"   YES - {best_improvement_var} improves R² by {imp:.3f}\n"
        elif imp > 0:
            summary += f"   MARGINAL - {best_improvement_var} improves R² by {imp:.3f}\n"
        else:
            summary += f"   NO - No significant improvement\n"
    else:
        summary += "   NO - Insufficient data for improvement analysis\n"
    
    summary += """
4. IS TRIBE SPECIFICALLY EXPLAINED BY X?
"""
    
    if best_tribe is not None:
        if abs(best_tribe['system_r']) > 0.5:
            summary += f"   YES - {best_tribe['variable']} strongly correlates with TRIBE residuals (r={best_tribe['system_r']:.3f})\n"
        else:
            summary += f"   PARTIAL - {best_tribe['variable']} moderately correlates (r={best_tribe['system_r']:.3f})\n"
    
    summary += """
5. CANDIDATE FOR MISSING STATE VARIABLE?
"""
    
    if best_candidate is not None:
        if abs(best_candidate['pooled_r']) > 0.5:
            summary += f"   {best_candidate['variable'].upper()} is a strong candidate\n"
            summary += f"   - Pooled correlation: r = {best_candidate['pooled_r']:.4f}\n"
        elif abs(best_candidate['pooled_r']) > 0.3:
            summary += f"   {best_candidate['variable'].upper()} is a moderate candidate\n"
        else:
            summary += f"   No strong candidate identified (max |r| = {abs(best_candidate['pooled_r']):.3f})\n"
    
    summary += """
================================================================================
INTERPRETATION
================================================================================

"""
    
    # Determine interpretation
    if best_candidate is not None and abs(best_candidate['pooled_r']) > 0.5:
        summary += f"""SUCCESS: A third state variable has been identified.

{best_candidate['variable'].upper()} explains residual structure in the state-space model.

This suggests the true state-space model is:
    dβ/dlog(k) = F(D_eff, H, {best_candidate['variable']})

The identification of this variable provides:
1. Better prediction of beta dynamics
2. Explanation of TRIBE-specific behavior
3. A more complete theoretical framework

"""
    elif best_candidate is not None and abs(best_candidate['pooled_r']) > 0.3:
        summary += f"""PARTIAL SUCCESS: A moderate candidate variable identified.

{best_candidate['variable'].upper()} partially explains residual structure.

The relationship is:
    - Pooled correlation: r = {best_candidate['pooled_r']:.4f}
    - Effect may be system-specific

Further investigation needed to determine if this variable:
1. Has a causal role
2. Is a proxy for another latent variable
3. Represents a boundary condition

"""
    else:
        summary += """LIMITED SUCCESS: No strong candidate variable identified.

Residuals may represent:
1. Numerical noise in derivative estimation
2. System-specific boundary effects
3. Fundamental randomness in beta dynamics

Recommendations:
- Increase sampling resolution
- Test additional candidate variables
- Consider hierarchical models

"""
    
    summary += f"""================================================================================
FILES GENERATED
================================================================================

/v26_residual_analysis/data/residuals.csv              - Residual values
/v26_residual_analysis/data/candidates.csv             - Candidate variables
/v26_residual_analysis/data/correlation_results.csv   - Correlation analysis
/v26_residual_analysis/data/augmented_model_results.csv - Model improvements
/v26_residual_analysis/plots/residual_vs_candidates.png - Scatter plots
/v26_residual_analysis/plots/correlation_heatmap.png   - Heatmap
/v26_residual_analysis/plots/r2_improvement.png        - R² comparison

================================================================================
"""
    
    with open(OUTPUT_DIR / "residual_analysis_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
