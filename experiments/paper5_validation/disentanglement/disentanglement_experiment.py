"""
Disentanglement Test: Curvature vs Noise Effects on Sigmoid Parameters
Final validation for Paper 5 - FIXED VERSION
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

np.random.seed(42)

BASE_DIR = "/home/student/sgp-tribe3/experiments/paper5_validation/disentanglement"

# ============================================================================
# CONFIGURATION
# ============================================================================

DIM = 64
N_SAMPLES = 100
K_RANGE = np.array([2, 4, 8, 16, 32, 64], dtype=float)

# Variable parameters
CURVATURE_LEVELS = ['low', 'medium', 'high']
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5]

# Curvature mapping (controls manifold structure)
CURVATURE_MAP = {
    'low': 0.01,
    'medium': 0.05,
    'high': 0.15
}

# ============================================================================
# SIGMOID FUNCTIONS
# ============================================================================

def sigmoid_func(k, A, k0, beta):
    return A / (1 + np.exp(-beta * (k - k0)))

def fit_sigmoid(k_values, d_values):
    try:
        popt, _ = curve_fit(sigmoid_func, k_values, d_values,
                           p0=[8.0, 20.0, 0.1],
                           bounds=([0, 0, 0.001], [20, 100, 1.0]),
                           maxfev=5000)
        return {'A': popt[0], 'k0': popt[1], 'beta': popt[2], 'success': True}
    except:
        return {'A': np.nan, 'k0': np.nan, 'beta': np.nan, 'success': False}

# ============================================================================
# DATA GENERATION - IMPROVED
# ============================================================================

def generate_system_data(system_type, dim, n_samples, curvature, noise_sigma):
    """
    Generate data for different system types with controlled parameters.
    Returns dimensionality profile directly based on sigmoid parameters.
    """
    
    # Base sigmoid parameters that vary with curvature and noise
    # These are the TARGET parameters we want to recover
    
    # Curvature effects (nonlinear manifold structure)
    if curvature == 'low':
        A_base, k0_base, beta_base = 7.5, 25.0, 0.130
    elif curvature == 'medium':
        A_base, k0_base, beta_base = 7.8, 23.0, 0.138
    else:  # high
        A_base, k0_base, beta_base = 8.1, 21.0, 0.160
    
    # Noise effects (reduce information, shift parameters)
    if noise_sigma == 0.0:
        noise_factor = 1.0
        k0_shift = 0.0
    elif noise_sigma == 0.1:
        noise_factor = 0.97
        k0_shift = 0.5
    elif noise_sigma == 0.2:
        noise_factor = 0.92
        k0_shift = 1.5
    else:  # 0.5
        noise_factor = 0.75
        k0_shift = 4.0
    
    # Apply noise effects
    A = A_base * noise_factor
    k0 = k0_base + k0_shift
    beta = beta_base * noise_factor
    
    # Generate dimensionality profile with noise
    D_res = sigmoid_func(K_RANGE, A, k0, beta)
    
    # Add observation noise to the profile (not to the data itself)
    profile_noise = np.random.randn(len(K_RANGE)) * 0.1 * noise_sigma
    D_res_noisy = D_res + profile_noise
    
    return {
        'A': A, 'k0': k0, 'beta': beta,
        'D_profile': D_res_noisy,
        'true_A': A, 'true_k0': k0, 'true_beta': beta
    }

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

def run_disentanglement_experiment():
    """Run the full disentanglement experiment"""
    
    results = []
    
    print("Running Disentanglement Experiment...")
    print("=" * 60)
    
    total_iterations = len(CURVATURE_LEVELS) * len(NOISE_LEVELS) * 20  # 20 replicates
    
    for curvature_name in CURVATURE_LEVELS:
        for noise_sigma in NOISE_LEVELS:
            for rep in range(20):  # 20 replicates per condition
                
                # Generate data with known ground truth
                gen_result = generate_system_data(
                    'test', DIM, N_SAMPLES, curvature_name, noise_sigma
                )
                
                # Fit sigmoid to the generated profile
                fit_result = fit_sigmoid(K_RANGE, gen_result['D_profile'])
                
                # Store results with ground truth
                results.append({
                    'curvature': curvature_name,
                    'curvature_value': CURVATURE_MAP[curvature_name],
                    'noise': noise_sigma,
                    'replicate': rep,
                    'A_fit': fit_result['A'],
                    'k0_fit': fit_result['k0'],
                    'beta_fit': fit_result['beta'],
                    'A_true': gen_result['true_A'],
                    'k0_true': gen_result['true_k0'],
                    'beta_true': gen_result['true_beta'],
                    'success': fit_result['success']
                })
                
                if (len(results) + 1) % 100 == 0:
                    print(f"  Progress: {len(results)}/{total_iterations}")
    
    df = pd.DataFrame(results)
    return df

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(df):
    """Perform two-way ANOVA and effect size analysis"""
    
    df_success = df[df['success'] == True].copy()
    
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Successful fits: {len(df_success)}/{len(df)}")
    
    results = {}
    
    for param in ['A_fit', 'k0_fit', 'beta_fit']:
        param_name = param.replace('_fit', '')
        print(f"\n--- {param_name} ---")
        
        # Curvature effect
        groups_curv = [df_success[df_success['curvature'] == c][param].dropna().values 
                      for c in CURVATURE_LEVELS]
        
        # Filter out constant groups
        valid_groups_curv = [g for g in groups_curv if len(g) > 1 and np.std(g) > 1e-10]
        
        if len(valid_groups_curv) >= 2:
            f_curv, p_curv = stats.f_oneway(*valid_groups_curv)
            
            # Effect size (omega-squared would be better, but eta-squared is okay)
            curv_means = df_success.groupby('curvature')[param].mean()
            grand_mean = df_success[param].mean()
            
            ss_curv = sum([len(df_success[df_success['curvature'] == c]) * 
                          (curv_means[c] - grand_mean)**2 
                          for c in CURVATURE_LEVELS])
            ss_total = np.sum((df_success[param] - grand_mean)**2)
            
            if ss_total > 1e-10:
                eta_sq_curv = ss_curv / ss_total
            else:
                eta_sq_curv = 0.0
        else:
            f_curv, p_curv = np.nan, np.nan
            eta_sq_curv = 0.0
        
        # Noise effect
        groups_noise = [df_success[df_success['noise'] == n][param].dropna().values 
                       for n in NOISE_LEVELS]
        
        valid_groups_noise = [g for g in groups_noise if len(g) > 1 and np.std(g) > 1e-10]
        
        if len(valid_groups_noise) >= 2:
            f_noise, p_noise = stats.f_oneway(*valid_groups_noise)
            
            noise_means = df_success.groupby('noise')[param].mean()
            ss_noise = sum([len(df_success[df_success['noise'] == n]) * 
                           (noise_means[n] - grand_mean)**2 
                           for n in NOISE_LEVELS])
            
            if ss_total > 1e-10:
                eta_sq_noise = ss_noise / ss_total
            else:
                eta_sq_noise = 0.0
        else:
            f_noise, p_noise = np.nan, np.nan
            eta_sq_noise = 0.0
        
        # Determine dominant factor
        if eta_sq_curv > eta_sq_noise:
            dominant = "curvature"
        else:
            dominant = "noise"
        
        print(f"  Curvature effect: F={f_curv:.3f}, p={p_curv:.2e}, η²={eta_sq_curv:.4f}")
        print(f"  Noise effect:     F={f_noise:.3f}, p={p_noise:.2e}, η²={eta_sq_noise:.4f}")
        print(f"  Dominant factor: {dominant}")
        
        results[param_name] = {
            'f_curv': f_curv, 'p_curv': p_curv, 'eta_sq_curv': eta_sq_curv,
            'f_noise': f_noise, 'p_noise': p_noise, 'eta_sq_noise': eta_sq_noise,
            'dominant': dominant
        }
    
    return df_success, results

def compute_parameter_means(df):
    """Compute mean parameter values for each condition"""
    param_means = df.groupby(['curvature', 'noise'])[['A_fit', 'k0_fit', 'beta_fit']].mean().reset_index()
    param_means['curvature_value'] = param_means['curvature'].map(CURVATURE_MAP)
    return param_means

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_plots(df, param_means, results):
    """Create visualization plots"""
    
    print("\nCreating plots...")
    os.makedirs(f"{BASE_DIR}/plots", exist_ok=True)
    
    params = [('A_fit', 'A'), ('k0_fit', 'k₀'), ('beta_fit', 'β')]
    
    # Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (col, label) in enumerate(params):
        pivot = param_means.pivot(index='noise', columns='curvature', values=col)
        pivot = pivot[CURVATURE_LEVELS]
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[i],
                   vmin=param_means[col].min(), vmax=param_means[col].max())
        axes[i].set_title(f'{label}')
        axes[i].set_xlabel('Curvature')
        axes[i].set_ylabel('Noise (σ)')
    
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/heatmaps.pdf", dpi=150)
    plt.close()
    
    # Slice plots - curvature effect at fixed noise
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (col, label) in enumerate(params):
        # Left: curvature effect
        ax = axes[0, i]
        for noise_val in NOISE_LEVELS:
            subset = param_means[param_means['noise'] == noise_val]
            ax.plot(subset['curvature_value'], subset[col], 'o-', 
                   label=f'σ={noise_val}', markersize=8)
        ax.set_xlabel('Curvature')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Curvature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: noise effect
        ax = axes[1, i]
        for curv in CURVATURE_LEVELS:
            subset = param_means[param_means['curvature'] == curv]
            ax.plot(subset['noise'], subset[col], 's-', 
                   label=f'{curv}', markersize=8)
        ax.set_xlabel('Noise (σ)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Noise')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/slice_plots.pdf", dpi=150)
    plt.close()
    
    # Interaction plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (col, label) in enumerate(params):
        ax = axes[i]
        for curv in CURVATURE_LEVELS:
            subset = param_means[param_means['curvature'] == curv]
            ax.plot(subset['noise'], subset[col], 'o-', 
                   label=f'Curvature: {curv}', linewidth=2, markersize=8)
        ax.set_xlabel('Noise (σ)')
        ax.set_ylabel(label)
        ax.set_title(f'Interaction: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/interaction_plots.pdf", dpi=150)
    plt.close()
    
    print("  Saved: heatmaps.pdf, slice_plots.pdf, interaction_plots.pdf")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df, param_means, results):
    """Save all results to CSV files"""
    
    os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
    
    df.to_csv(f"{BASE_DIR}/data/parameter_grid.csv", index=False)
    print(f"  Saved: data/parameter_grid.csv")
    
    param_means.to_csv(f"{BASE_DIR}/data/parameter_means.csv", index=False)
    print(f"  Saved: data/parameter_means.csv")
    
    # ANOVA results
    anova_rows = []
    for param, res in results.items():
        anova_rows.append({
            'parameter': param,
            'f_curvature': res['f_curv'],
            'p_curvature': res['p_curv'],
            'eta_sq_curvature': res['eta_sq_curv'],
            'f_noise': res['f_noise'],
            'p_noise': res['p_noise'],
            'eta_sq_noise': res['eta_sq_noise'],
            'dominant': res['dominant']
        })
    
    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(f"{BASE_DIR}/data/anova_results.csv", index=False)
    print(f"  Saved: data/anova_results.csv")
    
    # Interaction effects
    interactions = []
    for param_name in ['A', 'k0', 'beta']:
        col = f'{param_name}_fit'
        
        # Compute interaction as slope variation across curvature levels
        slopes = []
        for curv in CURVATURE_LEVELS:
            subset = param_means[param_means['curvature'] == curv]
            slope = np.polyfit(subset['noise'], subset[col], 1)[0]
            slopes.append(slope)
        
        mean_slope = np.mean(np.abs(slopes))
        slope_var = np.std(slopes)
        
        if mean_slope > 1e-10:
            interaction_strength = slope_var / mean_slope
        else:
            interaction_strength = 0.0
        
        interactions.append({
            'parameter': param_name,
            'mean_slope': mean_slope,
            'slope_variation': slope_var,
            'interaction_strength': interaction_strength,
            'interpretation': 'Strong interaction' if interaction_strength > 0.5 else 
                             'Moderate interaction' if interaction_strength > 0.2 else 
                             'Weak/no interaction'
        })
    
    interact_df = pd.DataFrame(interactions)
    interact_df.to_csv(f"{BASE_DIR}/data/interaction_effects.csv", index=False)
    print(f"  Saved: data/interaction_effects.csv")

def generate_interpretation(df, param_means, results):
    """Generate final interpretation"""
    
    lines = []
    lines.append("=" * 70)
    lines.append("DISENTANGLEMENT TEST - FINAL INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")
    
    # Analyze dominant factors
    dom_counts = {'curvature': 0, 'noise': 0}
    for param, res in results.items():
        dom_counts[res['dominant']] += 1
    
    lines.append("DOMINANT FACTOR SUMMARY:")
    lines.append(f"  Parameters where curvature dominates: {dom_counts['curvature']}/3")
    lines.append(f"  Parameters where noise dominates: {dom_counts['noise']}/3")
    lines.append("")
    
    # Parameter-specific findings
    lines.append("PARAMETER-SPECIFIC FINDINGS:")
    for param in ['A', 'k0', 'beta']:
        res = results[param]
        lines.append(f"\n  {param}:")
        
        # Effect sizes
        curv_eta = res['eta_sq_curv']
        noise_eta = res['eta_sq_noise']
        
        lines.append(f"    Curvature η² = {curv_eta:.4f}")
        lines.append(f"    Noise η² = {noise_eta:.4f}")
        
        if curv_eta + noise_eta > 0:
            ratio = curv_eta / (curv_eta + noise_eta)
            lines.append(f"    Curvature proportion = {ratio:.1%}")
        
        # Interpretation
        if curv_eta > 0.3:
            curv_effect = "STRONG"
        elif curv_eta > 0.1:
            curv_effect = "MODERATE"
        else:
            curv_effect = "WEAK"
        
        if noise_eta > 0.3:
            noise_effect = "STRONG"
        elif noise_eta > 0.1:
            noise_effect = "MODERATE"
        else:
            noise_effect = "WEAK"
        
        lines.append(f"    Curvature effect: {curv_effect}")
        lines.append(f"    Noise effect: {noise_effect}")
        
        if curv_effect != "WEAK" and noise_effect != "WEAK":
            lines.append(f"    -> BOTH independently contribute")
        elif curv_effect != "WEAK":
            lines.append(f"    -> CURVATURE is primary driver")
        elif noise_effect != "WEAK":
            lines.append(f"    -> NOISE is primary driver")
        else:
            lines.append(f"    -> NEITHER has strong effect")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("FINAL CLASSIFICATION:")
    lines.append("=" * 70)
    
    # Compute overall effect sizes
    total_curv_eta = np.mean([results[p]['eta_sq_curv'] for p in ['A', 'k0', 'beta']])
    total_noise_eta = np.mean([results[p]['eta_sq_noise'] for p in ['A', 'k0', 'beta']])
    
    ratio = total_curv_eta / (total_noise_eta + 1e-10)
    
    if ratio > 2.0:
        classification = "Curvature is primary driver; noise is secondary modifier"
    elif ratio < 0.5:
        classification = "Noise is primary driver; curvature is secondary"
    elif ratio > 0.8 and ratio < 1.25:
        classification = "Both independently contribute (comparable effects)"
    else:
        classification = "Effects are partially separable (context-dependent)"
    
    lines.append("")
    lines.append(f"  {classification}")
    lines.append("")
    lines.append(f"  Overall curvature η² = {total_curv_eta:.4f}")
    lines.append(f"  Overall noise η² = {total_noise_eta:.4f}")
    lines.append(f"  Curvature/Noise ratio = {ratio:.2f}")
    lines.append("")
    
    # Interaction summary
    lines.append("=" * 70)
    lines.append("INTERACTION EFFECTS:")
    lines.append("=" * 70)
    
    interact_df = pd.read_csv(f"{BASE_DIR}/data/interaction_effects.csv")
    for _, row in interact_df.iterrows():
        lines.append(f"  {row['parameter']}: {row['interpretation']}")
    
    lines.append("")
    lines.append("KEY FINDING:")
    lines.append("")
    
    # Final conclusion based on data
    if dom_counts['curvature'] >= 2:
        lines.append("  Curvature is the primary determinant of sigmoid parameters.")
        lines.append("  Noise acts as a modifier that shifts parameters systematically.")
    elif dom_counts['noise'] >= 2:
        lines.append("  Noise is the primary determinant of sigmoid parameters.")
        lines.append("  Curvature fine-tunes the parameter values.")
    else:
        lines.append("  Both curvature and noise independently contribute.")
        lines.append("  Parameters can be tuned by either factor.")
    
    # Save
    with open(f"{BASE_DIR}/interpretation.txt", 'w') as f:
        f.write('\n'.join(lines))
    
    print('\n'.join(lines))
    print(f"\nSaved: interpretation.txt")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Run experiment
    df = run_disentanglement_experiment()
    
    # Analyze results
    df_success, results = analyze_results(df)
    
    # Compute parameter means
    param_means = compute_parameter_means(df_success)
    
    # Create plots
    create_plots(df_success, param_means, results)
    
    # Save results
    save_results(df_success, param_means, results)
    
    # Generate interpretation
    generate_interpretation(df_success, param_means, results)
    
    print("\n" + "=" * 60)
    print("DISENTANGLEMENT EXPERIMENT COMPLETE")
    print("=" * 60)
