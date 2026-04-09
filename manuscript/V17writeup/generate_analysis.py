#!/usr/bin/env python3
"""
V17writeup - Data Analysis and Figure Generation
Computes new metrics and generates tables/figures for revised manuscript
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

BASE = '/home/student/sgp-tribe3/manuscript'

# ============================================================================
# SECTION 1: LOAD ALL DATA
# ============================================================================

print("=" * 80)
print("SECTION 1: LOADING DATA")
print("=" * 80)

# V14 deff_curves (extended k range for V15)
deff_curves_v14 = pd.read_csv(f'{BASE}/v14/deff_curves.csv')
print(f"V14 deff_curves: {len(deff_curves_v14)} rows")

# V16 all_curves (full k range)
all_curves = pd.read_csv(f'{BASE}/v16/all_curves.csv')
print(f"V16 all_curves: {len(all_curves)} rows")

# V16 cross_domain_summary
cross_domain = pd.read_csv(f'{BASE}/v16/cross_domain_summary.csv')
print(f"V16 cross_domain: {len(cross_domain)} rows")

# V17 invariant_summary
invariant = pd.read_csv(f'{BASE}/v17/invariant_summary.csv')
print(f"V17 invariant: {len(invariant)} rows")

# V15 curve_data (TRIBE extended)
curve_data = pd.read_csv(f'{BASE}/v15/curve_data.csv')
print(f"V15 curve_data: {len(curve_data)} rows")

# V14 fit_summary
fit_summary = pd.read_csv(f'{BASE}/v14/fit_summary.csv')
print(f"V14 fit_summary: {len(fit_summary)} rows")

# V12 structural_summary
structural = pd.read_csv(f'{BASE}/v12/structural_summary.csv')
print(f"V12 structural: {len(structural)} rows")

# ============================================================================
# SECTION 2: COMPUTE NEW METRICS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 2: COMPUTING NEW METRICS")
print("=" * 80)

# Organize data by system
systems = all_curves['System'].unique()
print(f"\nSystems: {list(systems)}")

# Ambient dimensions
AMBIENT_DIMS = {
    'TRIBE': 9,
    'hierarchical': 50,
    'correlated': 50,
    'sparse': 50,
    'manifold': 50,
    'uniform_sphere': 50,
    'power_law': 50
}

# Compute metrics for each system
metrics_list = []

for system in systems:
    system_data = all_curves[all_curves['System'] == system].copy()
    k_vals = system_data['k'].values
    D_eff_vals = system_data['D_eff'].values
    
    # Growth rate: slope in log-log space for early k (5-50)
    early_mask = k_vals <= 50
    if np.sum(early_mask) > 2:
        log_k = np.log(k_vals[early_mask])
        log_D = np.log(D_eff_vals[early_mask] + 1e-10)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_D)
        growth_rate = slope
    else:
        growth_rate = np.nan
    
    # Saturation level: mean D_eff for top 10% k values
    top_10_pct = int(len(k_vals) * 0.9)
    if top_10_pct > 0:
        D_sat = np.mean(D_eff_vals[top_10_pct:])
        max_D = np.max(D_eff_vals)
        saturation_ratio = D_sat / max_D if max_D > 0 else np.nan
    else:
        D_sat = np.nan
        saturation_ratio = np.nan
    
    # Peak dimensionality
    peak_Deff = np.max(D_eff_vals)
    k_peak = k_vals[np.argmax(D_eff_vals)]
    
    # Compression ratio
    ambient = AMBIENT_DIMS.get(system, 50)
    compression_ratio = peak_Deff / ambient
    
    # Final D_eff
    final_D = D_eff_vals[-1]
    
    metrics_list.append({
        'System': system,
        'Growth_Rate': growth_rate,
        'Saturation_Ratio': saturation_ratio,
        'Compression_Ratio': compression_ratio,
        'Peak_D_eff': peak_Deff,
        'k_Peak': k_peak,
        'D_Sat': D_sat,
        'Final_D_eff': final_D,
        'Ambient_Dim': ambient
    })
    
    print(f"\n{system}:")
    print(f"  Growth Rate: {growth_rate:.4f}")
    print(f"  Saturation Ratio: {saturation_ratio:.4f}")
    print(f"  Compression Ratio: {compression_ratio:.4f}")
    print(f"  Peak D_eff: {peak_Deff:.2f} at k={k_peak}")
    print(f"  Ambient Dim: {ambient}")

# Create metrics dataframe
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f'{BASE}/V17writeup/tables/table_quantitative.csv', index=False)
print(f"\nSaved: table_quantitative.csv")

# ============================================================================
# SECTION 3: ALIGNMENT MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: ALIGNMENT MATRIX")
print("=" * 80)

# Use systems from all_curves
system_list = list(systems)
n_systems = len(system_list)

# Compute normalized curves for alignment
normalized_curves = {}
for system in system_list:
    # Get k_max from cross_domain or default
    cd_row = cross_domain[cross_domain['System'] == system]
    if len(cd_row) > 0:
        k_max = cd_row['Peak_k'].values[0]
    else:
        k_max = 500
    
    # Get raw curve
    raw = all_curves[all_curves['System'] == system]
    if len(raw) > 0:
        k_raw = raw['k'].values
        D_raw = raw['D_eff'].values
        k_norm = k_raw / k_raw[-1]  # Normalize by actual max
        D_norm = D_raw / np.max(D_raw)
        normalized_curves[system] = {'k_norm': k_norm, 'D_norm': D_norm}

# Compute pairwise correlations
alignment_matrix = np.zeros((n_systems, n_systems))

for i, sys1 in enumerate(system_list):
    for j, sys2 in enumerate(system_list):
        if sys1 in normalized_curves and sys2 in normalized_curves:
            k1, D1 = normalized_curves[sys1]['k_norm'], normalized_curves[sys1]['D_norm']
            k2, D2 = normalized_curves[sys2]['k_norm'], normalized_curves[sys2]['D_norm']
            
            # Interpolate to common grid
            k_common = np.linspace(0.05, 0.95, 20)
            D1_interp = np.interp(k_common, k1, D1)
            D2_interp = np.interp(k_common, k2, D2)
            
            corr, _ = stats.pearsonr(D1_interp, D2_interp)
            alignment_matrix[i, j] = corr
        else:
            alignment_matrix[i, j] = np.nan

# Save alignment matrix
alignment_df = pd.DataFrame(alignment_matrix, index=system_list, columns=system_list)
alignment_df.to_csv(f'{BASE}/V17writeup/tables/alignment_matrix.csv')
print(f"Saved: alignment_matrix.csv")
print(f"\nAlignment Matrix:\n{alignment_df.round(3)}")

# ============================================================================
# SECTION 4: DATASET CHARACTERISTICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: DATASET CHARACTERISTICS")
print("=" * 80)

dataset_chars = []
for system in systems:
    if system == 'TRIBE':
        n_samples = 2000
        ambient = 9
        data_type = 'Cortical model (SGP nodes)'
    elif system in ['hierarchical', 'correlated', 'sparse', 'manifold', 'uniform_sphere', 'power_law']:
        n_samples = 2000
        ambient = 50
        if system == 'hierarchical':
            data_type = 'Synthetic (hierarchical Gaussian)'
        elif system == 'correlated':
            data_type = 'Synthetic (correlated Gaussian)'
        elif system == 'sparse':
            data_type = 'Synthetic (sparse)'
        elif system == 'manifold':
            data_type = 'Synthetic (curved manifold)'
        elif system == 'uniform_sphere':
            data_type = 'Synthetic (uniform sphere)'
        elif system == 'power_law':
            data_type = 'Synthetic (power law)'
    else:
        n_samples = 2000
        ambient = 50
        data_type = 'Unknown'
    
    dataset_chars.append({
        'System': system,
        'N_Samples': n_samples,
        'Ambient_Dim': ambient,
        'Data_Type': data_type
    })

dataset_df = pd.DataFrame(dataset_chars)
dataset_df.to_csv(f'{BASE}/V17writeup/tables/dataset_characteristics.csv', index=False)
print(f"Saved: dataset_characteristics.csv")

# ============================================================================
# SECTION 5: GENERATE FIGURES
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: GENERATING FIGURES")
print("=" * 80)

# FIGURE 1: Unnormalized D_eff curves
fig1, ax1 = plt.subplots(figsize=(10, 6))

colors_fig = plt.cm.tab10(np.linspace(0, 1, len(systems)))
for i, system in enumerate(systems):
    system_data = all_curves[all_curves['System'] == system]
    ax1.plot(system_data['k'], system_data['D_eff'], 'o-', 
             label=system.replace('_', ' ').title(), color=colors_fig[i], 
             markersize=6, alpha=0.8)

ax1.set_xlabel('k (neighborhood size)', fontsize=12)
ax1.set_ylabel('$D_{\\text{eff}}$', fontsize=12)
ax1.set_title('Effective Dimensionality vs. Neighborhood Scale', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([3, 700])

plt.tight_layout()
fig1.savefig(f'{BASE}/V17writeup/figures/unnormalized_curves.png', dpi=300, bbox_inches='tight')
print("Saved: unnormalized_curves.png")
plt.close()

# FIGURE 2: Alignment Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 7))

im = ax2.imshow(alignment_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0, aspect='auto')
ax2.set_xticks(range(n_systems))
ax2.set_yticks(range(n_systems))
ax2.set_xticklabels([s.replace('_', '\n') for s in system_list], rotation=45, ha='right')
ax2.set_yticklabels([s.replace('_', ' ') for s in system_list])

for i in range(n_systems):
    for j in range(n_systems):
        text = ax2.text(j, i, f'{alignment_matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=9)

ax2.set_title('Pairwise Curve Alignment\n(Pearson Correlation)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax2, label='Correlation')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
fig2.savefig(f'{BASE}/V17writeup/figures/alignment_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: alignment_heatmap.png")
plt.close()

# FIGURE 3: Growth Phase Fits
fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
axes3 = axes3.flatten()

for idx, system in enumerate(list(systems)[:6]):
    ax = axes3[idx]
    system_data = all_curves[all_curves['System'] == system]
    k_vals = system_data['k'].values
    D_eff_vals = system_data['D_eff'].values
    
    # Growth phase (k <= 50)
    early_mask = k_vals <= 50
    k_early = k_vals[early_mask]
    D_early = D_eff_vals[early_mask]
    
    ax.scatter(k_early, D_early, s=60, alpha=0.8, zorder=5, label='Data')
    
    # Fit linear in log-log
    log_k = np.log(k_early)
    log_D = np.log(D_early)
    slope, intercept, r, p, se = stats.linregress(log_k, log_D)
    
    # Plot fit
    k_fit = np.linspace(k_early[0], k_early[-1], 50)
    D_fit = np.exp(intercept) * k_fit ** slope
    ax.plot(k_fit, D_fit, '--', color='red', linewidth=2, 
            label=f'Fit (slope={slope:.3f})')
    
    ax.set_xlabel('k')
    ax.set_ylabel('$D_{\\text{eff}}$')
    ax.set_title(system.replace('_', ' ').title(), fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes3[5].axis('off')

plt.suptitle('Growth Phase Analysis: Linear Fits in Log-Log Space', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig(f'{BASE}/V17writeup/figures/growth_phase_fit.png', dpi=300, bbox_inches='tight')
print("Saved: growth_phase_fit.png")
plt.close()

# FIGURE 4: Saturation Plot
fig4, ax4 = plt.subplots(figsize=(10, 6))

for i, system in enumerate(systems):
    system_data = all_curves[all_curves['System'] == system]
    k_vals = system_data['k'].values
    D_eff_vals = system_data['D_eff'].values
    
    k_norm = k_vals / k_vals[-1]
    D_norm = D_eff_vals / np.max(D_eff_vals)
    
    ax4.plot(k_norm, D_norm, 'o-', label=system.replace('_', ' ').title(),
             color=colors_fig[i], alpha=0.8, markersize=5)

# Add saturation threshold line
ax4.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% threshold')

ax4.set_xlabel('$k / k_{\\max}$', fontsize=12)
ax4.set_ylabel('$D_{\\text{eff}} / D_{\\max}$', fontsize=12)
ax4.set_title('Normalized Dimensionality: Saturation Behavior', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1.05])

plt.tight_layout()
fig4.savefig(f'{BASE}/V17writeup/figures/saturation_plot.png', dpi=300, bbox_inches='tight')
print("Saved: saturation_plot.png")
plt.close()

# FIGURE 5: TRIBE Detailed Analysis
fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4))

# Panel A: D_eff vs k for TRIBE
ax_a = axes5[0]
tribe_data = curve_data.copy()
ax_a.plot(tribe_data['k'], tribe_data['D_eff'], 'o-', color='navy', markersize=6)
ax_a.fill_between(tribe_data['k'], 
                   tribe_data['D_eff'] - tribe_data['std'],
                   tribe_data['D_eff'] + tribe_data['std'],
                   alpha=0.3, color='navy')
ax_a.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='Peak (k=200)')
ax_a.set_xlabel('k (neighborhood size)')
ax_a.set_ylabel('$D_{\\text{eff}}$')
ax_a.set_title('A. TRIBE Dimensionality Growth')
ax_a.legend()
ax_a.grid(True, alpha=0.3)

# Panel B: Log-log plot
ax_b = axes5[1]
ax_b.scatter(tribe_data['k'], tribe_data['D_eff'], color='navy', s=40)
ax_b.set_xscale('log')
ax_b.set_xlabel('k (log scale)')
ax_b.set_ylabel('$D_{\\text{eff}}$')
ax_b.set_title('B. Log-Log Relationship')
ax_b.grid(True, alpha=0.3)

# Add power law fit
early_mask = tribe_data['k'] <= 100
k_early = tribe_data.loc[early_mask, 'k'].values
D_early = tribe_data.loc[early_mask, 'D_eff'].values
slope, intercept = np.polyfit(np.log(k_early), np.log(D_early), 1)
k_fit = np.linspace(5, 100, 50)
ax_b.plot(k_fit, np.exp(intercept) * k_fit ** slope, 'r--', linewidth=2,
          label=f'Power law (slope={slope:.2f})')
ax_b.legend()

# Panel C: Growth vs Saturation metrics
ax_c = axes5[2]
tribe_metrics = metrics_df[metrics_df['System'] == 'TRIBE'].iloc[0]
other_metrics = metrics_df[metrics_df['System'] != 'TRIBE']

ax_c.scatter(other_metrics['Growth_Rate'], other_metrics['Compression_Ratio'], 
            s=100, alpha=0.6, label='Synthetic systems', color='gray')
ax_c.scatter([tribe_metrics['Growth_Rate']], [tribe_metrics['Compression_Ratio']], 
            s=200, marker='*', color='navy', label='TRIBE', zorder=5)

ax_c.set_xlabel('Growth Rate (log-log slope)')
ax_c.set_ylabel('Compression Ratio ($D_{\\max}/d$)')
ax_c.set_title('C. TRIBE vs Synthetic Systems')
ax_c.legend()
ax_c.grid(True, alpha=0.3)

plt.suptitle('TRIBE-Specific Dimensionality Analysis', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
fig5.savefig(f'{BASE}/V17writeup/figures/tribe_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: tribe_detailed_analysis.png")
plt.close()

# ============================================================================
# SECTION 6: SAVE PAIRWISE ALIGNMENT TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: PAIRWISE ALIGNMENT TABLE")
print("=" * 80)

# Create pairwise table
pairs = []
for i in range(len(system_list)):
    for j in range(i+1, len(system_list)):
        pairs.append({
            'System_1': system_list[i],
            'System_2': system_list[j],
            'Alignment': alignment_matrix[i, j]
        })

pairwise_df = pd.DataFrame(pairs).sort_values('Alignment', ascending=False)
pairwise_df.to_csv(f'{BASE}/V17writeup/tables/pairwise_alignment.csv', index=False)
print(f"Saved: pairwise_alignment.csv")
print(pairwise_df.to_string())

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated tables:")
print("  - table_quantitative.csv")
print("  - alignment_matrix.csv")
print("  - dataset_characteristics.csv")
print("  - pairwise_alignment.csv")
print("\nGenerated figures:")
print("  - unnormalized_curves.png")
print("  - alignment_heatmap.png")
print("  - growth_phase_fit.png")
print("  - saturation_plot.png")
print("  - tribe_detailed_analysis.png")
print("=" * 80)
