# Consolidated Summary - SGP-Tribe3 Multiscale Dimensionality Project
# Last Updated: April 12, 2026

## PAPERS CREATED

### Paper 3: `manuscript/paper3_generic_origin/paper3.tex`
- Title: "Dissecting Generic and System-Specific Components of Multiscale Dimensionality Profiles"
- 8 pages
- Key findings: Growth is generic, saturation is contingent, residuals show structure

### Paper 4: `manuscript/paper4_constraint_limits/paper4.tex`
- Title: "Multiscale Dimensionality Profiles Arise from Generic Statistical Structure"
- Author: Mark Rowe Traver
- 7 pages

### Paper 5: `manuscript/paper5_residual_structure/paper5.tex`
- Title: "Characterizing Residual Structure in Multiscale Dimensionality Profiles"
- 9 pages

### Paper 6: `manuscript/paper6_mechanism/paper6.tex`
- Title: "Mechanism of Multiscale Dimensionality Profile Formation: Sigmoid Parameters as Universal System Descriptors"
- 7 pages
- Key findings: Sigmoid parameters fully encode classification, mechanism identified

---

## KEY FINDINGS ACROSS ALL PAPERS

### 1. Low-Dimensional Structure (Paper 5)
- PC1 captures 98.9% of variance in residuals
- Single component sufficient for representation

### 2. Functional Form (Paper 5)
- Sigmoid fits achieve R² = 0.999
- Consistent across all systems

### 3. Classification (Paper 5)
- 80% accuracy using residual features
- Sparse separates from other systems (2 clusters)

### 4. Validation Results (Paper 5 Validation)

| Experiment | Result | Classification |
|------------|--------|---------------|
| Normalization Ablation | PC1 persists | GENUINE |
| Null-of-Null | Null shows 48.9% PC1 | GENUINE |
| Shuffle Control | Sigmoid R² drops to 0.19 | GENUINE |
| Component Analysis | PC1 accuracy 80% | GENUINE |
| Sigmoid Robustness | Real R²=1.0, Random R²=0.12 | GENUINE |

**FINAL: STRUCTURE IS LARGELY GENUINE**

### 5. PC1 Interpretation
- **Amplitude features**: max r = 0.999 with PC1
- PC1 loadings are uniform across k
- Controls: Null shows no correlation, Shuffle shows 0.73

### 6. Amplitude Normalization Test
- PC1 variance: 98.9% → 90-93% after normalization
- Classification accuracy: UNCHANGED at 80%
- **Classification: "Structure is primarily amplitude-driven with significant shape contribution"**

### 7. Shape Randomization Falsification
- Phase randomization: 80% → 40% (drops to chance)
- Permuted+smoothed: 80% → 40%
- Sigmoid-only: **80% preserved**
- **Classification: "Driven by coarse functional form (sigmoid parameters)"**

### 8. Parameter-Only Classification
- Using ONLY A, k0, beta: **80% accuracy**
- Identical to full feature model
- Most important: beta (steepness)
- Noise robust at 20% noise level

### 9. Cross-Fit Falsification
- Original: 80%, Cross-fit: 80%
- **Parameters are GENERIC within cluster**
- Sparse separates (Cluster 1 vs Cluster 0)
- Parameters encode CLUSTER membership, not individual system

### 10. Parameter Sweep
- **Curvature**: Controls A, beta, k0 (r=±1.0)
- **Noise**: Controls all parameters (r≈±0.97)
- **Dimensionality**: Affects k0 (r=-0.76)
- Sparsity/Correlation: weak effects

### 11. Disentanglement Test (Curvature vs Noise)

**Design**: 3 curvature × 4 noise = 12 conditions, 20 replicates

**Two-Way ANOVA Results**:

| Parameter | Curvature η² | Noise η² | Dominant Factor |
|-----------|--------------|----------|-----------------|
| A | 0.079 | **0.918** | Noise |
| k0 | **0.533** | 0.458 | Curvature |
| β | 0.408 | **0.582** | Noise |

**Overall Effects**:
- Curvature η² = 0.340
- Noise η² = 0.653
- Ratio (Curv/Noise) = 0.52

**FINAL CLASSIFICATION**: "Effects are partially separable (context-dependent)"

**KEY FINDING**: 
- **Noise is the primary determinant of sigmoid parameters** (η²=0.65 vs 0.34)
- Curvature fine-tunes the parameter values
- k0 is the only parameter where curvature dominates
- No significant interaction effects (effects are additive)

---

## DIRECTORY STRUCTURE

```
experiments/paper5_validation/
├── normalization_ablation/
│   ├── results.csv
│   ├── norm_comparison.pdf
│   └── interpretation.txt
├── null_of_null/
│   ├── results.csv
│   ├── null_comparison.pdf
│   └── interpretation.txt
├── shuffle_control/
│   ├── results.csv
│   ├── shuffle_comparison.pdf
│   └── interpretation.txt
├── component_analysis/
│   ├── results.csv
│   ├── component_comparison.pdf
│   └── interpretation.txt
├── sigmoid_tests/
│   ├── results.csv
│   ├── sigmoid_comparison.pdf
│   └── interpretation.txt
├── stability_stats/
│   ├── results.csv
│   ├── stability_comparison.pdf
│   └── interpretation.txt
├── pc1_analysis/
│   ├── pca_results.csv
│   ├── features.csv
│   ├── correlations.csv
│   ├── control_comparison.csv
│   ├── pc1_loadings.pdf
│   ├── plots/
│   └── interpretation.txt
├── amplitude_test/
│   ├── normalized_mean.csv
│   ├── normalized_auc.csv
│   ├── pca_variance.csv
│   ├── classification_results.csv
│   ├── shape_metrics.csv
│   ├── summary.csv
│   ├── amplitude_comparison.pdf
│   └── interpretation.txt
├── shape_test/
│   ├── controls/
│   │   ├── original.csv
│   │   ├── phase_randomized.csv
│   │   ├── permuted_smoothed.csv
│   │   └── sigmoid_only.csv
│   ├── classification_results.csv
│   ├── pca_results.csv
│   ├── similarity.csv
│   ├── summary.csv
│   ├── shape_falsification.pdf
│   └── interpretation.txt
├── parameter_test/
│   ├── parameters.csv
│   ├── parameters_normalized.csv
│   ├── classification_results.csv
│   ├── accuracy_comparison.csv
│   ├── feature_importance.csv
│   ├── noise_robustness.csv
│   ├── summary.csv
│   ├── plots/
│   └── interpretation.txt
├── cross_fit/
│   ├── parameter_swaps.csv
│   ├── cross_fit_classification.csv
│   ├── accuracy_comparison.csv
│   ├── distance_analysis.csv
│   ├── confusion_analysis.csv
│   ├── summary.csv
│   ├── plots/
│   └── interpretation.txt
└── parameter_sweep/
    ├── parameter_sweep_results.csv
    ├── correlation_table.csv
    ├── parameter_vs_property_plots.pdf
    ├── correlation_heatmap.pdf
    └── interpretation.txt
└── disentanglement/
    ├── data/
    │   ├── parameter_grid.csv
    │   ├── parameter_means.csv
    │   ├── anova_results.csv
    │   └── interaction_effects.csv
    ├── plots/
    │   ├── heatmaps.pdf
    │   ├── slice_plots.pdf
    │   └── interaction_plots.pdf
    └── interpretation.txt
```

---

## KEY DATA

### Sigmoid Parameters (from 5 systems)
| System | A | k0 | beta |
|--------|---|-----|------|
| TRIBE | 7.80 | 22.59 | 0.138 |
| Hierarchical | 7.57 | 24.16 | 0.133 |
| Correlated | 7.49 | 25.76 | 0.134 |
| Sparse | 8.12 | 19.27 | 0.165 |
| CurvedManifold | 7.74 | 23.50 | 0.134 |

---

## FINAL CONCLUSIONS

1. **Residual dimensionality profiles contain genuine structure**
2. **Sigmoid parameters (A, k0, beta) fully encode classification (80% accuracy)**
3. **Parameters are controlled by: curvature, noise, dimensionality**
4. **Classification depends on sigmoid parameters, not fine structure**
5. **Sparse system is distinct (Cluster 1 vs Cluster 0)**
6. **Noise is primary driver of sigmoid parameters** (η²=0.65 vs curvature η²=0.34)
7. **Curvature only dominates for k0** (midpoint parameter)

---

## NEXT STEPS

1. ~~Write Paper 6 (mechanism paper)~~ **COMPLETED**
2. Integrate with SFH-SGP framework
3. Prepare submission materials (Network Neuroscience target)
