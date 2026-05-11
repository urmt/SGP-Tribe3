# Phase 246: Artifact Risk Report

Generated: 2026-05-11 11:48:52

## Risks

### 1. Ablation Reconstruction Normalization
- **Severity**: MODERATE
- **Description**: Ablated matrices are renormalized to [-1,1], which may mask impairment.
- **Mitigation**: Consistent across real and control ablations.

### 2. Control 1 vs Targeted Ablation
- **Severity**: LOW
- **Description**: Random eigenvector removal removes different amounts of variance.
- **Mitigation**: 20 trials averaged per control condition.

### 3. Small Matrix (8x8)
- **Severity**: MODERATE
- **Description**: 8 eigendimensions limits ablation to max 3 removals.
- **Mitigation**: Consistent across all systems.

### 4. Synthesized Recovery for Gaussian Control
- **Severity**: LOW
- **Description**: Gaussian synthetic baseline may not reflect null recovery.
- **Mitigation**: 4 additional distinct controls provide cross-validation.

### Threshold Log
- Adjacency threshold: 70th percentile
- Ablation: 0, 1, 2, 3 eigenvectors removed
- 20 random trials per control condition
- JS divergence for spectral distortion
- Causal impact = real - control impairment
