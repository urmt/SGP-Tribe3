# Phase 245: Artifact Risk Report

Generated: 2026-05-11 11:29:32

## Risks

### 1. Reconstruction Blending Artifact
- **Severity**: MODERATE
- **Description**: Reconstruction blends 70% spectral + 30% centroid. Ratio choice affects scores.
- **Mitigation**: Fixed ratio applied identically across all systems and controls.

### 2. MST Enforcement
- **Severity**: MODERATE
- **Description**: MST edges are enforced at 0.9 correlation, which may over-constrain.
- **Mitigation**: Consistent across all conditions.

### 3. Saturation Definition
- **Severity**: LOW
- **Description**: Saturation threshold at 0.02 gain between consecutive k values.
- **Mitigation**: Explicit threshold logged.

### 4. Small Graph (8 nodes)
- **Severity**: MODERATE
- **Description**: 8-channel data limits max rank reconstruction.
- **Mitigation**: k values capped at n_ch=8 for practical rank.

### 5. EEG Recovery Definition
- **Severity**: LOW
- **Description**: EEG uses cross-segment comparison.
- **Mitigation**: Kuramoto/Logistic provide true dynamical validation.
