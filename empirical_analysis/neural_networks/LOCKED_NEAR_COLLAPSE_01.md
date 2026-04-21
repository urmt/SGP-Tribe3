# SFH-SGP_NEAR_COLLAPSE_01 (LOCKED)

## Research Question
Does std(Dk) scale continuously with noise level?

## System
- Base: Logistic map at r = 3.5 (periodic)
- Noise: σ = 0 to 0.01 (20 levels)

## Results

| sigma | std_Dk |
|-------|--------|
| 0.000 | 0.0000 |
| 0.001 | 0.0924 |
| 0.005 | 0.0919 |
| 0.010 | 0.0913 |

## Analysis

| Metric | Value |
|--------|-------|
| Correlation (σ vs std_Dk) | 0.39 |
| Monotonically increasing | No |
| Coefficient of variation | 0.23 |

---

## Decision

**Case C: NO CLEAR RELATIONSHIP**

std(Dk) shows:
1. **Abrupt jump** from 0 to ~0.09 at tiny noise (σ > 0)
2. **Plateau** - remains constant (~0.09) regardless of noise level
3. **No smooth scaling** with noise

This indicates the periodic→non-periodic transition is **DISCONTINUOUS**, not continuous.

---

## Implications

1. D(k) cannot be used as continuous "distance from periodicity" estimator
2. The collapse represents a **bifurcation** (phase transition), not a gradual degradation
3. This connects to dynamical systems theory: stable periodic orbits have finite noise tolerance

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**