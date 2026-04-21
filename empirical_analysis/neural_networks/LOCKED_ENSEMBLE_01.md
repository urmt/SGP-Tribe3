# SFH-SGP_ENSEMBLE_01 (LOCKED)

## Research Question
Can segment-based ensemble method provide noise-robust collapse detection?

## System
- Logistic map: r = [3.5 (periodic), 3.57 (boundary), 3.9 (chaotic)]
- Noise: σ ∈ [0, 1e-4, 1e-3]
- 50 segments × 300 points each

## Results

| r | σ | collapse_prob | mean_std_Dk |
|---|---|--------------|-------------|
| 3.5 | 0.0 | **1.0** | 0.00 |
| 3.5 | 1e-4 | 0.0 | 0.11 |
| 3.5 | 1e-3 | 0.0 | 0.11 |
| 3.57 | 0.0 | 0.0 | 0.95 |
| 3.57 | 1e-4 | 0.0 | 0.69 |
| 3.57 | 1e-3 | 0.0 | 0.28 |
| 3.9 | 0.0 | 0.0 | 0.29 |
| 3.9 | 1e-4 | 0.0 | 0.30 |
| 3.9 | 1e-3 | 0.0 | 0.31 |

---

## Decision

**Statistical extension DOES NOT solve noise robustness**

Collapse probability remains 0 for any σ > 0.

Interesting observation:
- mean_std_Dk at boundary (r=3.57) DECREASES with noise (0.95 → 0.28)
- Suggests noise "regularizes" boundary dynamics

---

## Interpretation

The D(k) collapse mechanism is fundamentally noise-fragile.
Segment-based ensembling does not provide robustness.

Alternative approaches needed.

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**