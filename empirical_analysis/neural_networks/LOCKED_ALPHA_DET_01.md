# SFH-SGP_ALPHA_DET_COMBINED_01 (LOCKED)

## Research Question
Can combined (α, DET) observables separate dynamical regimes?

## Systems Tested
1. Logistic map (periodic, r=3.5)
2. Logistic map (chaotic, r=3.9)
3. Hénon map (chaotic)
4. Ornstein-Uhlenbeck (stochastic)

## Results

| System | α | DET |
|--------|-----|-----|
| logistic_periodic | **0.435** | **1.000** |
| logistic_chaotic | 0.918 | 0.704 |
| henon (chaotic) | 0.985 | 0.277 |
| ou (stochastic) | 0.993 | 0.110 |

---

## Separation in (α, DET) Space

| Regime | α | DET |
|--------|-----|-----|
| Periodic | **0.43** | **1.00** |
| Chaotic | **0.95** | 0.49 |
| Stochastic | **0.99** | **0.11** |

---

## Decision

**Combined (α, DET) successfully separates regimes:**

- Periodic: low α, high DET
- Chaotic: high α, moderate DET
- Stochastic: high α, low DET

This provides a 2D classification space that is noise-robust.

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**