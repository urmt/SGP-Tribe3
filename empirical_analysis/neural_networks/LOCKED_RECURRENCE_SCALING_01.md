# SFH-SGP_RECURRENCE_SCALING_01 (LOCKED)

## Research Question
How does recurrence_rate scale with ε across dynamical regimes?

## System
- Logistic map: r = [3.5 (periodic), 3.57 (boundary), 3.9 (chaotic)]
- Noise: σ = 1e-3
- ε sweep: 1e-4 to 1e-1 (15 points)

## Results

| r | Regime | α (scaling exponent) |
|---|--------|---------------------|
| 3.5 | Periodic | **0.428** |
| 3.57 | Boundary | 0.705 |
| 3.9 | Chaotic | **0.909** |

---

## Key Finding

**Scaling law separates regimes:**

- Periodic: Low α (0.43) → rapid saturation at small ε
- Chaotic: High α (0.91) → gradual increase with ε

Recurrence follows: recurrence_rate ∝ ε^α

---

## Interpretation

The scaling exponent α is a **regime signature**:
- α < 0.5 → periodic-like (attractor-dominated)
- α > 0.8 → chaotic-like (diffusive)

This provides a quantitative, noise-robust classifier.

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**