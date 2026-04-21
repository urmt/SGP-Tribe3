# SFH-SGP_BOUNDARY_NOISE_ROBUSTNESS_01 (LOCKED)

## Research Question
How does collapse probability behave near the chaos boundary under noise?

## System
- Logistic map: r ∈ [3.55, 3.58] (spanning boundary at r ≈ 3.569)
- Noise: σ ∈ [0, 1e-4, 5e-4, 1e-3, 5e-3]
- 20 initial conditions per (r, σ)

## Results

| r | σ=0 | σ>0 |
|----|-----|-----|
| 3.550 | 1.0 | 0.0 |
| 3.560 | 1.0 | 0.0 |
| 3.565 | 1.0 | 0.0 |
| 3.568 | 1.0 | 0.0 |
| 3.570 | 0.0 | 0.0 |
| 3.575 | 0.0 | 0.0 |
| 3.580 | 0.0 | 0.0 |

---

## Decision

**Boundary is unstable under noise**

Key findings:
1. At σ=0: Collapse at r = 3.55-3.568 (periodic), no collapse at r >= 3.57
2. At ANY σ > 0: Collapse = 0 everywhere
3. Boundary at r ≈ 3.569 (matches Lyapunov zero-crossing)

---

## Interpretation

D(k) collapse is:
- Binary: detects periodic OR not periodic
- Extremely noise-sensitive: any perturbation destroys signal
- Aligned with Lyapunov exponent at σ = 0
- Not recoverable under noise

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**