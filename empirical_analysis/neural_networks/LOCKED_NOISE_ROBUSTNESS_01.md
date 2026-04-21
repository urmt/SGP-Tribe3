# SFH-SGP_NOISE_ROBUSTNESS_01 (LOCKED)

## Research Question
At what noise level does D(k) collapse break?

## System
- Base: Logistic map at r = 3.5 (strong periodic window)
- Noise: x_noisy = x + σ * N(0,1)

## Results

| sigma | std_Dk | collapse |
|-------|--------|----------|
| 0.000 | 0.0000 | **TRUE** |
| 0.001 | 0.0924 | FALSE |
| 0.010 | 0.0918 | FALSE |
| 0.050 | 0.0929 | FALSE |
| 0.100 | 0.0905 | FALSE |
| 0.200 | 0.0901 | FALSE |

---

## Decision

**Collapse breaks at sigma = 0.001**

D(k) collapse is **FRAGILE** - detects only exact mathematical periodicity.

---

## Strategic Implication

Case A confirmed: The detector is fragile
- Breaks with even tiny noise (σ = 0.1% of signal std)
- Does NOT detect approximate repetition

This means:
- D(k) collapse is a **pure mathematical detector**
- NOT suitable for noisy real-world data without formalization

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**