# SFH-SGP_HENON_VALIDATION_01 (LOCKED)

## Research Question
Does D(k) collapse occur in Hénon map (chaotic system with fractal attractor)?

## System
- **Hénon map**: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
- Parameter sweep: a = 1.2 to 1.4 (40 points, 32 valid after NaN filtering)
- b = 0.3 (standard chaotic parameter)

## Results

| Metric | Value |
|--------|-------|
| Total points | 32 |
| Chaotic points (λ > 0) | **32** |
| Collapse points | **0** |
| Overlap (chaos & collapse) | **0** |
| Collapse rate | 0.000 |
| Chaos rate | 1.000 |
| Overlap rate | 0.000 |
| λ min | 0.2550 |
| λ max | 0.5507 |

---

## Decision

**D(k) collapse is NOT a generic chaos detector**

D(k) collapse does NOT occur in Hénon map even though:
- All 32 points are chaotic (λ > 0)
- The system has a known fractal strange attractor

This falsifies the hypothesis that D(k) collapse = chaos detection.

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**
**DECISIVE EVIDENCE**