# SFH-SGP_INVARIANT_ALIGNMENT_01 (LOCKED)

## System
**Logistic Map** - x_{n+1} = r * x_n * (1 - x_n)

## Parameter Range
r = 3.54 → 3.60, step = 0.002 (31 points)

## Key Results

### Boundary Detection

| Metric | Value |
|--------|-------|
| λ zero-crossing | r = **3.569429** |
| D(k) collapse ends | r = **3.568000** |
| **Alignment error** | **0.00143** |

### Decision
**STRONG ALIGNMENT** (error < 0.01)

### Conclusion
Does D(k) collapse align with λ = 0 boundary? **YES**

---

## Interpretation

The dimensional collapse in D(k) occurs in the periodic regime (λ < 0) and D(k) variance reappears exactly at the chaos onset. The transition is sharp and well-defined at r ≈ 3.569.

This confirms that:
- D(k) captures the fundamental dynamical transition from periodic to chaotic
- The collapse boundary marks the exact bifurcation point
- The metric serves as a precise chaos detector

---

## Validation Checks

- [x] λ crosses zero exactly once
- [x] std_Dk increases near transition
- [x] Collapse region is contiguous
- [x] No NaN or constant values

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**
