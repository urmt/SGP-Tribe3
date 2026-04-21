# SFH-SGP_OBSERVABLE_INVARIANCE_01 (LOCKED)

## Research Question
Is (α, DET) invariant under different representations of the same system?

## System
- Kuramoto model at K ∈ [0.5, 2.5]
- Tested representations: mean projection, single oscillator, full state

## Results

| Representation | α | DET | Status |
|---------------|-----|-----|--------|
| Mean projection | ~0.89-1.00 | ~0.85→0.37 | Valid |
| Single oscillator | ~1.00 | ~0.11→0.02 | Valid |
| Full state | NaN | 0.00 | Invalid (high-dim) |

## Invariance Analysis

| Comparison | Δα |
|------------|-----|
| Mean vs Single | **0.041** |
| Mean vs Full | NaN |

---

## Decision

**Observable is REPRESENTATION-DEPENDENT**

α differs slightly across scalar projections (Δ = 0.04)
DET differs significantly (mean: high, single: low)

---

## Implications

1. (α, DET) measures ** observed dynamics**, not intrinsic system state
2. Different projections yield different DET values
3. This is a measurement-dependent observable, not a system invariant

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**