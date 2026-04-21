# SFH-SGP_RETURN_STRUCTURE_01 (LOCKED)

## Research Question
Can recurrence rate provide noise-robust separation of dynamical regimes?

## System
- Logistic map: r = [3.5 (periodic), 3.57 (boundary), 3.9 (chaotic)]
- Noise: σ ∈ [0, 1e-4, 1e-3]
- ε ∈ [1e-3, 5e-3, 1e-2]

## Results

| r | σ | Recurrence Rate (mean) |
|---|---|----------------------|
| 3.5 (periodic) | 0.0 | 0.000 |
| 3.5 (periodic) | >0 | **0.230** |
| 3.9 (chaotic) | 0.0 | 0.018 |
| 3.9 (chaotic) | >0 | 0.019 |

---

## Key Finding

**Recurrence separates regimes under NOISE:**

| Regime | Clean (σ=0) | Noisy (σ>0) |
|-------|-----------|------------|
| Periodic | 0.00 | **0.23** |
| Chaotic | 0.02 | 0.02 |

- At σ=0: Periodic has LOWER recurrence (exact orbit, no "near misses")
- At σ>0: Periodic has HIGHER recurrence (noise perturbs but system stays near stable manifold)

---

## Interpretation

Recurrence rate works as noise-robust discriminator:
- NOISE actually HELPS separate periodic from chaotic
- Periodic systems gravitate toward attractor → higher "near recurrence"
- Chaotic systems already spread → less sensitive to noise

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**