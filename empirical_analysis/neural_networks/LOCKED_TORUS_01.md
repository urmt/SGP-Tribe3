# SFH-SGP_TORUS_01 (LOCKED)

## Research Question
Does D(k) collapse occur in quasi-periodic torus attractor?

## System
- **Torus signal**: x(t) = sin(t) + sin(√2 * t)
- Incommensurate frequencies → never repeats
- Lies on 2D torus manifold

## Results

| k | D(k) |
|----|------|
| 2 | -3.7762 |
| 4 | -3.5057 |
| 8 | -3.1996 |
| 16 | -2.8670 |

| Metric | Value |
|--------|-------|
| std(Dk) | 0.3396 |
| Collapse | **FALSE** |

---

## Interpretation

**D(k) collapse = FALSE**

D(k) does NOT collapse in quasi-periodic torus attractor, indicating:
- D(k) specifically detects **discrete orbit closure** (periodicity)
- D(k) is NOT sensitive to low-dimensional manifolds

---

## Cross-System Summary

| System | Type | D(k) Collapse |
|--------|------|-------------|
| Logistic (periodic) | Discrete repeat | **YES** |
| Logistic (chaotic) | No repeat | NO |
| Hénon | Chaotic | NO |
| Kuramoto | Continuous sync | NO |
| OU | Stochastic | NO |
| Torus | Quasi-periodic (2D manifold) | NO |

---

## Decision

**D(k) collapse specifically detects discrete periodic orbits**

The metric is NOT a:
- Chaos detector
- Complexity metric
- Manifold detector

It IS a:
- Periodic orbit detector

---

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**