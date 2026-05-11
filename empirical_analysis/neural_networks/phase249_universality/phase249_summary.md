# Phase 249: Dynamical Universality & Attractor Class Transfer Audit

**Verdict:** SYNCHRONIZATION_SPECIFIC_GENERATION
**Confidence:** MODERATE
**Date:** 2026-05-11 12:13:58

---

## Aggregate Evidence

| Metric | Value |
|--------|-------|
| Mean DNI (all) | 0.1278 |
| Mean ARP | 0.6000 |
| Systems above threshold | 2/10 |

### DNI by Attractor Class

| Class | Mean DNI |
|-------|----------|
| chaotic | 0.1433 |
| fixed_point | -0.1633 |
| metastable | -0.0731 |
| oscillatory | 1.2103 |
| self_organized_critical | -0.3492 |
| stochastic | 0.0402 |

---

## Per-System Results

| System | Class | DNI | ARP | A sim | B sim | D beh |
|--------|-------|-----|-----|-------|-------|-------|
| EEG | metastable | -0.0731 | 0.00 | 0.0035 | 0.0766 | chaotic_recurrence |
| Kuramoto | oscillatory | 1.2103 | 1.00 | 0.9965 | -0.2138 | chaotic_recurrence |
| Logistic | chaotic | -0.0891 | 1.00 | 0.1227 | 0.2117 | chaotic_recurrence |
| Lorenz | chaotic | 0.4410 | 0.00 | 0.5465 | 0.1055 | slow_convergence |
| Rossler | chaotic | 0.0000 | 0.00 | 0.0000 | 0.0000 | stable |
| CellularAutomata | self_organized_critical | -0.3492 | 0.00 | -0.1333 | 0.2158 | fixed_point_convergence |
| Hopfield | fixed_point | -0.1633 | 1.00 | 0.0000 | 0.1633 | stable |
| RNN | chaotic | 0.2214 | 1.00 | 0.0591 | -0.1622 | chaotic_recurrence |
| Noise | stochastic | 0.1674 | 1.00 | 0.2428 | 0.0754 | chaotic_recurrence |
| PhaseSurrogate | stochastic | -0.0870 | 1.00 | -0.1447 | -0.0577 | chaotic_recurrence |

---

## Interpretation

1. Which attractor classes regenerate organization? — See DNI table above
2. Is synchronization necessary? — Oscillatory DNI=1.210
3. Is metastability necessary? — Metastable DNI=-0.073
4. Can chaos regenerate structure? — Chaotic DNI=0.143
5. Do fixed-point systems recover? — Fixed-point DNI in table
6. EEG similarity — Compare EEG row to others
7. Universal recovery law — Verdict determines

COMPLIANCE: LEP
