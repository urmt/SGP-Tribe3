# Phase 248: Dynamical Attractor Necessity Audit

**Verdict:** DYNAMICAL_ATTRACTOR_GENERATION
**Confidence:** LOW
**Date:** 2026-05-11 12:02:58

---

## Core Question

If neither spectral generators (246) nor relational constraints (247) are causally necessary, is the TRUE causal substrate the dynamical update rule?

---

## Aggregate Evidence

| Metric | Value |
|--------|-------|
| Mean dynamical necessity index | 0.2478 |
| Mean attractor return probability | 0.6667 |
| Mean recovery regeneration score | 0.2322 |
| A vs B wins | 2/3 |

### Per-System

| System | DNI | ARP | A sim | B sim | D behavior |
|--------|-----|-----|-------|-------|-----------|
| EEG | -0.1511 | 0.00 | -0.4120 | -0.2609 | chaotic_recurrence |
| Kuramoto | 0.8376 | 1.00 | 0.9926 | 0.1550 | chaotic_recurrence |
| Logistic | 0.0570 | 1.00 | 0.1160 | 0.0590 | chaotic_recurrence |

---

### EEG

| Condition | Org | Similarity | Half-life | Behavior |
|-----------|-----|-----------|-----------|----------|
| Dyn preserved | 0.8949 | -0.4120 | 1.7 | chaotic_recurrence |
| Struct preserved | 0.2925 | -0.2609 | 3.4 | chaotic_recurrence |
| Both destroyed | 0.2596 | -0.4172 | 0.0 | N/A |
| True recovery | 1.8483 | 1.0000 | 47.6 | chaotic_recurrence |

---

### Kuramoto

| Condition | Org | Similarity | Half-life | Behavior |
|-----------|-----|-----------|-----------|----------|
| Dyn preserved | 6.6479 | 0.9926 | 10.3 | stochastic_drift |
| Struct preserved | 0.3003 | 0.1550 | 693.1 | chaotic_recurrence |
| Both destroyed | 0.3094 | 0.4922 | 0.0 | N/A |
| True recovery | 6.6713 | 1.0000 | 0.5 | chaotic_recurrence |

---

### Logistic

| Condition | Org | Similarity | Half-life | Behavior |
|-----------|-----|-----------|-----------|----------|
| Dyn preserved | 0.4035 | 0.1160 | 5.9 | chaotic_recurrence |
| Struct preserved | 0.3065 | 0.0590 | 14.0 | chaotic_recurrence |
| Both destroyed | 0.3036 | 0.2094 | 0.0 | N/A |
| True recovery | 0.4112 | 1.0000 | 14.7 | chaotic_recurrence |

---

