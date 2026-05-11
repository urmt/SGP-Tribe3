# Phase 247: Relational Constraint Invariance Audit

**Verdict:** NONRELATIONAL_RECOVERY
**Confidence:** LOW
**Date:** 2026-05-11 11:53:48

---

## Core Question

Can recovery survive arbitrary basis transformations as long as relational constraints are preserved?

---

## Aggregate Evidence

| Metric | Value |
|--------|-------|
| Mean basis invariance | 0.6538 |
| Mean relational causal impact | 0.0541 |

### Per-System

| System | Basis Invariance | Relational Impact |
|--------|-----------------|-------------------|
| EEG | 0.7456 | 0.0276 |
| Kuramoto | 0.6579 | -0.0088 |
| Logistic | 0.5577 | 0.1435 |

---

### EEG

**Basis-Preserving Transforms:**

| Transform | Similarity |
|-----------|-------------|
| pca_basis_identity | 1.0000 |
| nonlinear_warp | 0.9935 |
| eigenbasis_permutation | 0.7648 |
| invertible_linear | 0.7138 |
| coordinate_scramble | 0.6554 |
| orthogonal_rotation | 0.3463 |

**Relation-Destroying Transforms:**

| Control | Similarity | Impairment |
|---------|-------------|------------|
| recurrence_destruction | 0.1511 | 0.8489 |
| coordinate_destruction | 0.1569 | 0.8431 |
| triadic_destruction | 1.0000 | 0.0000 |
| mst_scrambling | 1.0000 | 0.0000 |
| shortest_path_randomization | 1.0000 | 0.0000 |
| random_graph_rewiring | 1.0000 | 0.0000 |

Basis invariance: 0.7456
Relational causal impact: 0.0276

---

### Kuramoto

**Basis-Preserving Transforms:**

| Transform | Similarity |
|-----------|-------------|
| pca_basis_identity | 1.0000 |
| nonlinear_warp | 0.9997 |
| invertible_linear | 0.7764 |
| coordinate_scramble | 0.7707 |
| eigenbasis_permutation | 0.2502 |
| orthogonal_rotation | 0.1507 |

**Relation-Destroying Transforms:**

| Control | Similarity | Impairment |
|---------|-------------|------------|
| coordinate_destruction | -0.0110 | 1.0110 |
| recurrence_destruction | 0.0116 | 0.9884 |
| triadic_destruction | 1.0000 | 0.0000 |
| mst_scrambling | 1.0000 | 0.0000 |
| shortest_path_randomization | 1.0000 | 0.0000 |
| random_graph_rewiring | 1.0000 | 0.0000 |

Basis invariance: 0.6579
Relational causal impact: -0.0088

---

### Logistic

**Basis-Preserving Transforms:**

| Transform | Similarity |
|-----------|-------------|
| pca_basis_identity | 1.0000 |
| nonlinear_warp | 0.9367 |
| eigenbasis_permutation | 0.4465 |
| coordinate_scramble | 0.3467 |
| orthogonal_rotation | 0.3164 |
| invertible_linear | 0.3001 |

**Relation-Destroying Transforms:**

| Control | Similarity | Impairment |
|---------|-------------|------------|
| mst_scrambling | 0.3017 | 0.6983 |
| recurrence_destruction | 0.3533 | 0.6467 |
| coordinate_destruction | 0.3816 | 0.6184 |
| random_graph_rewiring | 0.4385 | 0.5615 |
| triadic_destruction | 0.4655 | 0.5345 |
| shortest_path_randomization | 0.5446 | 0.4554 |

Basis invariance: 0.5577
Relational causal impact: 0.1435

---

