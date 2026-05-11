# Phase 244: Organizational Topological Skeleton Audit

**Verdict:** WEAK_RECOVERY_SCAFFOLD
**Confidence:** LOW
**Date:** 2026-05-11 11:22:35

---

## Executive Summary

Tests whether a minimal topological skeleton survives TRUE destructive intervention across 3 systems (CHB-MIT EEG, Kuramoto, Logistic).

Skeleton score = mean of 7 metrics: component overlap, cycle overlap,
backbone preservation, edge survival, MST similarity, centroid retention,
peak conservation.

---

## Aggregate Evidence

| Metric | Value |
|--------|-------|
| Mean skeleton score (overall) | 0.3689 |
| Mean skeleton score (recovery) | 0.4520 |
| Mean effect size vs random | 0.7129 |

### Per-System Skeleton Scores

- **EEG**: 0.5247
- **Kuramoto**: 0.2709
- **Logistic**: 0.3111

---

## Per-System Details

### EEG

**Skeleton Scores:**

| Comparison | Score |
|------------|-------|
| Pre vs Destroyed | 0.4142 |
| Pre vs Recovery | 0.6924 |
| Destroyed vs Recovery | 0.4673 |
| Overall | 0.5247 |

**Pre-Collapse Topology:**

| Property | Value |
|----------|-------|
| components | 4 |
| edges | 2 |
| mst_edges | 7 |
| cycles | 0 |
| peaks | 15 |

**Per-Metric Breakdown (Recovery):**

| Metric | Value |
|--------|-------|
| component_overlap | 0.7778 |
| cycle_overlap | 1.0000 |
| backbone_preservation | 0.9859 |
| persistent_edge_survival | 0.2500 |
| mst_similarity | 0.7500 |
| centroid_retention | 0.9680 |
| peak_conservation | 0.1154 |

**Control Scores:**

| Control | Score |
|---------|-------|
| random_topology | 0.3831 |
| edge_shuffled | 0.4766 |
| phase_randomized | 0.4170 |
| synthetic_gaussian | 0.1992 |
| random_backbone | 0.3801 |

---

### Kuramoto

**Skeleton Scores:**

| Comparison | Score |
|------------|-------|
| Pre vs Destroyed | 0.2048 |
| Pre vs Recovery | 0.3608 |
| Destroyed vs Recovery | 0.2471 |
| Overall | 0.2709 |

**Pre-Collapse Topology:**

| Property | Value |
|----------|-------|
| components | 4 |
| edges | 3 |
| mst_edges | 7 |
| cycles | 0 |
| peaks | 9 |

**Per-Metric Breakdown (Recovery):**

| Metric | Value |
|--------|-------|
| component_overlap | 0.4000 |
| cycle_overlap | 0.0000 |
| backbone_preservation | 0.9992 |
| persistent_edge_survival | 0.0000 |
| mst_similarity | 0.0769 |
| centroid_retention | 0.9942 |
| peak_conservation | 0.0556 |

**Control Scores:**

| Control | Score |
|---------|-------|
| random_topology | 0.3569 |
| edge_shuffled | 0.3784 |
| phase_randomized | 0.3527 |
| synthetic_gaussian | 0.2633 |
| random_backbone | 0.2633 |

---

### Logistic

**Skeleton Scores:**

| Comparison | Score |
|------------|-------|
| Pre vs Destroyed | 0.3286 |
| Pre vs Recovery | 0.3026 |
| Destroyed vs Recovery | 0.3021 |
| Overall | 0.3111 |

**Pre-Collapse Topology:**

| Property | Value |
|----------|-------|
| components | 4 |
| edges | 2 |
| mst_edges | 7 |
| cycles | 0 |
| peaks | 56 |

**Per-Metric Breakdown (Recovery):**

| Metric | Value |
|--------|-------|
| component_overlap | 0.4000 |
| cycle_overlap | 1.0000 |
| backbone_preservation | -0.4089 |
| persistent_edge_survival | 0.0000 |
| mst_similarity | 0.2727 |
| centroid_retention | 0.6876 |
| peak_conservation | 0.1667 |

**Control Scores:**

| Control | Score |
|---------|-------|
| random_topology | 0.3072 |
| edge_shuffled | 0.1958 |
| phase_randomized | 0.2822 |
| synthetic_gaussian | 0.1927 |
| random_backbone | 0.1927 |

---

