# Phase 246: Causal Generator Dimension Ablation Audit

**Verdict:** NONCAUSAL_COMPRESSED_SUMMARY
**Confidence:** HIGH
**Date:** 2026-05-11 11:48:52

---

## Core Question

Does removing the recovered low-rank spectral backbone destroy recovery itself?

---

## Aggregate Evidence

| Metric | Value |
|--------|-------|
| Mean causal impact | 0.0079 |
| Mean λ1 share of max impairment | 0.8925 |
| Mean dominant dim dependence | 1.1403 |

### Per-System

| System | Causal Impact | λ1 Share | λ1 Fraction |
|--------|--------------|----------|-------------|
| EEG | 0.0000 | 0.8354 | 0.356 |
| Kuramoto | 0.0236 | 0.8422 | 0.943 |
| Logistic | 0.0000 | 1.0000 | 0.132 |

---

### EEG

Eigenvalues: ['2.849', '1.883', '1.145', '0.668', '0.605']

| Ablation | Real Impairment | Control Mean | Causal Impact |
|----------|----------------|-------------|---------------|
| λ1_only | 0.4083 | 0.4916 | 0.0000 |
| λ1_λ2 | 0.4688 | 0.6326 | 0.0000 |
| λ1_λ2_λ3 | 0.4888 | 0.7567 | 0.0000 |

---

### Kuramoto

Eigenvalues: ['7.546', '0.373', '0.049', '0.015', '0.010']

| Ablation | Real Impairment | Control Mean | Causal Impact |
|----------|----------------|-------------|---------------|
| λ1_only | 0.6188 | 0.5865 | 0.0323 |
| λ1_λ2 | 0.6584 | 0.6430 | 0.0154 |
| λ1_λ2_λ3 | 0.7347 | 0.7253 | 0.0094 |

---

### Logistic

Eigenvalues: ['1.056', '1.032', '1.014', '1.007', '0.990']

| Ablation | Real Impairment | Control Mean | Causal Impact |
|----------|----------------|-------------|---------------|
| λ1_only | 0.3145 | 0.4418 | 0.0000 |
| λ1_λ2 | 0.2915 | 0.5838 | 0.0000 |
| λ1_λ2_λ3 | 0.2946 | 0.6976 | 0.0000 |

---

