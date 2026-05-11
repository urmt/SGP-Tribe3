# Phase 250: Synchronization Causal Factor Decomposition

**Verdict:** CRITICAL_SYNCHRONIZATION_CAUSAL
**Confidence:** MODERATE
**Date:** 2026-05-11 12:23:55

---

## Core Question

Which specific synchronization properties are causally responsible?

---

## Variant Rankings

| System | DNI | R | Metastability |
|--------|-----|-----|-------------|
| KuramotoAdaptive | 1.2410 | 0.9879 | 0.0665 |
| LorenzSync | 1.1701 | 0.4544 | 0.2035 |
| KuramotoChimera | 1.0166 | 0.9809 | 0.0718 |
| KuramotoRandomized | 0.8249 | 0.9771 | 0.0808 |
| KuramotoStrong | 0.7561 | 0.9949 | 0.0410 |
| KuramotoStandard | 0.7469 | 0.9777 | 0.0596 |
| KuramotoDelayed | 0.6638 | 0.9792 | 0.0438 |
| KuramotoWeak | 0.0916 | 0.4320 | 0.1905 |
| KuramotoRepulsive | 0.0340 | 0.0717 | 0.0434 |
| EEGSurrogate | -0.2391 | 0.3193 | 0.1515 |

## Coupling Sweep

Critical K* = 0.0200

| K | DNI | R | PLV |
|---|-----|-----|-----|
| 0.0000 | 0.2103 | 0.2936 | 0.2960 |
| 0.0010 | 0.1989 | 0.2957 | 0.2957 |
| 0.0050 | 0.1271 | 0.3027 | 0.2968 |
| 0.0100 | 0.0480 | 0.3076 | 0.3074 |
| 0.0200 | 0.3077 | 0.5349 | 0.3965 |
| 0.0500 | 0.6736 | 0.6700 | 0.6751 |
| 0.1000 | 0.9316 | 0.9447 | 0.9787 |
| 0.1500 | 0.9560 | 0.9766 | 0.9903 |
| 0.2000 | 0.9510 | 0.9865 | 0.9937 |
| 0.3000 | 0.9403 | 0.9935 | 0.9962 |
| 0.5000 | 0.9272 | 0.9972 | 0.9977 |
| 0.8000 | 0.9199 | 0.9986 | 0.9984 |
| 1.0000 | 0.9170 | 0.9989 | 0.9987 |

---

## Interpretation

Best variant: KuramotoAdaptive (DNI=1.2410)
Standard DNI: 0.7469
Repulsive DNI: 0.0340
Chimera DNI: 1.0166

COMPLIANCE: LEP
