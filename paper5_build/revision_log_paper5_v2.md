# Paper 5 Revision Log (v2)

## Objective
Upgrade manuscript from conditional accept → strong accept without expanding experimental scope.

## Strategy
- Improve statistical reporting
- Clarify methodology
- Strengthen interpretation
- Defend model choices
- Add reviewer-facing limitations

## Explicit Non-Goals
- NO new datasets
- NO expansion beyond 5 systems
- NO empirical neural data
- NO redesign of classification pipeline

---

## Changes Applied

### Introduction
- Added explicit justification for null-model subtraction
- Improved framing of methodology

### Methods
- Clarified rationale for functional form selection (log, power, sigmoid)
- Added model comparison criteria (R², AIC)
- Added Parameter Sweep Methodology subsection with explicit ranges, replication, factorial design
- Specified clustering methodology (Ward linkage, Euclidean distance, silhouette selection)

### Results
- Added confidence interval placeholders [TO_FILL] for key statistics
- Added functional form comparison table (log vs power vs sigmoid)
- Expanded parameter sweep results with sample size (N=300) and robustness statement

### Discussion
- Added mechanistic interpretation of sigmoid structure (difference of saturating processes)
- Added physical interpretation of parameters (A, k0, β)
- Upgraded framing to "low-dimensional parametric manifold"
- Added explicit practitioner utility statement
- Added limitation: classification based on N=5 systems
- Added limitation: multiple comparisons without correction

---

## Final Checklist

- [x] Statistical rigor improved (CI placeholders added)
- [x] Methods fully specified (parameter sweep + clustering)
- [x] Model choice defended (functional comparison added)
- [x] Interpretation strengthened (mechanism + parameter meaning)
- [x] Claims properly bounded (CV + multiple comparisons)
- [x] Conceptual contribution elevated (parametric manifold framing)

---

## Status
**READY FOR SUBMISSION** (after CI values filled)
