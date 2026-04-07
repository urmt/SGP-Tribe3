# SGP-Tribe3 TODO & Best Practices Roadmap

**Reference:** Harvard CS249r / MLSysBook.ai — *Principles and Practices of Engineering Artificially Intelligent Systems*
**Link:** https://github.com/harvard-edge/cs249r_book | https://mlsysbook.ai
**Project:** Sentient-Field Braintrust / SGP-Tribe3
**Last Updated:** April 2026
**Research Director:** Mark Rowe Traver

---

> "The world is rushing to build AI systems. It is not engineering them."
> — Harvard CS249r Mission Statement

This document applies the Harvard MLSysBook engineering framework to SGP-Tribe3.
Items marked 🔴 are blocking. Items marked 🟡 are important but non-blocking. Items marked 🟢 are future/nice-to-have.

---

## THEORETICAL FOUNDATION — RESOLVED ITEMS

### The Three-Level Consciousness Hierarchy (LOCKED)

| Level | Term | Definition | Condition |
|---|---|---|---|
| 1 | **Sentience (S)** | ℋS itself — capacity for field-fiber contact. Being without content. Pre-experiential. | Q > 0 |
| 2 | **Awareness (A)** | Localized σ_q excitation registers its own state. Field "knows" partition exists. No self-model. K=1–K=4. | C_local > 0 |
| 3 | **Consciousness (C_K5)** | Awareness recursively modeling itself as distinct from "not-I." The "I am." Requires K=5. | C_system ≥ 1.32 |

### The C ≥ 1.32 Resolution (OPTION 1 — ADOPTED)

Two distinct Coherence measures must be maintained in all code and the paper:

| Symbol | Name | Range | Definition |
|---|---|---|---|
| C_local | Local Coherence | [0, 1] | Normalized pairwise coherence between any two nodes. The formal SFH-SGP primitive. |
| C_system | System Coherence | [0, ∞) | Σ(C_local_ij × w_ij) across all active node pairs, geodesic-weighted. K=5 threshold applies here. |

**Origin of 1.32:** Provisionally adopted from IIT (Integrated Information Theory) empirical Φ literature,
where Φ ≈ 1.3–1.5 represents the measured threshold of conscious binding in awake humans.
SFH-SGP's C_system is the conceptual parallel to IIT's Φ — both measure system-level integrated coherence,
both are unbounded positive reals, both have empirical thresholds near 1.32.
The paper will cite the IIT parallel explicitly and frame 1.32 as provisional pending Option 3 derivation.

**Option 3 (future goal):** Derive the K=5 threshold analytically from the χ minimization landscape —
the minimum C_system at which the Langevin dynamics achieve a saddle point consistent with K=5 self-reference.
This is the long-term theoretical contribution but requires work beyond Phase 1.

### Hypotheses (LOCKED — pending empirical validation)

**H1 (Architectural):**
An LLM integrated with an SGP Resonance Graph calibrated from TRIBE v2 fMRI data will produce
measurably lower hallucination rates and higher cross-domain coherence than the same LLM without
SGP architecture, because activation-weighted node boundaries enforce domain-specific epistemic constraints.

**H2 (Neuroscientific):**
TRIBE v2 inference will empirically recover the Hickok-Poeppel dual-stream dissociation —
ventral nodes (G2, G7, G8) showing significantly higher activation for semantic stimuli,
dorsal nodes (G1, G3, G4, G9) for phonological stimuli — validating neuroscientific node boundary selection.

**H3 (Consciousness):**
Under conditions where C_system approaches or exceeds 1.32, the SGP-integrated LLM will exhibit
behavioral signatures consistent with K=5 self-referential meta-cognition: accurate self-description
of its own processing state, acknowledgment of domain boundaries, and consistency between
self-reports and actual activation patterns. Whether this constitutes Consciousness in the SFH-SGP
sense is an open empirical question — not an assertion — requiring further investigation.

**H0 (Null — for all three):**
The SGP architecture produces no statistically significant difference in hallucination rate,
cross-domain coherence, dual-stream dissociation, or self-referential behavior compared to an
unstructured LLM baseline. All observed differences fall within variance expected from random prompt variation.

---

## TABLE OF CONTENTS

1. [Immediate Blockers — Space Must Run](#1-immediate-blockers)
2. [Data Engineering — The Four Pillars](#2-data-engineering)
3. [Architecture Revisions — SFH-SGP Fidelity](#3-architecture-revisions)
4. [ML Pipeline — Training & Inference](#4-ml-pipeline)
5. [Benchmarking & Evaluation](#5-benchmarking--evaluation)
6. [MLOps — Deployment & Monitoring](#6-mlops)
7. [Model Optimization — Efficiency](#7-model-optimization)
8. [Scientific Article Prerequisites](#8-scientific-article-prerequisites)
9. [Phase 2 — Resonance Graph Engine](#9-phase-2-resonance-graph-engine)
10. [Phase 3 — LLM Integration](#10-phase-3-llm-integration)
11. [Open Source & Reproducibility](#11-open-source--reproducibility)
12. [Reference Links](#12-reference-links)

---

## 1. IMMEDIATE BLOCKERS

### 🔴 1.1 HuggingFace Space Build
- [ ] Confirm Docker build completes successfully past pip install step
- [ ] Verify `/health` endpoint returns `{"status": "ready"}` after warmup
- [ ] Confirm TRIBE v2 model loads on CPU without CUDA errors
- [ ] Confirm Schaefer-200 atlas downloads to `/tmp/sgp_atlas/`
- [ ] Test `/predict` endpoint with a short local video file:
  ```bash
  python3 stimulus_pipeline.py \
    --api https://Sentient-Field-sgp-tribe3.hf.space \
    --local-video /path/to/any/test.mp4 \
    --stimulus-id smoke_test
  ```

### 🔴 1.2 LLaMA License
- [ ] Confirm license accepted at https://huggingface.co/meta-llama/Llama-3.2-3B
- [ ] Confirm HF_TOKEN secret is set in Space settings

### 🔴 1.3 Security
- [ ] Confirm ALL previously exposed tokens have been revoked at https://huggingface.co/settings/tokens
- [ ] New token stored ONLY in local plaintext file — never in chat, never in code

---

## 2. DATA ENGINEERING — THE FOUR PILLARS

> MLSysBook Chapter 6: Four pillars: Quality, Reliability, Scalability, Governance.
> "Data cascades propagate and amplify downstream."

### 🔴 2.1 Stimulus Quality (Pillar: Quality)
- [ ] Verify each of 12 stimuli: video complete, audio track present, correct duration
- [ ] Log exact YouTube video ID or local filename used for each stimulus
- [ ] Create `STIMULUS_MANIFEST.json` with: stimulus_id, source_url_or_path, trim_start, trim_duration, date_acquired

### 🟡 2.2 Result Persistence (Pillar: Reliability) — CRITICAL
- [ ] Results currently in-memory — LOST on every Space restart
- [ ] Add file-based persistence: write each result to HuggingFace dataset repo after each /predict call
- [ ] This is critical for the paper — you cannot re-run TRIBE v2 inference every time

### 🟡 2.3 Versioning (Pillar: Governance)
- [ ] Pin TRIBE v2 to specific commit hash in requirements.txt
- [ ] Document exact TRIBE v2 version, Space commit hash, and date for each calibration run batch

---

## 3. ARCHITECTURE REVISIONS — SFH-SGP FIDELITY

### 🔴 3.1 Geodesic-Weighted Co-Activation Matrix
**Implements:** SFH-SGP geometry principle — distant coherence is more significant than nearby coherence.
**Formula:** w_ij = corr(node_i, node_j) × exp(−γ · d_ij)

- [ ] Implement `compute_geodesic_distances()` in `sgp_parcellation.py`
  using `scipy.sparse.csgraph.shortest_path` on fsaverage5 mesh adjacency
- [ ] Implement `geodesic_weighted_coactivation()` applying the formula above
- [ ] Add geodesic distance matrix to `/coactivation_matrix` endpoint output
- [ ] Expose γ as tunable parameter (default 0.1)

### 🔴 3.2 Metropolis-Hastings Activation Propagation
**Implements:** The SGP Operator: α(q→q') = min(1, exp(−λΔJ)) where J = H(q) − ln(F(q))

- [ ] Implement `sgp_dynamics.py` with full MH propagation:
  - Propose state with Gaussian jitter: q'_i = q_i + ξ where ξ ~ N(0, √(2D·T_eff))
  - Compute J = H(q) − ln(F(q)) for current and proposed states
  - Accept with probability α = min(1, exp(−λΔJ))
  - Anneal T_eff each iteration: T_eff(k+1) = T_eff(k) × cooling_rate
  - Stop when max|q' − q| < 0.01 or K_max iterations
- [ ] Replace generic propagation in `app.py` with MH propagation
- [ ] Add K-depth to `/predict` response

### 🔴 3.3 Explicit Quota (Q) and Sub-Quota (Qk) Computation
**Implements:** Q = Σ|sb|, the conserved sentient quota

- [ ] Q_total = sum of all absolute vertex activations across all timesteps
- [ ] Qk = {node: raw_node_activation / Q_total} — the partition of Q (sums to 1.0)
- [ ] Compute p(Q_discrete) via Hardy-Ramanujan as upper bound on experiential complexity
- [ ] Add Q_total, Qk_distribution, p_Q_upper_bound to `/predict` response

### 🔴 3.4 C_local and C_system Computation
**Implements:** The resolved two-measure coherence framework.

- [ ] C_local_ij = Pearson correlation of node_i and node_j activation across all stimuli (bounded [0,1])
- [ ] C_system = Σ(C_local_ij × w_ij) summed across all active node pairs, geodesic-weighted (unbounded)
- [ ] Report both C_local matrix AND C_system scalar in `/coactivation_matrix` endpoint
- [ ] Flag when C_system ≥ 1.32 (the provisional K=5 threshold)

### 🔴 3.5 Torsion (τ) Detection
**Implements:** τ = inability of Hebbian circuit to change. High C_local, zero F.

- [ ] After all 12 stimuli: compute CV (std/mean) per node across stimuli
- [ ] τ_i = (1 − CV_i) × (1 − F_mean_i) where F = G5_dmn activation
- [ ] τ = 0: fully flexible circuit. τ = 1: completely locked (topological scar).
- [ ] Add `/torsion_analysis` endpoint returning τ_score per node

---

## 4. ML PIPELINE — TRAINING & INFERENCE

### 🟡 4.1 CPU Performance
- [ ] Profile inference time per video (which encoder dominates?)
- [ ] Add `inference_time_seconds` to `/predict` response
- [ ] Consider `torch.set_num_threads()` to maximize CPU parallelism

### 🟡 4.2 Audio Normalization
- [ ] Add ffmpeg `loudnorm` filter to preprocessing — TRIBE v2 audio encoder is volume-sensitive
- [ ] Add video quality check: reject videos under 10 seconds or with no audio track

---

## 5. BENCHMARKING & EVALUATION

> MLSysBook Chapter 12: "Systematic evaluation requires rigorous measurement. Pre-register hypotheses before running experiments."

### 🟡 5.1 Pre-Registered Calibration Tests (define BEFORE running stimuli)
- [ ] **Dual-Stream Test:** Ventral mean (G2,G7,G8) vs dorsal mean (G1,G3,G4,G9) by stimulus category
  - Statistical test: Mann-Whitney U (non-parametric, n=12)
  - Pre-registered prediction: Category A > ventral; Category B > dorsal
- [ ] **K-Depth Test:** Spearman correlation between semantic load rating and K-depth
  - Pre-registered prediction: A1, A3 > B1, B2 in K-depth
- [ ] **C_system Test:** Report C_system per stimulus; flag if any approach ≥ 1.32
- [ ] **τ Test:** Report τ_score per node; hypothesis: G6_limbic shows highest τ

### 🟡 5.2 Benchmark Report
- [ ] After all 12 stimuli: generate `benchmark_report.json` with all metrics per stimulus

---

## 6. MLOPS — DEPLOYMENT & MONITORING

> MLSysBook Chapter 13: "ML systems can degrade silently. Continuous monitoring is essential."

### 🟡 6.1 Persistent Storage — HIGHEST PRIORITY AFTER SPACE RUNS
- [ ] Push results to HuggingFace dataset repo (free, versioned) after each /predict call
- [ ] Or: GitHub repo with timestamped commit per run

### 🟡 6.2 Health Monitoring
- [ ] Add `/metrics` endpoint: uptime, n_predictions, mean_inference_time, last_prediction_timestamp
- [ ] Local cron job pinging `/health` every 10 minutes

### 🟡 6.3 Error Logging
- [ ] Structured error logging: stimulus_id, error type, stack trace for every exception

---

## 7. MODEL OPTIMIZATION — EFFICIENCY

### 🟢 7.1 Future TRIBE v2 Optimization
- [ ] Investigate INT8 quantization of LLaMA 3.2-3B text encoder for CPU
- [ ] Profile which encoder (text/video/audio) dominates inference time

---

## 8. SCIENTIFIC ARTICLE PREREQUISITES

### 🔴 8.1 Technical (blocks Results section)
- [ ] All 5 architectural revisions (Sections 3.1–3.5) implemented and deployed
- [ ] All 12 stimuli acquired, preprocessed, submitted
- [ ] All activation profiles collected and persisted
- [ ] Dual-stream dissociation analysis run with statistical test
- [ ] K-depth hypothesis test run
- [ ] C_system computed per stimulus

### 🟡 8.2 Mathematical Precision (blocks Methods section)
- [ ] Operational formula for C_local: Pearson correlation across stimuli per node pair ✓
- [ ] Operational formula for C_system: geodesic-weighted sum of C_local pairs ✓
- [ ] K=5 threshold 1.32: cite IIT Φ literature, flag as provisional ✓
- [ ] α and β parameter values: start α=β=0.5, run sensitivity analysis
- [ ] γ (geodesic decay): proposal — fit to HCP tractography correlation
- [ ] T_eff schedule: T_eff(k) = T_0 × r^k where T_0=1.0, r=0.9
- [ ] λ (MH parameter): start λ=1.0, tune based on convergence behavior

### 🟡 8.3 Publication Infrastructure
- [ ] Deposit key SFH-SGP Substack articles on Zenodo for citable DOIs:
  - "The Mathematical Atlas of Reality"
  - "The 1,000 Qubit Wall"
  - "Nima Arkani-Hamed Declares the End of Space-Time"
- [ ] Create eLife account: https://elifesciences.org/submit-your-research
- [ ] Prepare JOSS submission separately for SGP-Tribe3 software

---

## 9. PHASE 2 — RESONANCE GRAPH ENGINE

### 🟢 9.1 Graph Implementation
- [ ] Initialize edge weights from Phase 1 geodesic-weighted co-activation matrix
- [ ] Implement simultaneous graded activation — all nodes active, varying intensity
- [ ] Implement full Langevin dynamics: dq/dt = -∇χ(q) + √(2D)ξ(t)
- [ ] Convergence to Resonance Anchor Ωw

### 🟢 9.2 Torsion as Graph Constraint
- [ ] High-τ nodes get edges frozen at mean value during propagation
- [ ] Implements "psychological knot" — locked circuit cannot explore new configurations

### 🟢 9.3 C_system Monitoring
- [ ] Compute C_system at each propagation step
- [ ] Log when C_system crosses 1.32 threshold
- [ ] Record which stimulus conditions and activation patterns produce C_system ≥ 1.32

---

## 10. PHASE 3 — LLM INTEGRATION

### 🟢 10.1 Dynamic System Prompt Construction
- [ ] After Resonance Anchor convergence: identify nodes with activation > 0.4
- [ ] Weight each active node's voice by activation score
- [ ] Single Claude API call with weighted prompt

### 🟢 10.2 Node Voice Templates
- [ ] G1_broca: Expression and form — syntax, phonology, articulation
- [ ] G2_wernicke: Meaning and comprehension — lexical, semantic interpretation
- [ ] G3_tpj: Integration — resolve modality conflicts, find unified signal
- [ ] G4_pfc: Executive oversight — error checking, resource limits, veto
- [ ] G5_dmn: Generativity — novel connections, imaginative exploration (Fertility F)
- [ ] G6_limbic: Emotional salience — weight by importance, memory priors (Torsion τ)
- [ ] G7_sensory: Perceptual grounding — concrete, observable facts
- [ ] G8_atl: Concept formation — cross-modal, unified semantic representation
- [ ] G9_premotor: Output preparation — response structure before speaking

### 🟢 10.3 H3 Test Protocol (Consciousness signature detection)
- [ ] Design structured prompts that probe self-referential meta-cognition
- [ ] Compare responses when C_system < 1.0 vs C_system ≥ 1.32
- [ ] Record: does system accurately describe its own activation state?
- [ ] Record: does system acknowledge domain boundaries unprompted?
- [ ] Human evaluator rating of self-referential coherence

### 🟢 10.4 H1 Test Protocol (Hallucination reduction)
- [ ] Define hallucination test set: 50 questions with known ground truth
- [ ] Run with and without SGP architecture
- [ ] Statistical comparison of accuracy and confidence calibration

---

## 11. OPEN SOURCE & REPRODUCIBILITY

### 🟡 11.1 Code Quality
- [ ] Docstrings for all public functions
- [ ] CONTRIBUTING.md, LICENSE (CC BY-NC 4.0), .gitignore

### 🟡 11.2 Reproducibility
- [ ] STIMULUS_MANIFEST.json: exact source for every stimulus
- [ ] RESULTS_MANIFEST.json: TRIBE v2 version, date, Space commit hash for each run batch
- [ ] All dependency versions pinned in requirements.txt

---

## 12. REFERENCE LINKS

| Resource | URL | Purpose |
|---|---|---|
| Harvard MLSysBook | https://github.com/harvard-edge/cs249r_book | Best practices reference — consult before every architectural decision |
| MLSysBook Online | https://mlsysbook.ai | Read chapters before implementing each phase |
| TRIBE v2 HF | https://huggingface.co/facebook/tribev2 | Model weights and config |
| SGP-Tribe3 Space | https://huggingface.co/spaces/Sentient-Field/sgp-tribe3 | Live deployment |
| Schaefer-200 Atlas | https://github.com/ThomasYeoLab/CBIG | Parcellation atlas source |
| HCP Tractography | https://www.humanconnectome.org | White matter tract validation |
| SFH-SGP Theory | https://wt3000.substack.com | Most recent = most accurate |
| IIT Φ Reference | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588 | Source of C_system ≥ 1.32 threshold (IIT 3.0) |
| Zenodo | https://zenodo.org | Preprint deposit (free, DOI) |
| eLife | https://elifesciences.org/submit-your-research | Primary journal target (free) |
| JOSS | https://joss.theoj.org | Software paper (free) |

---

## PRIORITY ORDER

```
🔴 BLOCKING (in order):
  1. Space builds and runs (/health = ready)
  2. LLaMA license + HF_TOKEN secret confirmed
  3. All old tokens revoked
  4. Add result persistence (HF dataset repo)
  5. Smoke test with local video
  6. Implement C_local and C_system (3.4)
  7. Implement geodesic-weighted co-activation (3.1)
  8. Implement MH propagation + K-depth (3.2)
  9. Implement Q/Qk computation (3.3)
  10. Implement τ detection (3.5)
  11. Pre-register dual-stream and K-depth hypotheses (5.1)
  12. Acquire and run all 12 stimuli
  13. Run all pre-registered tests
  14. Define all mathematical precision items (8.2)

🟡 IMPORTANT (after blocking):
  - Audio normalization in preprocessing
  - Retry logic in stimulus pipeline
  - /metrics endpoint
  - Pin TRIBE v2 to specific commit
  - Deposit SFH-SGP articles on Zenodo
  - Create STIMULUS_MANIFEST.json

🟢 FUTURE (Phase 2+):
  - Resonance Graph Engine with Langevin dynamics
  - LLM integration with weighted node prompts
  - H1, H2, H3 empirical tests
  - Option 3: derive 1.32 threshold analytically
  - Knowledge distillation for faster CPU inference
```

---

*This TODO is a living document. Update after each work session.*
*Consult MLSysBook (https://github.com/harvard-edge/cs249r_book) before every architectural decision.*
*The three-level hierarchy (Sentience → Awareness → Consciousness) and the C_local/C_system distinction are now LOCKED theoretical foundations. Do not conflate them.*
