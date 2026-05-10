# MANUSCRIPT EVIDENCE ARCHITECTURE - PHASE 198

## 1. TITLE OPTIONS

**Option A (Conservative):**
"Reduced-order preservation failure in EEG synchrony structure: An intervention-based falsification study"

**Option B (More explanatory):**
"Multi-factor alignment requirement for EEG eigenvalue organization: Falsification of low-order synchrony models"

**Option C (Minimal):**
"Combinatorial falsification of reduced-order EEG synchrony structure"

**Recommended: Option A** - Focuses on the method (falsification) rather than the claim.

---

## 2. ABSTRACT STRUCTURE

**Background**: Current theories propose various single or multi-factor explanations for EEG synchrony organization.

**Objective**: Test whether any reduced-order feature combination can preserve synchrony structure.

**Methods**: Intervene to preserve specific feature combinations (single, pairwise, triple, quadruple) while destroying others. Measure eigenvalue/efficiency across 10 observables.

**Results**: 0 of 30+ reduced-order models survived 15% survival threshold. Core metrics (O1-O8) show universal collapse; propagation asymmetry and graph entropy preserved. Full five-factor model achieves R²=1.0.

**Conclusions**: Multi-factor alignment required; no reduced-order sufficiency found. Partial generalization to 8/10 metrics.

---

## 3. INTRODUCTION CLAIMS

### Tier 1 Claims (Core):
1. Multiple alignment mechanisms exist in EEG data
2. Reducing to single features fails (kurtosis, common-mode, etc.)
3. Multi-factor model is required (R²=1.0)

### Tier 2 Claims (Bounded):
4. Five specific features required
5. Structure generalizes to 8/10 metrics
6. Internal burst timing not critical

### Tier 3 Claims (Exploratory):
7. Burst coincidence most destructive (96%)
8. Coalition persistence least destructive (78%)

---

## 4. METHODS ORGANIZATION

### 4.1 Dataset
- CHB-MIT (4 subjects from Phase 112)
- Preprocessing: FFT phase, Hilbert transform verification

### 4.2 Controls
- Phase randomization (A)
- Temporal shift (B)
- Burst shuffle (C)
- Channel permutation (D)
- Colored noise (E)

### 4.3 Interventions (Phase 191-194)
- Single-feature preservation (F1-F5)
- Pairwise (M1-M10)
- Triple (M11-M20)
- Quadruple (Q1-Q5)

### 4.4 Metrics (Phase 196)
- O1: eigenvalue
- O2: spectral gap
- O3: efficiency
- O4: sync variance
- O5: coalition
- O6: coincidence
- O7: PLV
- O8: zero-lag
- O9: propagation
- O10: entropy

---

## 5. RESULTS NARRATIVE

### 5.1 Control Elimination (Phases 186-190)
- Kurtosis: not causal
- Burst amplitude: partial
- Internal timing: not critical
- Common-mode: not causal

### 5.2 Combinatorial Falsification (Phases 191-194)
- 5 single-feature models failed (>85%)
- 10 pairwise models failed (>89%)
- 10 triple models failed (>94%)
- 5 quadruple models failed (>78%)

### 5.3 Cross-Observable Validation (Phase 196)
- 8/10 metrics collapse (0 survivors)
- 2/10 preserved (O9, O10)

---

## 6. NEGATIVE RESULTS SECTION

**Critical for manuscript integrity:**

1. Kurtosis correlation NOT causal
2. Common-mode NOT causal
3. Zero-lag alone NOT sufficient
4. Propagation alone NOT sufficient
5. Any single factor NOT sufficient
6. Any pairwise NOT sufficient
7. Any triple NOT sufficient
8. Universal irreducibility NOT supported (2/10 metrics preserved)

---

## 7. GOVERNANCE & VALIDATION SECTION

### Phase 180: Procedural Failure
- "Fast version" parameter drift
- Classified as FAILED (Phase 181)
- LEP established (Phase 182)

### Phase 191-192: Incomplete Search
- Specified 20 models, executed 6
- Audit identified 30% completeness
- Full search re-executed (Phase 193)

### Phase 194-195: Verdict Correction
- "MIXED" language was inconsistent
- Corrected to "IRREDUCIBLE_FIVE_FACTOR"

### Phase 196: Partial Generalization
- Documented O9, O10 preservation
- Adjusted verdict to "PARTIAL"

---

## 8. LIMITATIONS SECTION

1. Dataset: CHB-MIT only (4 subjects)
2. Preprocessing: FFT-based, 10% threshold
3. Threshold: 15% a priori, not theoretically grounded
4. Temporal: 512-sample windows
5. Causal mechanism: Unknown (correlation, not causation)

---

## 9. DISCUSSION BOUNDARIES

### CAN CLAIM:
- Multi-factor alignment required for eigenvalue/efficiency metrics
- No reduced-order sufficiency demonstrated within framework
- 8/10 metrics show collapse; 2 preserved

### CANNOT CLAIM:
- Consciousness / cognition / sentience
- Universal neural law
- Biological uniqueness
- Mechanism (only correlation, not causation)
- Generalizability beyond CHB-MIT

---

## 10. SUPPLEMENTARY MATERIAL PLAN

### Required Supplements:
1. **All 30+ model results** - detailed tables
2. **Governance corrections** - Phase 192, 195 reports
3. **Negative results registry** - failed hypotheses
4. **Phase 180 audit** - procedural failure documentation
5. **Full observable metrics** - O1-O10 for all models

### Not for publication:
- Failed "fast version" details from Phase 180
- Internal debates about verdict language
- Exploratory analyses not yet replicated