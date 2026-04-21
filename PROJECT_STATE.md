# SFH-SGP PROJECT STATE (APRIL 2026)

## CURRENT STATUS

We have completed initial empirical validation of multiscale dimensionality (D(k)) using fMRI datasets.

### Key Findings:

1. Dimensionality (D(k)) contains REAL but WEAK task-discriminative information.
2. Strong differences appear for:
   - Task vs Rest
   - GO vs STOP
3. Weak or no differences for:
   - Load gradients
   - Outcome (success vs failure)
   - Task switching vs repetition

### CRITICAL DISCOVERY:

Dimensionality does NOT encode cognitive content directly.

Dimensionality encodes:

→ GLOBAL ORGANIZATION of system activity
→ INTERACTION-SPACE STRUCTURE

NOT:
→ specific thoughts
→ fine-grained task differences

### VALIDATED PRINCIPLE:

Dimensionality is:

→ COARSE-GRAINED
→ REGIME-SENSITIVE
→ STABLE across subjects
→ WEAKLY discriminative at fine scales

### FAILED METRICS (IMPORTANT):

The following metrics are NOT valid indicators of task specificity:

- Within-task correlation (artifact)
- Raw amplitude boundary values
- Uncontrolled AUC comparisons without validation

### VALIDATED METHODS:

- k-NN dimensionality at k = [2,4,8,16]
- Cross-validation classification
- Shuffle controls (MANDATORY)

------------------------------------------------------------

## NEW STRATEGY

We are transitioning from:

→ Noisy neuroscience validation

TO:

→ CLEAN cross-domain validation

Primary target domains:

1. Neural networks (PRIMARY)
2. Dynamical systems (SECONDARY)
3. Physics simulations (OPTIONAL)
4. Ecology/network systems (OPTIONAL)

------------------------------------------------------------

## HARD RULES FOR ALL FUTURE ANALYSIS

1. NO synthetic fallback data EVER
2. ALL results must pass shuffle validation
3. ALL classification must include:
   - real vs shuffled comparison
   - confusion matrix
4. FAIL HARD on:
   - NaN values
   - reused outputs
   - missing data
5. ALWAYS print sample feature vectors
6. NEVER reuse prior results files

------------------------------------------------------------

## CURRENT PRIORITY

NEXT EXPERIMENT:

→ Neural Network Dimensionality Study

Goal:

Test whether D(k) encodes task/regime information in a CLEAN, noise-free system.

------------------------------------------------------------

END OF DOCUMENT
