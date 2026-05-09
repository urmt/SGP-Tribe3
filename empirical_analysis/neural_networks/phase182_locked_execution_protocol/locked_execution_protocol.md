# LOCKED EXECUTION PROTOCOL (LEP)
## Phase 182 - Governance Framework for All Future Phases

**Effective Date**: Phase 182 completion  
**Status**: MANDATORY for Phases 183+  
**Authority**: Research Director Override  

---

## 1. EXECUTION LOCKS

### 1.1 Absolute Prohibitions

The following are FORBIDDEN under ALL circumstances:

| Lock | Description | Penalty |
|------|-------------|---------|
| L1 | "Fast versions" or simplified pipelines | AUTOMATIC_INVALIDATION |
| L2 | Fallback analyses without explicit declaration | AUTOMATIC_INVALIDATION |
| L3 | Runtime simplifications | AUTOMATIC_INVALIDATION |
| L4 | Control substitutions without documented rationale | INVALIDATE_ROBUSTNESS |
| L5 | Silent threshold changes | INVALIDATE_RESULTS |
| L6 | Altered window sizes | AUTOMATIC_INVALIDATION |
| L7 | Altered sampling rates | AUTOMATIC_INVALIDATION |
| L8 | Altered subject pools | INVALIDATE_CLAIMS |
| L9 | Altered LOSO behavior | AUTOMATIC_INVALIDATION |
| L10 | Post-hoc parameter tuning | INVALIDATE_RESULTS |

### 1.2 Enforcement

Any phase entering execution with ANY of the above automatically enters **FAILURE STATE**.

---

## 2. FAILURE STATES

### 2.1 Hard Fail (CRITICAL)
- Protocol violation with no recovery path
- Result: Phase marked INVALID
- Examples: No audit chain, missing controls, parameter drift

### 2.2 Soft Fail (RECOVERABLE)
- Runtime interruption with resolution
- Result: Phase marked PROVISIONAL, retry allowed
- Examples: Timeout (documented), library missing (install retry)

### 2.3 Partial Execution
- Some controls incomplete
- Result: Phase marked PARTIAL, claims limited
- Examples: 3/5 controls completed

### 2.4 Invalid Replication
- Methodology mismatch with source
- Result: Phase marked REPLICATION_FAILURE
- Examples: Different threshold, different windows

### 2.5 Unverifiable
- Insufficient audit evidence
- Result: Phase marked INVALID
- Examples: No runtime log, no control matrix

---

## 3. CONTROL REQUIREMENTS

### 3.1 Mandatory Control Set

All discovery phases (183+) MUST declare control completion:

| Control | Code | Purpose |
|---------|------|---------|
| Phase Randomization | A | Destroy phase relationships |
| IAAFT Surrogate | B | Iterative amplitude-adjusted Fourier |
| Temporal Block | C | Destroy temporal structure |
| Channel Permutation | D | Test spatial invariance |
| Noise Control | E | White/pink noise baseline |
| Burst Timing | F | Test burst-pattern dependency |

### 3.2 Completion Matrix

Every phase MUST report:

```
Controls Required: [A, B, C, D, E, F]
Controls Completed: [X, X, X, -, X, X]
Omitted: [B (reason), D (reason)]
```

### 3.3 Robustness Claims

Claims of robustness require:
- MINIMUM: 4/6 controls completed (TIER 2+)
- FULL: 6/6 controls completed (TIER 4+)

---

## 4. PARAMETER IMMUTABILITY

### 4.1 Immutable Parameter Registry

ALL phases MUST use the central registry (see `immutable_parameter_registry.json`):

| Parameter | Value | Scope |
|-----------|-------|-------|
| random_state | 42 | ALL |
| window_size | 512 | ALL |
| overlap | 256 | ALL |
| burst_threshold | 90 | Percentile |
| pca_variance_target | 0.80 | Proportion |
| pca_max_components | 20 | Count |
| fs | 256 | Hz |
| filtering_order | 3 | Butterworth |

### 4.2 Drift Detection

Any deviation from registry invalidates results.

---

## 5. RUNTIME REPORTING

### 5.1 Mandatory Logging

Every phase MUST log:

```json
{
  "phase": 183,
  "runtime_seconds": 120.5,
  "cpu_time_seconds": 480.2,
  "memory_mb_estimate": 2048,
  "failed_iterations": 0,
  "convergence_status": "COMPLETED",
  "retry_count": 0,
  "timeout_events": 0
}
```

### 5.2 Failure Logging

Runtime failures MUST be logged with:
- Exact error message
- Stack trace
- Retry attempts
- Final resolution

---

## 6. REPLICATION TIERS

### Tier Definitions

| Tier | Name | Requirements | Evidence Level |
|------|------|--------------|----------------|
| T0 | Exploratory | Single observation | PROVISIONAL |
| T1 | Single-control | 1 control surviving | OBSERVED |
| T2 | Multi-control | 3+ controls surviving | REPLICATED |
| T3 | Surrogate-resistant | 5+ controls surviving | CONTROL_DEPENDENT |
| T4 | Replication-ready | Full control set + LOSO | SURROGATE_RESISTANT |
| T5 | Publication-grade | Independent verification | VERIFIED |

---

## 7. EVIDENCE CLASSIFICATION

### Allowed Labels

| Label | Meaning | Requires |
|-------|---------|----------|
| OBSERVED | Single-phase observation | T0+ |
| REPLICATED | Multiple controls pass | T2+ |
| SURROGATE_RESISTANT | All surrogates fail to destroy | T3+ |
| CONTROL_DEPENDENT | Some controls pass, some fail | T2+ |
| SURROGATE_EXPLAINED | All controls explain result | T1+ |
| PROVISIONAL | Incomplete, needs verification | T0+ |
| INVALIDATED | Failed verification | NONE |

### Prohibited Labels

- "ROBUST" (use SURROGATE_RESISTANT)
- "STRONG" (use OBSERVED or REPLICATED)
- "SIGNIFICANT" (use statistical p-values only)

---

## 8. COMMIT COMPLIANCE RULES

### Required Commit Fields

Every commit message MUST include:

```
Phase: XXX
Protocol-Hash: [registry version]
Controls: [A,B,C,D,E,F completion]
Runtime: [COMPLETED/FAILED/PROVISIONAL]
Tier: [T0-T5]
Evidence: [label]
```

### Non-Compliant Commits

Commits without this format are AUTOMATICALLY REJECTED by audit system.

---

## 9. DATA INTEGRITY RULES

### Protection Mechanisms

| Risk | Protection |
|------|------------|
| Overwrite corruption | Versioned output files |
| Duplicate outputs | Unique phase prefixes |
| Silent replacement | SHA256 checksums |
| Partial save states | Atomic writes only |
| Seed inconsistency | random_state in all outputs |

---

## 10. AUDIT CHAIN ENFORCEMENT

### Required Outputs

Every phase MUST generate:

- `audit_manifest.json` - Phase metadata
- `parameter_snapshot.json` - Registry values used
- `runtime_log.txt` - Execution log
- `control_matrix.csv` - Control completion
- `integrity_summary.txt` - Director assessment

### Verification

Phases without complete audit chains are INVALID.

---

## ENFORCEMENT

This protocol is MANDATORY for Phases 183+.

**Director Authority**: Phase 182 establishes binding rules.

**Violation Consequence**: AUTOMATIC_INVALIDATION

---

*End of Locked Execution Protocol*
*Phase 182 Complete*