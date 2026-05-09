# EVIDENCE CLASSIFICATION
## Allowed Labels and Definitions

---

## APPROVED LABELS

### 1. OBSERVED

**Definition:** Single-phase finding without replication or controls

**Usage:**
- Tier 1 evidence
- First detection of phenomenon
- Requires follow-up

**Requirements:** T1+

**Example:**
```
verdict: "OBSERVED"
# Only phase random control tested
```

---

### 2. REPLICATED

**Definition:** Finding confirmed with multiple controls, survival pattern clear

**Usage:**
- Tier 2 evidence
- Stronger than OBSERVED

**Requirements:** T2+, 3+ controls

**Example:**
```
verdict: "REPLICATED"
# 4/5 controls survive destruction
```

---

### 3. SURROGATE_RESISTANT

**Definition:** All or nearly all controls fail to destroy finding

**Usage:**
- Tier 3 evidence
- Highest for robustness claims
- Requires: 5+ controls, survival

**Requirements:** T3+

**Example:**
```
verdict: "SURROGATE_RESISTANT"
# 6/6 controls tested, 5+ survive
```

---

### 4. CONTROL_DEPENDENT

**Definition:** Mixed result - some controls destroy, some don't

**Usage:**
- Tier 2 evidence
- Requires interpretation

**Requirements:** T2+, mixed pattern

**Example:**
```
verdict: "CONTROL_DEPENDENT"
# Phase random DESTROYS, burst timing SURVIVES
```

---

### 5. SURROGATE_EXPLAINED

**Definition:** All controls explain/simplify/destroy finding

**Usage:**
- Tier 1 evidence
- Finding is linear artifact

**Requirements:** T1+, all fail

**Example:**
```
verdict: "SURROGATE_EXPLAINED"
# 4/4 controls explain the phenomenon
```

---

### 6. PROVISIONAL

**Definition:** Incomplete, needs verification

**Usage:**
- Tier 0 evidence
- Early stage

**Requirements:** Any incomplete

**Example:**
```
verdict: "PROVISIONAL"
# Only 2/5 controls completed
```

---

### 7. INVALIDATED

**Definition:** Failed verification or replication

**Usage:**
- Terminal state
- No further claims

**Requirements:** Verification failure

**Example:**
verdict: "INVALIDATED"
# Parameter drift detected
```

---

## PROHIBITED LABELS

The following are NOT allowed:

| Prohibited | Use Instead |
|------------|--------------|
| ROBUST | SURROGATE_RESISTANT |
| STRONG | OBSERVED/REPLICATED |
| SIGNIFICANT | Statistical p-value only |
| NOVEL | OBSERVED |
| BREAKTHROUGH | PROVISIONAL |
| PROOF | VERIFIED (T5 only) |

---

## LABEL ENFORCEMENT

**Rule:** Only labels from APPROVED list may appear in:
- Phase verdict fields
- Summary documentation
- Commit messages
- Publication text

**Penalty:** Invalid labels invalidate the phase.

---

*End of Evidence Classification*