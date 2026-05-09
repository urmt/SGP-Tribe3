# COMMIT COMPLIANCE RULES
## Phase 182 Governance

---

## MANDATORY FORMAT

All commits for Phases 183+ MUST follow this format:

```
Phase: XXX
Protocol-Hash: [version from registry]
Controls: [completion matrix]
Runtime: [COMPLETED|FAILED|PROVISIONAL|SOFT_FAIL]
Tier: [T0-T5]
Evidence: [label]
```

---

## EXAMPLE COMPLIANT COMMITS

### Compliant Example 1:
```
Phase: 183
Protocol-Hash: v1.0.0-183
Controls: A✓ B✓ C✓ D✓ E✓ F-
Runtime: COMPLETED
Tier: T2
Evidence: REPLICATED
```

### Compliant Example 2:
```
Phase: 184
Protocol-Hash: v1.0.0-184
Controls: A✓ B- C✓ D✓ E✓ F✓
Runtime: SOFT_FAIL (timeout, retry success)
Tier: T2
Evidence: CONTROL_DEPENDENT
```

### Compliant Example 3:
```
Phase: 185
Protocol-Hash: v1.0.0-185
Controls: A✓ B✓ C✓ D✓ E✓ F✓
Runtime: COMPLETED
Tier: T3
Evidence: SURROGATE_RESISTANT
```

---

## NON-COMPLIANT EXAMPLES

### Missing Phase Number:
```
Fixed the bug in burst detection
```
**Violation:** No Phase number

### Missing Controls:
```
Phase 183 complete
```
**Violation:** No control matrix

### Missing Evidence:
```
Phase 183 - robust finding confirmed
```
**Violation:** No evidence label (ROBUST prohibited anyway)

### Prohibited Label:
```
Phase 183 - STRONG result
```
**Violation:** STRONG is prohibited

---

## AUTOMATIC REJECTION

Commits missing ANY of the mandatory fields are REJECTED.

Enforcement: Git hooks will validate format.

---

## LEGACY PHASES

Phases before 183 are EXEMPT from these rules for historical reasons:
- 177-181 can remain as-is
- But cannot be cited as robust (current audit shows issues)

---

*End of Commit Compliance Rules*