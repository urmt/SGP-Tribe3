# MANDATORY ANALYSIS RULES

## Data Requirements

- Only real data - NO synthetic fallback
- All results must pass shuffle validation
- No reuse of prior results files
- Fail HARD on NaN values

## Classification Requirements

All classification experiments MUST include:

- Real vs shuffled comparison
- Confusion matrix
- Per-class accuracy
- Sample feature vectors printed

## Pipeline Standards

- k-NN at k = [2,4,8,16]
- StandardScaler for features
- LogisticRegression or SVM classifier
- 5-fold or LOO cross-validation

## FAIL Conditions

- Any NaN → STOP immediately
- Reused results → STOP
- Synthetic fallback → STOP
- Missing shuffle control → INVALID RESULT

## Hard Rules Summary

1. NO synthetic fallback data EVER
2. ALL results must pass shuffle validation
3. ALL classification must include real vs shuffled comparison
4. FAIL HARD on: NaN, reused outputs, missing data
5. ALWAYS print sample feature vectors
6. NEVER reuse prior results files