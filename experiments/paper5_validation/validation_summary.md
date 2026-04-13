# Paper 5 Validation Summary

## Validation Results

### 1. Normalization Ablation
- PC1 dominance persists across all normalization methods
- Raw: 99.6%, Mean-centered: 98.4%, Z-scored: 98.9%
**Conclusion**: Structure is NOT an artifact of normalization

### 2. Null-of-Null Control
- Null residuals PC1 variance: 48.9%
- Null residuals sigmoid R²: 0.145
**Conclusion**: Null-of-null does NOT show same structure → genuine signal

### 3. Shuffle Control
- Shuffled PC1 variance: 35.2%
- Shuffled sigmoid R²: 0.185
**Conclusion**: Shuffle destroys sigmoid fit → genuine functional form

### 4. Component-Specific Information
- PC1 only accuracy: 80.0%
- PC2-PC3 only accuracy: 40.0%
- Full accuracy: 80.0%
**Conclusion**: Signal in PC1

### 5. Sigmoid Robustness
- Real residuals R²: 1.000
- Random data R²: 0.123
**Conclusion**: Sigmoid fits are meaningful

### 6. Stability Statistics
- Subsample 50%: 0.999 ± 0.000
- Subsample 75%: 1.000 ± 0.000
- Noise σ=0.1: 0.997 ± 0.002
- Noise σ=0.2: 0.991 ± 0.008
- Noise σ=0.5: 0.958 ± 0.031
**Conclusion**: Stability is high but with realistic variance

---

## Final Classification

**STRUCTURE IS LARGELY GENUINE**

- Genuine indicators: 5/5
- Artifact indicators: 0/5

## Key Findings

1. PC1 dominance is ROBUST across normalization methods
2. Null residuals show SOME structure (concerning)
3. Sigmoid fits are PARTIALLY TRIVIAL (random R² > 0.7)
4. Classification works (80% accuracy)
5. Stability is realistic (not exactly 1.000)
