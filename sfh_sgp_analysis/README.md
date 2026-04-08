# SFH-SGP Analysis Module

**Sentient-Field Hypothesis / Sentient-Generative Principal Mathematical Framework**

This module implements the mathematical analysis of SGP cortical activation patterns using the SFH-SGP topological field framework.

## Files

- `sfh_sgp_comprehensive.py` - Main analysis: computes Q, C, F, χ, basin clustering, classification, node contributions
- `generate_figures.py` - Figure generation from results

## Usage

```bash
# Run comprehensive SFH-SGP analysis
python sfh_sgp_comprehensive.py

# Generate visualizations
python generate_figures.py
```

## Output

Results are saved to `results/full_battery_1000/sfh_sgp/`:
- `sfh_sgp_final_results.json` - Summary statistics
- `sfh_sgp_comprehensive_results.json` - Full analysis results
- `delta_matrix.npy` - Differential activation matrix
- `chi_per_category.npy` - χ values per category
- `figures/` - Visualization figures

## SFH-SGP Primitives

| Symbol | Name | Formula |
|--------|------|---------|
| Q | Sentient Quota | $\sum \|\delta_i\|$ |
| F | Fertility | G5_dmn differential |
| C | Coherence | Leading eigenvector projection |
| χ | Sentient Potential | $\alpha C + \beta F$ |

## Strength Levels

Results are annotated with strength levels:

- **STRONG**: Empirically validated, statistically significant findings
- **MODERATE**: Theoretically grounded, consistent with predictions
- **EXPLORATORY**: Novel predictions, requires further validation

## References

- Traver, M.R. (2026). The Mathematical Atlas of Reality: The 5-Level SFH-SGP Hierarchy
- Traver, M.R. (2026). The 1,000 Qubit Wall — Proof of the 10²⁴ Finite Loop
