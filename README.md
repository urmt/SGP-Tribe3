# SGP-Tribe3

**Compositional and Dynamical Metrics of Cortical Organization: Alignment with the Principal Cortical Gradient**

Analysis code for manuscript examining how metrics derived from representational data align with the principal cortical gradient from Margulies et al. (2016).

## Overview

This repository contains the V7 analysis testing whether:
1. **Composite Dispersion Index (CDI)**: A nonlinear combination of activation magnitude and dispersion
2. **Dynamical Torsion (χ)**: A measure inspired by the SFH-SGP framework capturing rotational dynamics

...align with established cortical organization as measured by the principal cortical gradient.

## Key Findings (V7 - April 2026)

| Metric | r with Gradient | p-value | Partial (|norm) |
|--------|----------------|---------|-----------------|
| **χ (torsion)** | **-0.183** | **2.3×10⁻⁴** | **-0.297** |
| CDI | +0.104 | 0.038 | +0.078 (n.s.) |
| L2 norm | +0.204 | 3.8×10⁻⁵ | - |

**Key finding:** χ maintains significant partial correlation after controlling for norm (r=-0.297, p<10⁻⁸), demonstrating it's NOT reducible to variance/norm.

## Repository Structure

```
sgp-tribe3/
├── manuscript/v7/                      # Current publication-ready manuscript
│   ├── manuscript.tex                 # LaTeX manuscript
│   ├── references.bib                 # Bibliography
│   ├── v7_complete_pipeline.py        # Analysis pipeline
│   ├── analysis_results.csv           # Full data (400 parcels)
│   └── figures/                       # Visualization figures
└── results/
    └── full_battery_1000/            # TRIBE v2 predictions
```

## Repository Structure

```
sgp-tribe3/
├── sfh_sgp_analysis/
│   ├── phase1_discovery_analysis.py      # Discovery analysis (10 categories)
│   ├── phase2_expanded_analysis.py      # Expanded analysis (30 categories)
│   └── sfh_sgp_v2_analysis.py          # V2 analysis suite
├── data/
│   ├── stimulus_bank.json                # Original 1,022 stimuli
│   └── expanded_stimulus_bank_v2.json   # V2: 400 additional stimuli
├── results/
│   ├── full_battery_1000/               # Original TRIBE v2 predictions
│   └── phase2_combined.json             # V2 combined dataset
└── generate_expanded_categories.py       # Stimulus generation
```

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib pandas nilearn
```

## Quick Start

```bash
cd manuscript/v7
python v7_complete_pipeline.py
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Key Metrics

### Composite Dispersion Index (CDI)
\[
\text{CDI} = \|x\| \cdot \text{Var}(x) \cdot \text{Std}(x)
\]

### Dynamical Torsion (χ)
\[
\chi = \|A\|_F \quad \text{where } A = (J - J^T)/2
\]
- J: Jacobian of velocity with respect to state
- A: Antisymmetric component capturing rotational dynamics

## Methods

1. **Data**: TRIBE v2 predictions for 1,422 stimuli × 30 categories
2. **Trajectories**: Generated pseudo-temporal trajectories per parcel
3. **Metrics**: CDI and χ computed for Schaefer-400 parcels (n=400)
4. **Validation**: Spin permutation testing (n=1,000)

## Citation

```bibtex
@article{SGPTribe3V7,
  title={Compositional and Dynamical Metrics of Cortical Organization: 
         Alignment with the Principal Cortical Gradient},
  author={Traver, Mark Rowe},
  year={2026}
}
```

## References

- **Cortical Gradient**: Margulies et al. (2016), PNAS
- **TRIBE v2**: d'Ascoli et al. (2026), Meta AI
- **Parcellation**: Schaefer et al. (2018), Cerebral Cortex

## License

CC BY 4.0

## Related

- **SFH-SGP Framework**: https://doi.org/10.5281/zenodo.19472283
- **TRIBE v2**: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
