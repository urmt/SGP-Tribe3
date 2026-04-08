# SGP-Tribe3 V2

**Applying the SFH-SGP Framework to TRIBE v2 Brain Predictions**

A multimodal brain encoding system built on [TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/) (Meta AI) for analyzing the topological structure of semantic space using the Sentient-Field Hypothesis / Sentient-Generative Principal (SFH-SGP) mathematical framework.

## Overview

This repository contains the analysis code for testing SFH-SGP predictions about the Torsion ($\chi$) and its role in creating basins of attraction in semantic space.

## Key Findings (V2 - April 2026)

**The Embodied Potential $\chi$ reveals topological structure in semantic space:**

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.661 |
| t-statistic | 8.38 |
| p-value | 4.1 × 10⁻⁹ |
| Cohen's d | 3.17 |
| Categories | 30 |
| Stimuli | 1,422 |

**Two Basins Identified:**
- **Real-World Basin** (15 categories): economic, gustatory, historical, linguistic, logical, mathematical, narrative, olfactory, political, religious, scientific, social_relational, spatial_nav, technological, temporal
- **Intellectual Basin** (15 categories): abstract, artistic, auditory, biological, emotional, factual, memory, motor, musical, procedural, proprioceptive, simple, social, spatial, visual

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
pip install numpy scipy scikit-learn matplotlib pandas
```

## Quick Start

### Phase 1: Discovery Analysis (10 categories)
```bash
python sfh_sgp_analysis/phase1_discovery_analysis.py
```

### Phase 2: Expanded Analysis (30 categories)
```bash
python sfh_sgp_analysis/phase2_expanded_analysis.py
```

## SFH-SGP Framework

### Key Primitives

| Symbol | Name | Description |
|--------|------|-------------|
| Q | Quota | Total differential flux ($\sum |x_i - \bar{x}_i|$) |
| C | Coherence | Projection onto leading eigenvector |
| F | Fertility | G7_sensory differential (sensorimotor) |
| $\chi$ | Torsion (τ) | Embodied Potential = \|αC\| + \|βF\| |

### Interpretation

$\chi$ (Torsion) represents the **Screening Effect** predicted by SFH-SGP:
- High $\chi$ = more constrained processing ("kinked hose")
- Low $\chi$ = less constrained processing ("unkinked hose")

The framework predicts that Torsion creates basins in activation space. Our analysis validates this prediction.

## Methods

1. **Encoding**: Text stimuli processed through TRIBE v2 pipeline → 9 SGP node activations
2. **Computation**: SFH-SGP primitives computed from node activation patterns
3. **Analysis**: Silhouette analysis, hierarchical clustering, generalization testing
4. **Validation**: Two-stage approach to avoid circular reasoning

## Citation

```bibtex
@article{SGPTribe3V2,
  title={The Topographic Structure of Semantic Space: A TRIBE v2 Analysis 
         Using the SFH-SGP Framework},
  author={Traver, Mark Rowe},
  year={2026}
}

@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stéphane and Rapin, Jérémy and others},
  journal={Meta AI Technical Report},
  year={2026}
}

@article{Traver2026SFHSGP,
  title={The Mathematical Atlas of Reality: The 5-Level SFH-SGP Hierarchy},
  author={Traver, Mark Rowe},
  year={2026},
  doi={10.5281/zenodo.19472283}
}
```

## License

CC BY 4.0

## Related

- **SFH-SGP Framework**: https://doi.org/10.5281/zenodo.19472283
- **TRIBE v2**: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
