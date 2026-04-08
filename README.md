# SGP-Tribe3

**Sentient Generative Principal — Brain Encoding Calibration System**

A multimodal brain encoding system built on [TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta AI) for predicting cortical responses to text, audio, and video stimuli, with SGP 9-node parcellation.

## Overview

This repository contains:

1. **SGP Encoding Pipeline** - TRIBE v2 integration with SGP parcellation
2. **SFH-SGP Analysis** - Sentient-Field Hypothesis / Sentient-Generative Principal mathematical framework
3. **Stimulus Bank** - 1,022 text stimuli across 10 semantic categories

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Full Battery Analysis

```bash
python run_full_battery.py
```

### SFH-SGP Topological Analysis

```bash
python sfh_sgp_analysis/sfh_sgp_comprehensive.py
```

## Project Structure

```
sgp-tribe3/
├── sfh_sgp_analysis/       # SFH-SGP mathematical analysis
│   ├── sfh_sgp_comprehensive.py   # Main analysis script
│   └── generate_figures.py       # Figure generation
├── data/                    # Embeddings and projections
├── sgp_parcellation.py     # SGP node definitions
├── run_full_battery.py     # Main analysis pipeline
└── requirements.txt        # Dependencies
```

## SGP Node Definitions

The 9 SGP nodes represent distinct functional territories:

| Node | Region | Function | Stream |
|------|--------|----------|--------|
| G1_broca | Inferior Frontal Gyrus | Speech production | Dorsal |
| G2_wernicke | Superior Temporal Gyrus | Language comprehension | Ventral |
| G3_tpj | Temporoparietal Junction | Theory of mind | Convergence |
| G4_pfc | Prefrontal Cortex | Executive function | Dorsal |
| G5_dmn | Default Mode Network | Self-referential | Generative |
| G6_limbic | Limbic System | Emotion | Modulatory |
| G7_sensory | Sensory Cortex | Multisensory | Ventral |
| G8_atl | Anterior Temporal Lobe | Semantic integration | Ventral/Convergence |
| G9_premotor | Premotor Cortex | Motor planning | Dorsal |

## SFH-SGP Framework

The Sentient-Field Hypothesis / Sentient-Generative Principal (SFH-SGP) framework analyzes cortical activation patterns using mathematical topology.

### Key Equations

**Sentient Potential:**
$$\chi = \alpha C + \beta F$$

Where:
- $C$ = Coherence (from differential co-activation)
- $F$ = Fertility (G5_dmn differential)

**Langevin Dynamics:**
$$\frac{dq}{dt} = -\nabla\chi + \sqrt{2D}\cdot\xi(t)$$

### Key Findings

The SFH-SGP analysis reveals **two basins of attraction** in semantic category space:

- **Basin 1 "Real-World"**: motor, auditory, emotional, memory, simple, social
- **Basin 2 "Intellectual"**: abstract, factual, logical, spatial

Statistical validation:
- $t = 8.69$, $p = 2.4 \times 10^{-5}$
- Cohen's $d = 5.75$ (very large effect)
- 100% Leave-One-Out Cross-Validation accuracy

## Citation

If you use this code, please cite:

```bibtex
@article{SGPTribe3,
  title={The Geometry of Thought: A Hilbert Space Analysis of Semantic Categories 
         Reveals Two Basins of Attraction in the SGP Topographic Field},
  author={Traver, Mark Rowe},
  year={2026}
}

@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stéphane and Rapin, Jérémy and others},
  journal={Nature},
  year={2026}
}
```

## License

CC BY-NC 4.0
