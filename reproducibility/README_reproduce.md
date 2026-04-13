# Reproducibility Bundle for SGP-Tribe3 Scale-Dependent Dimensionality Analysis

## Contents

| File | Description |
|------|-------------|
| `raw_data.npz` | Compressed NumPy archive containing raw TRIBE v2 predictions |
| `pipeline_reproduce.py` | Python script to reproduce main results |
| `requirements.txt` | Python dependencies |
| `README_reproduce.md` | This file |

## Raw Data Format

The `raw_data.npz` file contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `sgp_nodes` | (11522, 9) | SGP cognitive node activations |
| `streams` | (11522, 5) | Processing stream activations |
| `edge_weights` | (11522, 9) | White matter tract weights |

### SGP Node Indices (0-8)
0. G1_broca
1. G2_wernicke
2. G3_tpj
3. G4_pfc
4. G5_dmn
5. G6_limbic
6. G7_sensory
7. G8_atl
8. G9_premotor

### Stream Indices (0-4)
0. dorsal
1. ventral
2. generative
3. modulatory
4. convergence

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python pipeline_reproduce.py
```

### 3. Expected Outputs
The script will generate:
- `D_eff_curves.csv` - Effective dimensionality vs neighborhood size
- `beta_curves.csv` - Growth exponent vs neighborhood size
- `phase_model_results.csv` - Bilinear model parameters and R² values

## Method Summary

### D_eff(k) Computation
1. For each data point, identify k nearest neighbors using Euclidean distance
2. Compute local covariance matrix from neighbors
3. Calculate participation ratio: D_eff = (Σλ)² / Σλ²
4. Average across all points

### Beta(k) Computation
- Sliding window log-log regression on D_eff(k)
- Window size: 3 points
- Units: dD_eff / d(log k)

### Phase Model
- Bilinear equation: dβ/dlog(k) = a·β + b·D_eff + c·(β·D_eff)
- Fit via nonlinear least squares

## Expected Results

| System | D_eff (k=5) | D_eff (k=500) | Peak D_eff |
|--------|-------------|---------------|------------|
| sgp_nodes | ~2.5 | ~4.5 | ~4.9 |
| streams | ~2.0 | ~3.5 | ~4.0 |
| edge_weights | ~2.2 | ~4.0 | ~4.5 |

## Citation

If using this data, cite:
```
Traver, M.R. (2026). Scale-Dependent Dimensionality Reveals a Common 
Structural Motif Across Representational Systems.
GitHub: https://github.com/urmt/SGP-Tribe3
```
