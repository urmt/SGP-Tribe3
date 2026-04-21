# SFH-SGP_OBSERVER_GEOMETRY_01 (LOCKED)

## Research Question
How does (α, DET) change as a function of observation map?

## System
- Kuramoto model (K = 1.5, partially sync)
- Noise: σ = 1e-3

## Observation Types Tested

| Type | Observations |
|------|------------|
| Linear | mean, weighted mean |
| Partial | single oscillator, subset of 3 |
| Nonlinear | cos, sin, phase difference |
| PCA | PC1, PC2 |
| Embedding | delay-3, delay-5 |

## Results

| Observation | α | DET |
|-------------|-----|-----|
| mean | 0.991 | 0.385 |
| weighted | 0.985 | 0.032 |
| single | 1.001 | 0.060 |
| subset_3 | 0.999 | 0.106 |
| cos | 0.991 | 0.385 |
| sin | 0.876 | 0.864 |
| phase_diff | 0.997 | 0.078 |
| pca_1 | 1.002 | 0.107 |
| pca_2 | 0.997 | 0.106 |
| embed_3 | **2.480** | 0.568 |
| embed_5 | **3.185** | 0.805 |

## Geometric Structure

| Observation Type | α Range | DET Range |
|----------------|---------|----------|
| Linear | 0.985-0.991 | 0.032-0.385 |
| Partial | 0.999-1.001 | 0.060-0.106 |
| Nonlinear | 0.876-0.997 | 0.078-0.864 |
| PCA | 0.997-1.002 | 0.106-0.107 |
| **Embedding** | **2.48-3.19** | 0.568-0.805 |

## Key Finding

**Observation operator defines the geometry:**

- Different observation types cluster in different (α, DET) regions
- Delay embedding dramatically increases α (2.5→3.2)
- This confirms: (α, DET) is OBSERVER-DEPENDENT

## Interpretation

(α, DET) does not measure intrinsic system properties.
It measures the RELATIONSHIP between observer and system.

This transforms the result from:
- "dynamical classifier" → "observer-system mapping geometry"

## Locked Date
2026-04-20

## Status
**VERIFIED - DO NOT MODIFY**