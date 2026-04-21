# Neural Networks Validation Experiments

This directory contains clean-system validation experiments using neural networks to test SFH-SGP dimensionality predictions.

## Purpose

Test whether D(k) encodes task/regime information in a CLEAN, noise-free system.

## Approach

- Use trained neural networks with different architectures
- Generate activity data across different tasks/regimes
- Compute k-NN dimensionality D(k) at k = [2, 4, 8, 16]
- Validate with shuffle controls

## Hard Rules

- NO synthetic data - use real network activations
- Shuffle validation MANDATORY
- Print sample feature vectors
- Fail on NaN