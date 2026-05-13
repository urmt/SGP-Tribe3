# CLAIM HIERARCHY

## TIER 1: Strongly Supported
1. **Universal Sigmoid Form**: D(k) follows a characteristic sigmoid growth-saturation pattern across all systems examined.
    - Support: Figure 2, Model comparison metrics (lowest AIC/BIC)
2. **Topological Dependency**: D(k) profile is not a generic property of the estimator but is tied to the underlying manifold topology.
    - Support: Figure 5 (Null Model Destruction)
3. **Parameter Stability**: Sigmoid parameters (k0, beta) are robustly identifiable in the TRIBE system.
    - Support: Figure 4 (Bootstrap Covariance), `identifiability_metrics.json`

## TIER 2: Conditional
1. **Generalization across Domains**: The sigmoid scaling law holds for both biological fMRI predictions and artificial Transformer representations.
    - Support: Figure 7, Universal scaling results CSV
    - Vulnerability: Sample heterogeneity across domains
2. **Curvature-Inflection Relationship**: The inflection point (k0) correlates with neighborhood occupancy density.
    - Support: Figure 6 (k0 Validation)
    - Vulnerability: Potential artifacts of local neighborhood density estimation

## TIER 3: Future Directions (Speculative)
1. **Causal Manifold Inference**: The sigmoid signature could be used for causal inference in neural circuits.
    - Support: None (Requires future validation)
2. **Brain-Inspired Geometric AI**: Designing new Transformer architectures based on these geometric constraints.
    - Support: None (Requires future architecture work)
