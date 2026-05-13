# Abstract

**Background:** Neural processing in both biological and artificial systems is constrained by the manifold geometry of their state spaces. Understanding the scaling of dimensionality across these manifolds remains a central challenge in computational neuroscience.

**Objective:** To determine whether a universal signature exists in the multiscale dimensionality profiles, $D(k)$, of neural activations.

**Methods:** We analyzed $D(k)$ profiles across biological (fMRI) and artificial (Transformer) neural systems using the participation ratio estimator. We evaluated candidate models (Linear, Power Law, Sigmoid, Gompertz, Hill Function) and performed parameter identifiability audits, null topology destruction tests, and curvature-dissociation validation.

**Results:** All examined systems exhibit a robust growth-saturation pattern in $D(k)$ profiles, which is most accurately described by a sigmoid functional form. The sigmoid inflection point, $k_0$, serves as a stable organizational invariant. Identifiability analysis confirms parameter stability across bootstrap resamples, and null model destruction tests demonstrate that this signature is topologically dependent rather than an artifact of local sampling or noise. Transformer scaling confirms this pattern persists across hierarchy depths.

**Interpretation:** The sigmoid dimensionality signature suggests fundamental geometric constraints on information organization in neural systems. The stability of $k_0$ implies an organizational invariant relevant for cross-domain comparative analysis.

**Limitations:** Analysis relies on Euclidean approximations of non-linear manifolds and specific estimator choices. Biological ground truth is inferred from predictive encoding models rather than direct neuronal recordings. Curvature-inflection relationships may be sensitive to local neighborhood density estimation.
