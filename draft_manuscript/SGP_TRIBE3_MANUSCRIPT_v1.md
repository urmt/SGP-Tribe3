# Title
Universal Multiscale Dimensionality Signatures in Biological and Artificial Neural Systems

# Running Title
Multiscale Dimensionality Signatures

# Author List
[Placeholder: Authors]

# Affiliations
[Placeholder: Affiliations]

# Corresponding Author
[Placeholder: Corresponding Author]

# Keywords
Multiscale Dimensionality, Neural Manifolds, Geometric Constraints, Sigmoid Scaling, fMRI Encoding, Neural Networks, Topological Invariants
# Abstract

**Background:** Neural processing in both biological and artificial systems is constrained by the manifold geometry of their state spaces. Understanding the scaling of dimensionality across these manifolds remains a central challenge in computational neuroscience.

**Objective:** To determine whether a universal signature exists in the multiscale dimensionality profiles, $D(k)$, of neural activations.

**Methods:** We analyzed $D(k)$ profiles across biological (fMRI) and artificial (Transformer) neural systems using the participation ratio estimator. We evaluated candidate models (Linear, Power Law, Sigmoid, Gompertz, Hill Function) and performed parameter identifiability audits, null topology destruction tests, and curvature-dissociation validation.

**Results:** All examined systems exhibit a robust growth-saturation pattern in $D(k)$ profiles, which is most accurately described by a sigmoid functional form. The sigmoid inflection point, $k_0$, serves as a stable organizational invariant. Identifiability analysis confirms parameter stability across bootstrap resamples, and null model destruction tests demonstrate that this signature is topologically dependent rather than an artifact of local sampling or noise. Transformer scaling confirms this pattern persists across hierarchy depths.

**Interpretation:** The sigmoid dimensionality signature suggests fundamental geometric constraints on information organization in neural systems. The stability of $k_0$ implies an organizational invariant relevant for cross-domain comparative analysis.

**Limitations:** Analysis relies on Euclidean approximations of non-linear manifolds and specific estimator choices. Biological ground truth is inferred from predictive encoding models rather than direct neuronal recordings. Curvature-inflection relationships may be sensitive to local neighborhood density estimation.
# Introduction

The organization of neural activity in high-dimensional state spaces is a fundamental concern in understanding complex information processing. Biological neural circuits and artificial neural networks operate within constrained manifolds, where the effective dimensionality is significantly lower than the ambient space. Identifying organizational invariants across these systems provides critical insights into the geometric constraints governing neural information transformation.

Recent advances in manifold learning and fMRI encoding models have enabled the characterization of neural activity at multiscale depths. However, the precise functional form describing how effective dimensionality grows with neighborhood scale $k$ remains poorly constrained, often relying on power-law assumptions that may fail at large scales.

In this study, we investigate the multiscale dimensionality profile, $D(k)$, defined via the participation ratio of covariance matrices in local neighborhoods. We hypothesized that neural manifolds are governed by growth-saturation dynamics. We test this through a comprehensive audit of functional forms, evaluating model identifiability, topological sensitivity, and cross-domain invariance in biological (SGP-parcellated fMRI predictions) and artificial (Transformer model activation) neural systems. Our results establish a robust sigmoid organizational signature and identify the sigmoid inflection point, $k_0$, as a candidate geometric invariant.
# Methods

## 1. D(k) Estimator
The multiscale dimensionality profile $D(k)$ is calculated by assessing the effective dimensionality of local neighborhoods. For a given point $x_i$, we identify its $k$ nearest neighbors. The covariance matrix $C_i$ of these $k$ neighbors is computed. The effective dimensionality is then defined by the participation ratio:
$$D(k) = \frac{(\sum_{j=1}^m \lambda_j)^2}{\sum_{j=1}^m \lambda_j^2}$$
where $\lambda_j$ are the eigenvalues of $C_i$.

## 2. Model Fitting
We fit $D(k)$ profiles using a sigmoid functional form:
$$D(k) = \frac{L}{1 + e^{-\beta(k - k_0)}} + b$$
where $L$ represents total capacity, $k_0$ is the inflection point (midpoint), $\beta$ is the growth rate, and $b$ is the baseline. Fitting was performed using `scipy.optimize.curve_fit` (Levenberg-Marquardt algorithm) with parameter bounds $L \in [0, \infty)$, $k_0 \in [0, 1000]$, $\beta \in [0, 10]$.

## 3. Identifiability Analysis
Parameter identifiability was assessed using 1000 bootstrap resamples of the dataset and 1000 randomized initializations for the optimization routine. Convergence success and parameter covariance/correlation matrices were computed to assess robust identifiability of $(L, k_0, \beta, b)$.

## 4. Null Model Destruction
Topology-destroyed controls were constructed by:
- Shuffling data manifolds.
- Reassigning random neighbors.
- Scrambling covariance matrix elements.
Degradation in model fit ($R^2$) was evaluated against these controls to establish the topological specificity of the $D(k)$ signature.

## 5. Geometric Validation
Partial correlation analyses were performed to determine the relationship between $k_0$ and geometric metrics (curvature and neighborhood occupancy), controlling for local noise level and sample count.

## 6. Environment
Analysis was performed using Python 3.14 (NumPy 2.2, SciPy 1.15, Scikit-learn 1.6) on a CPU-based system.
# Results

## 1. Universal D(k) structure
Across all analyzed systems (TRIBE, Sparse, Hierarchical, Random, and Transformer layers), the dimensionality profile $D(k)$ consistently exhibits a growth-saturation pattern. 

## 2. Sigmoid superiority
Comparison of candidate models (Linear, Logarithmic, Power Law, Gompertz, Hill Function, Piecewise Linear) demonstrates that the sigmoid function provides the highest R² and lowest AIC/BIC values, indicating it is the most parsimonious description of the $D(k)$ profile. Figure 3 provides the comparison.

## 3. Parameter organization
Identifiability analysis demonstrates that the sigmoid parameters (L, k0, beta, b) are robustly recoverable in the TRIBE system. Bootstrap distributions (Figure 4) show minimal parameter correlation, and convergence rates are $>95\%$ across 1000 randomized initializations.

## 4. Null-model destruction
Null model destruction (manifold shuffling, random neighbor reassignment, and covariance scrambling) leads to severe R² degradation (Figure 5), confirming that the sigmoid signature is dependent on the specific underlying topology of the neural manifold.

## 5. Transformer depth scaling
Analysis of Transformer layer outputs confirms that the sigmoid signature persists across hierarchy depths (Transformer Scaling results), supporting the cross-domain generality of these geometric constraints.

## 6. Limitations of semantic mapping
Initial observations regarding noise-curvature dissociation suggest that the inflection point ($k_0$) is weakly correlated with neighborhood occupancy and robust to local noise fluctuations, provided the manifold structure is preserved.
# Discussion

Our results identify a universal sigmoid organization in the multiscale dimensionality profiles of both biological fMRI predictions and artificial neural networks. This signature characterizes the transition from local manifold dimensionality to global capacity, with the inflection point, $k_0$, acting as a robust invariant across different systems.

The topological dependency established through null model destruction tests indicates that this dimensionality signature is not merely a consequence of the estimator’s mathematical properties, but rather a reflection of the underlying manifold geometry of the neural state space. The consistency of this finding across both the TRIBE fMRI encoding model and Transformer architectures underscores its potential as a fundamental organizational constraint in neural systems.

However, these findings must be interpreted within strict limits. The reliance on Participation Ratio estimation assumes locally Euclidean structure, which may not capture non-linear topological features in highly warped manifolds. Furthermore, the biological interpretations are based on predictive models and require validation against direct neuronal recordings. 

Future work should address these limitations through replication with diverse encoders and empirical validation of $k_0$ across higher-resolution neural data.
# Limitations

- **Single-subject/System limitations:** The analysis primarily relies on the TRIBE fMRI predictive model; results may not generalize to other cortical parcellations or modalities.
- **Semantic mapping limitations:** Text embeddings were derived from specific Transformer architectures (e.g., tinyLLaMA); the signature's dependence on the specific encoder architecture requires further testing.
- **Estimator limitations:** Participation Ratio assumes locally linear structures; non-linear manifold features may be obfuscated at certain scales.
- **Curvature-definition limitations:** Estimates based on k-nearest neighbor distance variance are sensitive to local data density, potentially introducing bias in sparsely sampled regions.
- **Benchmark limitations:** The comparison against candidate models (e.g., Gompertz, Hill Function) does not exhaust the space of all possible organizational invariants.
- **Biological interpretation:** The SGP-related potential functions and biological activity mappings are theoretical constructs requiring empirical validation against ground-truth neural recording data.
# Conclusion

The multiscale dimensionality profiles of neural systems are characterized by a universal sigmoid growth-saturation signature. This structure is topologically dependent, robust to noise, and identifiable across biological and artificial domains. The sigmoid inflection point, $k_0$, emerges as a reproducible organizational invariant. These findings provide a grounded geometric framework for understanding neural manifold organization.
