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
