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
