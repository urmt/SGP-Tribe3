# Methods: Identifiability Audit (SGP-TRIBE3)

## 1. Model Definitions
- **Sigmoid**: $D(k) = L / (1 + \exp(-\beta(k - k_0))) + b$
- **Linear**: $D(k) = mk + b$
- **Logarithmic**: $D(k) = a \ln(k) + b$
- **Power Law**: $D(k) = a k^b + c$
- **Gompertz**: $D(k) = A \exp(-\exp(-k(k - x_0))) + b$
- **Hill Function**: $D(k) = V_{max} k^n / (K^n + k^n) + b$
- **Piecewise Linear**: Two-segment linear regression with variable breakpoint.

## 2. Optimizer & Setup
- **Optimizer**: `scipy.optimize.curve_fit` (Levenberg-Marquardt algorithm).
- **Initialization**: Informed guess based on data range/median; random restarts for robustness.
- **Bootstrap**: 1000 resamples for covariance estimation.
- **Hardware**: CPU-based execution.
- **Environment**: Python 3.14, NumPy 2.2, SciPy 1.15, Scikit-learn 1.6.

## 3. Methodology
- **Identifiability**: Bootstrap resampling & random initialization convergence.
- **Null Destruction**: Shuffled manifold, random neighbors, covariance scrambling.
- **Geometric Validation**: Partial correlation of $k_0$ with curvature and neighborhood density, controlling for noise/sample counts.
