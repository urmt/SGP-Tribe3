"""
OPENCODE RULES — DO NOT MODIFY

1. Use SAME data as Phase 7D
2. DO NOT recompute trajectories differently
3. DO NOT alter recurrence computation
4. DO NOT optimize loops
5. Dimension estimation must be explicit and logged
6. No placeholder constants allowed without justification

Goal: isolate dimension scaling effect only
"""

import numpy as np
import json

INPUT_PATH = "empirical_analysis/physics/phase7d_full_state/phase7d_results.json"
OUTPUT_PATH = "empirical_analysis/physics/phase7e_dimension_normalized/phase7e_results.json"

DIMENSIONS = {
    "harmonic": 1.0,
    "logistic": 1.0,
    "lorenz_full": 2.05
}

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

results = []

for entry in data:
    system = entry["system"]
    alpha = entry["alpha"]
    lyap = entry["lyapunov"]
    
    D = DIMENSIONS.get(system)
    if D is None:
        raise ValueError(f"Missing dimension for: {system}")
    
    alpha_norm = alpha / D
    
    results.append({
        "system": system,
        "lyapunov": lyap,
        "alpha_raw": alpha,
        "dimension": D,
        "alpha_normalized": alpha_norm
    })

alphas = np.array([r["alpha_normalized"] for r in results])
lyaps = np.array([r["lyapunov"] for r in results])

pearson = float(np.corrcoef(alphas, lyaps)[0, 1])

output = {"results": results, "correlation_alpha_norm_vs_lyapunov": pearson}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved: {OUTPUT_PATH}")
print(f"Correlation: {pearson}")
for r in results:
    print(r)