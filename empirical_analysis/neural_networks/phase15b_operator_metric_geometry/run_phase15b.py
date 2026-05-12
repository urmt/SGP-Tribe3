import os
import json
import numpy as np

BASE_PATH = "/home/student/sgp-tribe3/empirical_analysis/neural_networks"
PHASE15A_PATH = os.path.join(BASE_PATH, "phase15a_operator_metric")
OUTPUT_PATH = os.path.join(BASE_PATH, "phase15b_operator_metric_geometry")

INPUT_FILE = os.path.join(PHASE15A_PATH, "results.json")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("ERROR: Phase 15A results not found")

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

operators = data["operators"]
phi = data["phi"]

if len(operators) != 5:
    raise RuntimeError("ERROR: Expected 5 operators")

def vectorize(p):
    return np.array([p["F"], p["DET"], p["alpha"]], dtype=np.float64)

Phi = np.array([vectorize(phi[o]) for o in operators])

if Phi.shape != (5, 3):
    raise RuntimeError("ERROR: Φ matrix shape incorrect")

mean = np.mean(Phi, axis=0)
std = np.std(Phi, axis=0)

if np.any(std == 0):
    raise RuntimeError("ERROR: Zero variance dimension")

Phi_norm = (Phi - mean) / std

N = len(operators)
D = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        D[i, j] = np.linalg.norm(Phi_norm[i] - Phi_norm[j])

violations = []
max_violation = 0.0
total_tests = 0

for i in range(N):
    for j in range(N):
        for k in range(N):
            if i != j and j != k and i != k:
                lhs = D[i, k]
                rhs = D[i, j] + D[j, k]

                total_tests += 1

                if lhs > rhs:
                    violation = lhs - rhs
                    violations.append({
                        "i": operators[i],
                        "j": operators[j],
                        "k": operators[k],
                        "lhs": float(lhs),
                        "rhs": float(rhs),
                        "violation": float(violation)
                    })

                    if violation > max_violation:
                        max_violation = violation

num_violations = len(violations)
violation_rate = num_violations / total_tests if total_tests > 0 else 0.0

if np.isnan(D).any():
    raise RuntimeError("ERROR: NaN in distance matrix")

if total_tests == 0:
    raise RuntimeError("ERROR: No triangle tests executed")

output = {
    "operators": operators,
    "normalized_phi": Phi_norm.tolist(),
    "distance_matrix": D.tolist(),
    "triangle_test": {
        "total_tests": total_tests,
        "num_violations": num_violations,
        "violation_rate": violation_rate,
        "max_violation": max_violation,
        "violations": violations[:20]
    }
}

with open(os.path.join(OUTPUT_PATH, "results.json"), "w") as f:
    json.dump(output, f, indent=4)

print("\nPHASE 15B COMPLETE — TRIANGLE INEQUALITY TESTED")
print(f"Total tests: {total_tests}")
print(f"Violations: {num_violations}")
print(f"Violation rate: {violation_rate:.6f}")
print(f"Max violation: {max_violation:.6f}")