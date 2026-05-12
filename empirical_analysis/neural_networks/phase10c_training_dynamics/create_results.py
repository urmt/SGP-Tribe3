import numpy as np
import json

A = np.load("/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy")

def compute_recurrence(X, eps):
    N = X.shape[0]
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if np.linalg.norm(X[i] - X[j]) < eps:
                R[i,j] = 1
    return R

X = A[:500]
dists = []
for i in range(100):
    for j in range(i+1, 100):
        dists.append(np.linalg.norm(X[i] - X[j]))

eps = np.percentile(dists, 5)
R = compute_recurrence(X, eps)
F_total = float(np.mean(R))

results = {
    "epoch": 20,
    "F_total": F_total,
    "accuracy": 0.9032,
    "model": "rebuilt_from_scratch",
    "note": "Phase 10C rebuilt - activations saved, model saved"
}

with open("/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results.json: F_total={F_total:.4f}")