import numpy as np
import json
from scipy.stats import pearsonr, linregress

np.random.seed(42)

OUTPUT_DIR = "."

print("Loading spectral data...")
try:
    with open("../phase11a_spectral_replay/results.json") as f:
        spectral = json.load(f)
except:
    with open("../phase11a_spectral_replay/results.json") as f:
        spectral = json.load(f)

epoch_data = spectral["epoch_data"]

n_epochs = len(epoch_data)
print(f"Loaded {n_epochs} epochs from spectral replay")

if n_epochs < 15:
    print(f"ERROR: Insufficient data points: {n_epochs}")
    exit(1)

assert all(k in epoch_data[0] for k in ["accuracy", "entropy", "centroid", "concentration", "flatness"]), "Missing metrics"

print("Loading recurrence F_total data...")
paths_to_try = [
    "../phase10a_training_dynamics/results.json",
    "../phase10c_training_dynamics/results.json"
]

phase10c = None
for path in paths_to_try:
    try:
        with open(path) as f:
            phase10c = json.load(f)
            print(f"Loaded from: {path}")
            break
    except:
        continue

if phase10c is None:
    print("ERROR: Could not find Phase 10C results")
    exit(1)

F_values = [d["F_total"] for d in phase10c]

if len(F_values) != n_epochs:
    print(f"ERROR: Mismatch {len(F_values)} vs {n_epochs}")
    exit(1)

for i in range(n_epochs):
    epoch_data[i]["F_total"] = F_values[i]

print("Building data vectors...")
A = np.array([d["accuracy"] for d in epoch_data])
E = np.array([d["entropy"] for d in epoch_data])
CEN = np.array([d["centroid"] for d in epoch_data])
CONC = np.array([d["concentration"] for d in epoch_data])
FLAT = np.array([d["flatness"] for d in epoch_data])
F = np.array([d["F_total"] for d in epoch_data])

print("Computing correlations...")

r_F_CONC = pearsonr(F, CONC)[0]
r_F_ENTROPY = pearsonr(F, E)[0]
r_F_CENTROID = pearsonr(F, CEN)[0]
r_F_FLAT = pearsonr(F, FLAT)[0]
r_F_ACC = pearsonr(F, A)[0]
r_CONC_ACC = pearsonr(CONC, A)[0]

print(f"F vs CONC: {r_F_CONC:.3f}")
print(f"F vs entropy: {r_F_ENTROPY:.3f}")
print(f"F vs centroid: {r_F_CENTROID:.3f}")
print(f"F vs flatness: {r_F_FLAT:.3f}")
print(f"F vs accuracy: {r_F_ACC:.3f}")
print(f"CONC vs accuracy: {r_CONC_ACC:.3f}")

def partial_correlation(X, Y, Z):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    # Regress X on Z
    slope, intercept, _, _, _ = linregress(Z, X)
    pred_X = slope * Z + intercept
    resX = X - pred_X
    
    # Regress Y on Z
    slope, intercept, _, _, _ = linregress(Z, Y)
    pred_Y = slope * Z + intercept
    resY = Y - pred_Y
    
    return pearsonr(resX, resY)[0]

print("\nComputing partial correlations...")
partial_F_ACC_given_CONC = partial_correlation(F, A, CONC)
partial_CONC_ACC_given_F = partial_correlation(CONC, A, F)
partial_F_ACC_given_ENTROPY = partial_correlation(F, A, E)
partial_ENTROPY_ACC_given_F = partial_correlation(E, A, F)

print(f"Partial F_acc | CONC: {partial_F_ACC_given_CONC:.3f}")
print(f"Partial CONC_acc | F: {partial_CONC_ACC_given_F:.3f}")
print(f"Partial F_acc | entropy: {partial_F_ACC_given_ENTROPY:.3f}")
print(f"Partial entropy_acc | F: {partial_ENTROPY_ACC_given_F:.3f}")

print("\nValidation checks...")
if any(np.isnan(v) for v in [r_F_CONC, r_F_ENTROPY, r_F_CENTROID, r_F_FLAT, r_F_ACC]):
    print("ERROR: NaN values detected")
    exit(1)

if any(np.var(v) < 1e-10 for v in [A, E, CONC, FLAT, F]):
    print("ERROR: Zero variance in metrics")
    exit(1)

print("All checks passed")

results = {
    "n_epochs": n_epochs,
    "correlations": {
        "F_vs_CONC": r_F_CONC,
        "F_vs_entropy": r_F_ENTROPY,
        "F_vs_centroid": r_F_CENTROID,
        "F_vs_flatness": r_F_FLAT,
        "F_vs_accuracy": r_F_ACC,
        "CONC_vs_accuracy": r_CONC_ACC
    },
    "partial_correlations": {
        "F_acc_given_CONC": partial_F_ACC_given_CONC,
        "CONC_acc_given_F": partial_CONC_ACC_given_F,
        "F_acc_given_entropy": partial_F_ACC_given_ENTROPY,
        "entropy_acc_given_F": partial_ENTROPY_ACC_given_F
    }
}

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}/results.json")

print("\n=== SUMMARY ===")
if abs(r_F_ACC) > abs(r_CONC_ACC):
    print("Recurrence F_total is stronger predictor than CONC")
else:
    print("Spectral CONC is stronger predictor than F")
    
if abs(partial_F_ACC_given_CONC) < abs(r_F_ACC):
    print("Partial correlation suggests CONC mediates F-performance relationship")
else:
    print("Both F and CONC contribute independently")