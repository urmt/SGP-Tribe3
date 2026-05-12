import numpy as np
import json
import os
from scipy.fft import fft, ifft

seed = 42
np.random.seed(seed)

OUTPUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase11_frequency"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_recurrence_matrix(X, epsilon=0.1):
    N = X.shape[0]
    D = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    R = (D < epsilon).astype(float)
    np.fill_diagonal(R, 0)
    return R

def compute_DET(R, diagonal=False):
    return np.trace(R) / R.shape[0] if R.shape[0] > 0 else 0

def compute_F(R):
    N = R.shape[0]
    return np.sum(R) / (N * (N - 1))

def compute_C(R):
    N = R.shape[0]
    total = 0.0
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            total += np.abs(R[i, j] - R[j, i])
            count += 1
    return total / count if count > 0 else 0

def compute_alpha(R):
    vals = R[np.triu_indices(R.shape[0], k=1)]
    return np.std(vals)

print("Loading activations...")
X_full = np.load('/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase10c_training_dynamics/activations.npy')
print(f"Loaded: {X_full.shape}")

np.random.seed(seed)
idx = np.random.permutation(len(X_full))[:500]
X = X_full[idx]
print(f"Using subset: {X.shape}")

def fft_transform(X):
    return fft(X, axis=1)

BANDS = {
    "low": (0.0, 0.2),
    "mid": (0.2, 0.5),
    "high": (0.5, 1.0)
}

def bandpass_filter(X_fft, band):
    N = X_fft.shape[1]
    n_half = N // 2
    low_idx = int(band[0] * n_half)
    high_idx = int(band[1] * n_half)
    mask = np.zeros_like(X_fft)
    mask[:, low_idx:high_idx] = 1
    mask[:, N-low_idx:N] = 1
    return X_fft * mask

def reconstruct_signal(X_band):
    x = ifft(X_band, axis=1)
    return np.real(x)

X_fft = fft_transform(X)

RESULTS = {}

for band_name, band_range in BANDS.items():
    print(f"Processing band: {band_name}")
    
    X_band = bandpass_filter(X_fft, band_range)
    X_reconstructed = reconstruct_signal(X_band)
    
    X_norm = (X_reconstructed - np.mean(X_reconstructed, axis=1, keepdims=True)) / \
             (np.std(X_reconstructed, axis=1, keepdims=True) + 1e-8)
    
    R = compute_recurrence_matrix(X_norm, epsilon=2.0)
    
    DET = compute_DET(R)
    F = compute_F(R)
    C = compute_C(R)
    alpha = compute_alpha(R)
    
    RESULTS[band_name] = {
        "DET": float(DET),
        "F": float(F),
        "C": float(C),
        "alpha": float(alpha)
    }
    print(f"  {band_name}: F={F:.4f}, DET={DET:.4f}, C={C:.4f}, alpha={alpha:.4f}")

for band_name in RESULTS:
    if np.isnan(RESULTS[band_name]["F"]) or RESULTS[band_name]["F"] <= 0:
        raise ValueError(f"Invalid F for {band_name}")
    if RESULTS[band_name]["DET"] < 0 or RESULTS[band_name]["DET"] > 1:
        raise ValueError(f"Invalid DET for {band_name}")

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nSaved results.json")

print("\nPHASE 11 COMPLETE")