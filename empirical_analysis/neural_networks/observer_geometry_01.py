#!/usr/bin/env python3
"""
SFH-SGP_OBSERVER_GEOMETRY_01
How does (alpha, DET) change as a function of observation map?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

n_total = 6000
n_burn = 2000
sigma = 1e-3
eps_list = np.logspace(-4, -1, 12)
epsilon_det = 1e-2
l_min = 2

def kuramoto_full(K, n, N=10):
    theta = np.random.rand(N) * 2*np.pi
    xs = []
    for _ in range(n):
        dtheta = np.zeros(N)
        for i in range(N):
            dtheta[i] = np.sum(np.sin(theta - theta[i]))
        theta += (K/N)*dtheta + 0.1*np.random.randn(N)
        xs.append(theta.copy())
    return np.array(xs)

def recurrence_matrix(x, epsilon):
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    D = pairwise_distances(x)
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)
    return R

def recurrence_rate(R):
    N = R.shape[0]
    return np.sum(R) / (N*N - N)

def compute_alpha(x, eps_list):
    R_vals = []
    for eps in eps_list:
        R = recurrence_matrix(x, eps)
        R_vals.append(recurrence_rate(R))
    R_arr = np.array(R_vals)
    log_R = np.log(R_arr)
    log_eps = np.log(eps_list)
    valid = np.isfinite(log_R)
    if valid.sum() > 2:
        return np.polyfit(log_eps[valid], log_R[valid], 1)[0]
    return np.nan

def compute_det(R, l_min=2):
    N = R.shape[0]
    diag_counts = 0
    total_rec = np.sum(R)
    for k in range(-N+1, N):
        diag = np.diagonal(R, offset=k)
        length = 0
        for val in diag:
            if val == 1:
                length += 1
            else:
                if length >= l_min:
                    diag_counts += length
                length = 0
        if length >= l_min:
            diag_counts += length
    return diag_counts / total_rec if total_rec > 0 else 0

def delay_embed(x, m=3, tau=1):
    X = []
    for i in range(len(x) - (m-1)*tau):
        X.append([x[i + j*tau] for j in range(m)])
    return np.array(X)

np.random.seed(42)
K = 1.5
X = kuramoto_full(K, n_total)[n_burn:]
X += sigma * np.random.randn(*X.shape)

results = []

observations = {}

# linear projections
observations["mean"] = np.mean(np.cos(X), axis=1)
observations["weighted"] = np.dot(np.cos(X), np.linspace(1, 2, X.shape[1]))

# partial observations
observations["single"] = X[:, 0]
observations["subset_3"] = np.mean(X[:, :3], axis=1)

# nonlinear
observations["cos"] = np.mean(np.cos(X), axis=1)
observations["sin"] = np.mean(np.sin(X), axis=1)
observations["phase_diff"] = X[:, 0] - X[:, 1]

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
observations["pca_1"] = X_pca[:, 0]
observations["pca_2"] = X_pca[:, 1]

# delay embeddings
mean_obs = observations["mean"]
observations["embed_3"] = delay_embed(mean_obs, m=3)
observations["embed_5"] = delay_embed(mean_obs, m=5)

print("Running observer geometry test...")

for name, obs in observations.items():
    
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    
    alpha = compute_alpha(obs, eps_list)
    R = recurrence_matrix(obs, epsilon_det)
    det = compute_det(R)
    
    results.append({
        "observation": name,
        "alpha": alpha,
        "DET": det
    })

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("OBSERVER GEOMETRY RESULTS")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("GEOMETRIC STRUCTURE")
print("=" * 60)

groups = {
    "linear": df[df["observation"].isin(["mean", "weighted"])],
    "partial": df[df["observation"].isin(["single", "subset_3"])],
    "nonlinear": df[df["observation"].isin(["cos", "sin", "phase_diff"])],
    "pca": df[df["observation"].str.contains("pca")],
    "embedding": df[df["observation"].str.contains("embed")]
}

for group, subset in groups.items():
    if len(subset) > 0:
        print(f"\n{group.upper()}:")
        print(f"  α range: {subset['alpha'].min():.3f} - {subset['alpha'].max():.3f}")
        print(f"  DET range: {subset['DET'].min():.3f} - {subset['DET'].max():.3f}")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/observer_geometry.csv", index=False)
print(f"\nSaved: observer_geometry.csv")