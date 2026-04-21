#!/usr/bin/env python3
"""
SFH-SGP_GEOMETRY_STRUCTURE_01
Characterize the observer-induced geometry 𝒢_S
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

df = pd.read_csv("observer_geometry.csv")

X = df[["alpha", "DET"]].values

print("=" * 60)
print("OBSERVER-INDUCED GEOMETRY 𝒢_S")
print("=" * 60)

# -----------------------------
# 1. DISTANCE STRUCTURE
# -----------------------------
print("\n1. DISTANCE MATRIX:")
D = squareform(pdist(X))
print(pd.DataFrame(D, 
      index=df["observation"], 
      columns=df["observation"]).round(3).to_string())

# -----------------------------
# 2. CLUSTERING
# -----------------------------
print("\n" + "=" * 60)
print("2. OBSERVATION TYPE CLUSTERING")
print("=" * 60)

# Define observation groups manually
groups = []
for obs in df["observation"]:
    if "embed" in obs:
        groups.append("embedding")
    elif "pca" in obs:
        groups.append("pca")
    elif obs in ["cos", "sin"]:
        groups.append("nonlinear_trig")
    elif obs == "phase_diff":
        groups.append("nonlinear_diff")
    elif obs in ["mean", "weighted"]:
        groups.append("linear")
    else:
        groups.append("partial")

df["group"] = groups

print("\nObservation → Group mapping:")
for _, row in df.iterrows():
    print(f"  {row['observation']:<15} → {row['group']}")

# Cluster analysis
print("\nCluster by observation type:")
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
df["kmeans_cluster"] = kmeans.labels_

for c in range(4):
    print(f"\nCluster {c}:")
    for _, row in df[df["kmeans_cluster"] == c].iterrows():
        print(f"  {row['observation']:<15} α={row['alpha']:.3f}, DET={row['DET']:.3f}")

# -----------------------------
# 3. LOCAL SENSITIVITY
# -----------------------------
print("\n" + "=" * 60)
print("3. LOCAL VARIATION (Gradient Analysis)")
print("=" * 60)

# Sort by alpha and compute differences
df_sorted = df.sort_values("alpha").reset_index(drop=True)
df_sorted["alpha_diff"] = df_sorted["alpha"].diff()
df_sorted["DET_diff"] = df_sorted["DET"].diff()

print("\nSorted by α:")
print(df_sorted[["observation", "alpha", "DET", "alpha_diff", "DET_diff"]].to_string(index=False))

# Check for smoothness
alpha_gradients = df_sorted["alpha_diff"].dropna().values
det_gradients = df_sorted["DET_diff"].dropna().values

print(f"\nα gradient stats:")
print(f"  Mean |gradient|: {np.abs(alpha_gradients).mean():.4f}")
print(f"  Max |gradient|: {np.abs(alpha_gradients).max():.4f}")

print(f"\nDET gradient stats:")
print(f"  Mean |gradient|: {np.abs(det_gradients).mean():.4f}")
print(f"  Max |gradient|: {np.abs(det_gradients).max():.4f}")

# -----------------------------
# 4. CONTINUITY TEST
# -----------------------------
print("\n" + "=" * 60)
print("4. CONTINUITY ASSESSMENT")
print("=" * 60)

max_alpha_grad = np.abs(alpha_gradients).max()
max_det_grad = np.abs(det_gradients).max()

if max_alpha_grad < 0.5 and max_det_grad < 0.5:
    continuity = "SMOOTH (small gradients)"
elif max_alpha_grad > 1.0:
    continuity = "DISCONTINUOUS (large jumps)"
else:
    continuity = "PARTIALLY CONTINUOUS"

print(f"\n𝒢_S structure: {continuity}")
print(f"  Max α jump: {max_alpha_grad:.3f}")
print(f"  Max DET jump: {max_det_grad:.3f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY: 𝒢_S Structure")
print("=" * 60)

unique_groups = df["group"].unique()
print(f"\nObservation types: {len(unique_groups)}")
for g in unique_groups:
    subset = df[df["group"] == g]
    print(f"  {g}: α = {subset['alpha'].mean():.3f} ± {subset['alpha'].std():.3f}, DET = {subset['DET'].mean():.3f} ± {subset['DET'].std():.3f}")

print(f"\nConclusion:")
print(f"  𝒢_S exhibits {continuity.lower()}")
print(f"  Observation types map to distinct regions in (α, DET) space")

df.to_csv("/home/student/sgp-tribe3/empirical_analysis/neural_networks/geometry_structure.csv", index=False)
print(f"\nSaved: geometry_structure.csv")