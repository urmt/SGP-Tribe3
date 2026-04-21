#!/usr/bin/env python3
"""
SFH-SGP_CLUSTER_ANALYSIS_01
Test whether transition types separate naturally in D(k) feature space
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/"

print("=" * 70)
print("STEP 1: BUILD FEATURE MATRIX")
print("=" * 70)

feature_data = {
    'Logistic': {
        'max_derivative': 1123.4670,
        'max_jump': 22.3947,
        'baseline_std': 1.7639,
        'curvature': 110269.4826
    },
    'Kuramoto': {
        'max_derivative': 1.6237,
        'max_jump': 0.4598,
        'baseline_std': 0.0767,
        'curvature': 29.8292
    },
    'OU': {
        'max_derivative': 6.9311,
        'max_jump': 0.6931,
        'baseline_std': 0.3666,
        'curvature': 33.0679
    }
}

systems = list(feature_data.keys())
features = ['max_derivative', 'max_jump', 'baseline_std', 'curvature']

X = np.array([[feature_data[s][f] for f in features] for s in systems])

print(f"\nFeature matrix: {X.shape}")
print(f"Systems: {systems}")
print(f"Features: {features}")

print("\nRaw features:")
for i, sys in enumerate(systems):
    print(f"  {sys}: {X[i]}")

print("\n" + "=" * 70)
print("STEP 2: NORMALIZATION")
print("=" * 70)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

print("\nNormalized features:")
for i, sys in enumerate(systems):
    print(f"  {sys}: {X_norm[i]}")

print("\n" + "=" * 70)
print("STEP 3: PCA REDUCTION")
print("=" * 70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

print(f"\nPCA components:")
print(f"  PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2 explains: {pca.explained_variance_ratio_[1]*100:.1f}%")

print(f"\nPC coordinates:")
for i, sys in enumerate(systems):
    print(f"  {sys}: PC1={X_pca[i,0]:.4f}, PC2={X_pca[i,1]:.4f}")

print("\n" + "=" * 70)
print("STEP 4: CLUSTERING")
print("=" * 70)

kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_2 = kmeans_2.fit_predict(X_norm)

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_3 = kmeans_3.fit_predict(X_norm)

print(f"\nK=2 clustering:")
for i, sys in enumerate(systems):
    print(f"  {sys}: Cluster {labels_2[i]}")

print(f"\nK=3 clustering:")
for i, sys in enumerate(systems):
    print(f"  {sys}: Cluster {labels_3[i]}")

print("\n" + "=" * 70)
print("STEP 5: DISTANCE ANALYSIS")
print("=" * 70)

dist_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        dist_matrix[i, j] = euclidean(X_norm[i], X_norm[j])

print(f"\nPairwise distances (normalized feature space):")
print(f"\n{'System':<12}", end="")
for s in systems:
    print(f" {s[:6]:>8}", end="")
print()
for i, sys in enumerate(systems):
    print(f" {sys:<12}", end="")
    for j in range(3):
        print(f" {dist_matrix[i,j]:>8.4f}", end="")
    print()

dist_orig = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        dist_orig[i, j] = euclidean(X[i], X[j])

print(f"\nPairwise distances (original scale):")
print(f"\n{'System':<12}", end="")
for s in systems:
    print(f" {s[:6]:>8}", end="")
print()
for i, sys in enumerate(systems):
    print(f" {sys:<12}", end="")
    for j in range(3):
        print(f" {dist_orig[i,j]:>8.2f}", end="")
    print()

print("\n" + "=" * 70)
print("STEP 6: OUTPUT")
print("=" * 70)

print(f"\n{'System':<12} {'PC1':<10} {'PC2':<10} {'Cluster_k2':<12} {'Cluster_k3':<12}")
print("-" * 60)
for i, sys in enumerate(systems):
    print(f" {sys:<12} {X_pca[i,0]:<10.4f} {X_pca[i,1]:<10.4f} {labels_2[i]:<12} {labels_3[i]:<12}")

print(f"\n{'Pairwise Distance Matrix':<30}")
print("-" * 30)
for i, sys in enumerate(systems):
    print(f"{sys:<12}: ", end="")
    for j in range(3):
        print(f"{dist_orig[i,j]:<8.2f}", end=" ")
    print()

print("\n" + "=" * 70)
print("FINAL QUESTIONS")
print("=" * 70)

log_separate = labels_2[0] != labels_2[1] and labels_2[0] != labels_2[2]
print(f"\n1. Does Logistic separate from others? {'YES' if log_separate else 'NO'}")

kur_ou_same = (labels_2[1] == labels_2[2])
if kur_ou_same:
    relation = "IDENTICAL"
else:
    relation = "DISTINCT"

print(f"2. Are Kuramoto and OU: {relation}")

types_match = (log_separate and kur_ou_same)
print(f"3. Does clustering match intuitive types? {'YES' if types_match else 'NO'}")

print("\n" + "=" * 70)
print("CLUSTERING INTERPRETATION")
print("=" * 70)

print("\nKey observations:")
print(f"  - Logistic is a clear OUTLIER in feature space")
print(f"  - Distance from Logistic to others: {dist_orig[0,1]:.2f} (Kuramoto), {dist_orig[0,2]:.2f} (OU)")
print(f"  - Distance between Kuramoto and OU: {dist_orig[1,2]:.2f}")
print(f"  - This reflects: DISCONTINUOUS vs CONTINUOUS transition types")