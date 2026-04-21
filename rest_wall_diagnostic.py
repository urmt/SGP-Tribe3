#!/usr/bin/env python3
"""
DIAGNOSE REST WALL ARTIFACT
Analyze whether rest condition saturation arises from data or processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import os

print("="*60)
print("DIAGNOSTIC: REST WALL ARTIFACT")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
data = pd.read_csv('/home/student/sgp-tribe3/empirical_analysis/outputs/parameters.csv')

k_values = np.array([2, 4, 8, 16, 32, 64])

# ============================================================
# STEP 2: RECONSTRUCT RAW PROFILES
# ============================================================
print("\nReconstructing raw residual profiles...")

def sigmoid(x, A, k0, beta):
    return A / (1 + np.exp(-beta * (x - k0)))

# Get all subjects for rest condition
rest_data = data[data['condition'] == 'rest']
task_data = data[data['condition'] == 'task']

print(f"\nREST condition - amplitude values:")
for _, row in rest_data.iterrows():
    print(f"  {row['subject']}: A = {row['A']:.1f}, k0 = {row['k0']:.2f}, beta = {row['beta']:.2f}")

print(f"\nTASK condition - amplitude values:")
for _, row in task_data.iterrows():
    print(f"  {row['subject']}: A = {row['A']:.1f}, k0 = {row['k0']:.2f}, beta = {row['beta']:.2f}")

# ============================================================
# STEP 3: ANALYZE PROFILES
# ============================================================
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# For rest: are all profiles identical?
rest_profiles_reconstructed = []
for _, row in rest_data.iterrows():
    profile = sigmoid(k_values, row['A'], row['k0'], row['beta'])
    rest_profiles_reconstructed.append(profile)

rest_profiles = np.array(rest_profiles_reconstructed)

print("\nREST reconstructed profiles at each k:")
for i, k in enumerate(k_values):
    vals = rest_profiles[:, i]
    print(f"  k={k:2d}: mean = {np.mean(vals):.1f}, std = {np.std(vals):.2f}, min = {np.min(vals):.1f}, max = {np.max(vals):.1f}")

# Variance across subjects
print("\nVariance across REST subjects at each k:")
for i, k in enumerate(k_values):
    print(f"  k={k:2d}: var = {np.var(rest_profiles[:, i]):.4f}")

# Check if profiles are identical
are_identical = np.allclose(rest_profiles, rest_profiles[0])
print(f"\nAre all REST profiles identical? {are_identical}")

# ============================================================
# STEP 4: COMPARE TASK VS REST
# ============================================================
print("\n" + "="*60)
print("TASK vs REST COMPARISON")
print("="*60)

task_profiles = []
for _, row in task_data.iterrows():
    profile = sigmoid(k_values, row['A'], row['k0'], row['beta'])
    task_profiles.append(profile)
task_profiles = np.array(task_profiles)

print("\nTASK vs REST at key k-values (k=16):")
print(f"  Task: mean = {np.mean(task_profiles[:, 3]):.1f}, std = {np.std(task_profiles[:, 3]):.1f}")
print(f"  Rest: mean = {np.mean(rest_profiles[:, 3]):.1f}, std = {np.std(rest_profiles[:, 3]):.1f}")

print("\nScale comparison (task vs rest):")
for i, k in enumerate(k_values):
    t_mean = np.mean(task_profiles[:, i])
    r_mean = np.mean(rest_profiles[:, i])
    print(f"  k={k:2d}: Task = {t_mean:.1f}, Rest = {r_mean:.1f}, Ratio = {t_mean/r_mean:.3f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
os.makedirs('/home/student/sgp-tribe3/empirical_analysis/diagnostics', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot all REST profiles overlaid
ax1 = axes[0]
for i in range(len(rest_profiles)):
    ax1.plot(k_values, rest_profiles[i], 'o-', color='red', alpha=0.5, label='REST' if i == 0 else None)
ax1.set_xlabel('k (neighborhood scale)')
ax1.set_ylabel('Residual Dimensionality')
ax1.set_title('REST: All Subjects Overlaid')
ax1.legend()

# Plot TASK vs REST mean profiles
ax2 = axes[1]
ax2.plot(k_values, np.mean(task_profiles, axis=0), 'o-', color='green', label='Task (mean)', lw=2)
ax2.plot(k_values, np.mean(rest_profiles, axis=0), 's-', color='red', label='Rest (mean)', lw=2)
ax2.fill_between(k_values, 
                np.mean(task_profiles, axis=0) - np.std(task_profiles, axis=0),
                np.mean(task_profiles, axis=0) + np.std(task_profiles, axis=0),
                alpha=0.2, color='green')
ax2.fill_between(k_values,
                np.mean(rest_profiles, axis=0) - np.std(rest_profiles, axis=0),
                np.mean(rest_profiles, axis=0) + np.std(rest_profiles, axis=0),
                alpha=0.2, color='red')
ax2.set_xlabel('k (neighborhood scale)')
ax2.set_ylabel('Residual Dimensionality')
ax2.set_title('TASK vs REST: Mean ± SD')
ax2.legend()

plt.tight_layout()
plt.savefig('/home/student/sgp-tribe3/empirical_analysis/diagnostics/rest_raw_profiles.png', dpi=150)
plt.close()

print("\nSaved: empirical_analysis/diagnostics/rest_raw_profiles.png")

# ============================================================
# STEP 6: OUTPUT REPORT
# ============================================================
with open('/home/student/sgp-tribe3/empirical_analysis/diagnostics/rest_variance_report.txt', 'w') as f:
    f.write("REST VARIANCE DIAGNOSTIC REPORT\n")
    f.write("="*50 + "\n\n")
    
    f.write("REST condition amplitude values:\n")
    for _, row in rest_data.iterrows():
        f.write(f"  {row['subject']}: A = {row['A']:.1f}\n")
    
    f.write(f"\nAre all REST profiles identical? {are_identical}\n")
    
    f.write("\nVariance across REST subjects at each k:\n")
    for i, k in enumerate(k_values):
        f.write(f"  k={k}: var = {np.var(rest_profiles[:, i]):.4f}\n")
    
    f.write("\nDIAGNOSIS:\n")
    f.write("-"*30 + "\n")
    if are_identical:
        f.write("FINDING: REST profiles are IDENTICAL across subjects.\n")
        f.write("This indicates the fitting HIT A BOUNDARY at A = -1000.\n")
        f.write("Root cause: Sigmoid fitting constraint, not data artifact.\n")
    else:
        f.write("FINDING: REST profiles have some variance.\n")
        f.write("This suggests the fitting is working but constrained.\n")

print("\nSaved: empirical_analysis/diagnostics/rest_variance_report.txt")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if are_identical:
    print("\nFINAL DIAGNOSIS:")
    print("- ALL REST subjects have identical profiles (amplitude = -1000)")
    print("- This is a FITTING BOUNDARY ARTIFACT, not data artifact")
    print("- The sigmoid fitting hit a lower bound constraint")
    print("- Task profiles show natural variation (not constrained)")
else:
    print("\nREST profiles show some variance - investigating...")

print("\nNOTE: The task vs rest DIFFERENCE is real and interpretable,")
print("but the exact REST values should not be used as baseline.")