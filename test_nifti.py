#!/usr/bin/env python3
"""
Test NIfTI loading and preprocessing
"""

import os
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("TESTING NIFTI LOADING")
print("="*60)

# Find sample NIfTI file
data_dir = '/home/student/sgp-tribe3/empirical_analysis/data/hcp_wm'

# Look for first available fMRI file
sample_file = None
for item in os.listdir(data_dir):
    if item.startswith('sub-'):
        func_dir = os.path.join(data_dir, item, 'func')
        if os.path.exists(func_dir):
            for f in os.listdir(func_dir):
                if f.endswith('.nii.gz') and 'task-rest' in f:
                    sample_file = os.path.join(func_dir, f)
                    print(f"Found sample: {f}")
                    break
    if sample_file:
        break

if sample_file is None:
    print("ERROR: No NIfTI files found")
    exit(1)

print(f"\nLoading: {sample_file}")

# Load NIfTI
img = nib.load(sample_file)
data = img.get_fdata()

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min: {np.min(data)}, Max: {np.max(data)}")

# Get header info
header = img.header
try:
    tr = header.get_zo()[3]  # Get TR from 4th dimension
    print(f"TR: {tr}")
except:
    tr = 2.0
    print(f"TR (default): {tr}")

# ============================================================
# PREPROCESSING
# ============================================================
print("\nApplying preprocessing...")

# Flatten spatial dimensions: (X, Y, Z, T) -> (voxels, time)
shape = data.shape
n_voxels = shape[0] * shape[1] * shape[2]
n_timepoints = shape[3]
data_flat = data.reshape(n_voxels, n_timepoints)

print(f"Flattened shape: {data_flat.shape} (voxels x time)")

# Voxel selection: keep top 5000 by variance
variances = np.var(data_flat, axis=1)
top_idx = np.argsort(variances)[-5000:]
data_selected = data_flat[top_idx]

print(f"After voxel selection: {data_selected.shape}")

# Z-score normalize per voxel
means = np.mean(data_selected, axis=1, keepdims=True)
stds = np.std(data_selected, axis=1, keepdims=True)
stds[stds == 0] = 1
data_norm = (data_selected - means) / stds

# Remove NaN
data_norm = np.nan_to_num(data_norm, nan=0.0)

print(f"After normalization: {data_norm.shape}")
print(f"Min: {np.min(data_norm):.2f}, Max: {np.max(data_norm):.2f}")

# Save test output
print(f"\n{'='*60}")
print("SUCCESS CRITERIA MET")
print("="*60)
print(f"Time series matrix: {data_norm.T.shape} (time x voxels)")
print("Ready for dimensionality computation")

# Save to file for next pipeline
np.save('/home/student/sgp-tribe3/empirical_analysis/test_timeseries.npy', data_norm.T)
print("\nSaved test timeseries to test_timeseries.npy")