#!/usr/bin/env python3
"""
Test NIfTI loading from ds000114
"""

import os
import numpy as np
import nibabel as nib

print("="*60)
print("TESTING NIFTI LOADING (ds000114)")
print("="*60)

# Find sample NIfTI
data_dir = '/home/student/sgp-tribe3/empirical_analysis/data/ds000114/sub-10/ses-test/func'

sample_file = None
for f in os.listdir(data_dir):
    if f.endswith('.nii.gz'):
        sample_file = os.path.join(data_dir, f)
        print(f"Found: {f}")
        break

if sample_file is None:
    print("ERROR: No files found")
    exit(1)

# Load
print(f"\nLoading: {sample_file}")
img = nib.load(sample_file)
data = img.get_fdata()

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Volume: {np.min(data):.0f} to {np.max(data):.0f}")

# Preprocess
# Flatten: (X,Y,Z,T) -> (voxels, time)
shape = data.shape
n_voxels = shape[0] * shape[1] * shape[2]
n_time = shape[3]
data_flat = data.reshape(n_voxels, n_time)
print(f"Flattened: {data_flat.shape}")

# Voxel selection (keep top 3000 by variance)
variances = np.var(data_flat, axis=1)
top_idx = np.argsort(variances)[-3000:]
data_sel = data_flat[top_idx]
print(f"After selection: {data_sel.shape}")

# Z-score normalize
means = np.mean(data_sel, axis=1, keepdims=True)
stds = np.std(data_sel, axis=1, keepdims=True)
stds[stds == 0] = 1
data_norm = (data_sel - means) / stds
data_norm = np.nan_to_num(data_norm, nan=0.0)

# Final
print(f"Normalized: {data_norm.shape}")
print(f"Range: {np.min(data_norm):.2f} to {np.max(data_norm):.2f}")

print(f"\n{'='*60}")
print("SUCCESS: Data loaded and preprocessed")
print("="*60)
print(f"Time series: {data_norm.T.shape} (time x voxels)")
print("Ready for analysis")