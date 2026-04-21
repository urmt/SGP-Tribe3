#!/usr/bin/env python3
"""
Test ds000030 NIfTI loading after attempted download
"""

import os
import nibabel as nib

data_dir = '/home/student/sgp-tribe3/empirical_analysis/data/hcp_wm'

# Find first available NIfTI
for item in sorted(os.listdir(data_dir)):
    if item.startswith('sub-'):
        func_dir = os.path.join(data_dir, item, 'func')
        if os.path.exists(func_dir):
            for f in os.listdir(func_dir):
                if f.endswith('.nii.gz'):
                    fpath = os.path.join(func_dir, f)
                    size = os.path.getsize(fpath)
                    print(f"Found: {f}")
                    print(f"Size: {size} bytes ({size/1024/1024:.1f} MB)")
                    
                    if size > 1000:
                        try:
                            img = nib.load(fpath)
                            data = img.get_fdata()
                            print(f"Shape: {data.shape}")
                            print("SUCCESS: Data loaded!")
                        except Exception as e:
                            print(f"Error loading: {e}")
                    else:
                        print("Too small - likely pointer")
                    break
        break

# Check if any real files exist
real_count = 0
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.nii.gz'):
            fpath = os.path.join(root, f)
            if os.path.getsize(fpath) > 100000:
                real_count += 1

print(f"\nReal NIfTI files (>100KB): {real_count}")