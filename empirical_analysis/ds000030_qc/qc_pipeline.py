#!/usr/bin/env python3
"""
SFH-SGP_DATA_QC_DS000030
Validate dataset integrity and select clean subjects
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("SFH-SGP_DATA_QC_DS000030")
print("="*60)

# ============================================================
# STEP 1: SCAN DATASET
# ============================================================
print("\nSTEP 1: Scanning dataset...")

data_dir = '/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3'
required_tasks = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']

# Find all subjects
all_subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
print(f"Found {len(all_subjects)} total subjects")

# ============================================================
# STEP 2: FILE VALIDATION
# ============================================================
print("\nSTEP 2: File validation...")

results = []

for subj in all_subjects[:100]:  # Start with first 100 for speed
    func_dir = os.path.join(data_dir, subj, 'func')
    if not os.path.exists(func_dir):
        continue
    
    subj_valid = True
    task_results = {}
    
    for task in required_tasks:
        # Find file
        nii_file = None
        for f in os.listdir(func_dir):
            if f'task-{task}' in f and f.endswith('.nii.gz'):
                nii_file = os.path.join(func_dir, f)
                break
        
        if nii_file is None:
            task_results[task] = 'missing'
            subj_valid = False
            continue
        
        # Check file size
        size = os.path.getsize(nii_file)
        if size < 10_000_000:  # < 10MB
            task_results[task] = f'too_small_{size}'
            subj_valid = False
            continue
        
        # Try to load
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            
            # Check shape
            if len(data.shape) != 4:
                task_results[task] = 'not_4d'
                subj_valid = False
                continue
            
            # Check timepoints
            T = data.shape[3]
            if T < 100:
                task_results[task] = f'too_short_{T}'
                subj_valid = False
                continue
            
            task_results[task] = 'valid'
            
        except Exception as e:
            task_results[task] = f'error_{str(e)[:20]}'
            subj_valid = False
    
    results.append({
        'subject': subj,
        'valid': subj_valid,
        **task_results
    })

results_df = pd.DataFrame(results)
print(f"Validated {len(results_df)} subjects")

# ============================================================
# STEP 3: SUBJECT FILTERING
# ============================================================
print("\nSTEP 3: Subject filtering...")

# Select subjects with ALL tasks valid
valid_subjects = results_df[results_df['valid'] == True]['subject'].tolist()
print(f"Subjects with ALL 5 tasks valid: {len(valid_subjects)}")

# Check if we have enough
if len(valid_subjects) < 20:
    print(f"\nWARNING: Only {len(valid_subjects)} valid subjects (< 20 minimum)")
    # Keep going anyway

# ============================================================
# STEP 4: SIGNAL VALIDATION
# ============================================================
print("\nSTEP 4: Signal validation...")

signal_results = []
max_subjects = min(30, len(valid_subjects))

for subj in valid_subjects[:max_subjects]:
    func_dir = os.path.join(data_dir, subj, 'func')
    
    for task in required_tasks:
        # Find file
        nii_file = None
        for f in os.listdir(func_dir):
            if f'task-{task}' in f and f.endswith('.nii.gz'):
                nii_file = os.path.join(func_dir, f)
                break
        
        if nii_file is None:
            continue
        
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            
            # Flatten to (voxels, time)
            data_flat = data.reshape(-1, data.shape[-1])
            
            # Select top 3000 voxels by variance
            variances = np.var(data_flat, axis=1)
            top_idx = np.argsort(variances)[-3000:]
            data_sel = data_flat[top_idx]
            
            # Z-score normalize
            means = np.mean(data_sel, axis=1, keepdims=True)
            stds = np.std(data_sel, axis=1, keepdims=True)
            stds[stds == 0] = 1
            data_norm = (data_sel - means) / stds
            
            # Check for NaN
            has_nan = np.any(np.isnan(data_norm))
            
            # Check variance
            var_zero = np.sum(np.var(data_norm, axis=1) == 0)
            
            # Check range
            data_range = np.max(data_norm) - np.min(data_norm)
            
            signal_valid = not has_nan and var_zero < 10 and data_range < 50
            
            signal_results.append({
                'subject': subj,
                'task': task,
                'has_nan': has_nan,
                'zero_var_voxels': var_zero,
                'range': data_range,
                'signal_valid': signal_valid
            })
            
        except Exception as e:
            signal_results.append({
                'subject': subj,
                'task': task,
                'error': str(e)[:50],
                'signal_valid': False
            })

signal_df = pd.DataFrame(signal_results)
print(f"Signal validated: {len(signal_df)} subject-task combinations")

# ============================================================
# STEP 5: OUTPUT
# ============================================================
print("\nSTEP 5: Output...")

output_dir = '/home/student/sgp-tribe3/empirical_analysis/ds000030_qc'
os.makedirs(output_dir, exist_ok=True)

# Save clean subjects
clean_subj = signal_df[signal_df['signal_valid'] == True].groupby('subject').size()
clean_subj = clean_subj[clean_subj == 5].index.tolist()

pd.DataFrame({'subject': clean_subj}).to_csv(f'{output_dir}/clean_subjects.csv', index=False)
print(f"Clean subjects saved: {len(clean_subj)}")

# Save QC summary
with open(f'{output_dir}/qc_summary.txt', 'w') as f:
    f.write("SFH-SGP_DATA_QC_DS000030\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total subjects scanned: {len(all_subjects)}\n")
    f.write(f"Subjects with 5 valid tasks: {len(valid_subjects)}\n")
    f.write(f"Clean subjects after signal QC: {len(clean_subj)}\n\n")
    
    # Rejection reasons
    rejected = results_df[results_df['valid'] == False]
    f.write("Rejection reasons (file validation):\n")
    for col in required_tasks:
        failures = rejected[rejected[col] != 'valid'][col].value_counts()
        for reason, count in failures.items():
            f.write(f"  {col}: {reason} = {count}\n")
    
    f.write("\nSignal validation failures:\n")
    signal_fail = signal_df[signal_df['signal_valid'] == False]
    f.write(f"  Total: {len(signal_fail)}\n")

print("\n" + "="*60)
print("QC COMPLETE")
print("="*60)
print(f"Clean subjects available: {len(clean_subj)}")

if len(clean_subj) >= 25:
    print("✓ SUFFICIENT DATA FOR ANALYSIS")
elif len(clean_subj) >= 20:
    print("⚠ WARNING: Below target but usable")
else:
    print("✗ INSUFFICIENT DATA")