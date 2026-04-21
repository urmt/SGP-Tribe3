#!/usr/bin/env python3
"""Fast QC for ds000030 - just check 25 subjects"""

import os
import numpy as np
import pandas as pd
import nibabel as nib

data_dir = '/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3'
required_tasks = ['rest', 'bart', 'scap', 'stopsignal', 'taskswitch']

# Get subjects
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])[:50]

results = []
for subj in subjects:
    func_dir = os.path.join(data_dir, subj, 'func')
    if not os.path.exists(func_dir):
        continue
    
    valid_count = 0
    for task in required_tasks:
        nii_file = None
        for f in os.listdir(func_dir):
            if f'task-{task}' in f and f.endswith('.nii.gz'):
                nii_file = os.path.join(func_dir, f)
                break
        
        if nii_file and os.path.getsize(nii_file) > 10_000_000:
            try:
                img = nib.load(nii_file)
                if len(img.get_fdata().shape) == 4:
                    valid_count += 1
            except:
                pass
    
    results.append({'subject': subj, 'valid_tasks': valid_count})

df = pd.DataFrame(results)
clean = df[df['valid_tasks'] == 5]['subject'].tolist()

print(f"Subjects checked: {len(df)}")
print(f"Clean subjects (5/5 tasks): {len(clean)}")

# Save
pd.DataFrame({'subject': clean}).to_csv('/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv', index=False)

with open('/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/qc_summary.txt', 'w') as f:
    f.write(f"Clean subjects: {len(clean)}\n")

print(f"Saved: {len(clean)} clean subjects")