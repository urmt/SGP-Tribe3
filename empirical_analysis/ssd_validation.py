#!/usr/bin/env python3
"""SSD Data Validation - NO ANALYSIS"""
import os
import numpy as np
import pandas as pd

DATA_ROOT = "/home/student/sgp-tribe3/empirical_analysis/data/ds000030_s3/"
SUBJ_CSV = "/home/student/sgp-tribe3/empirical_analysis/ds000030_qc/clean_subjects.csv"

print("=" * 70)
print("MANDATORY SSD DATA VALIDATION")
print("=" * 70)

subjs = pd.read_csv(SUBJ_CSV)['subject'].tolist()

subject_stats = []

for s in subjs:
    events_path = f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv"
    
    if not os.path.exists(events_path):
        print(f"{s}: FILE NOT FOUND")
        continue
    
    ev = pd.read_csv(events_path, sep='\t')
    
    # All STOP trials
    stop_mask = ev['trial_type'] == 'STOP'
    total_stop = stop_mask.sum()
    
    # Trials with non-zero SSD
    ssd_col = 'StopSignalDelay'
    if ssd_col not in ev.columns:
        print(f"{s}: NO SSD COLUMN")
        continue
    
    ssds = ev[ssd_col].values
    non_zero_ssd = (ssds > 0).sum()
    
    ssd_values = ssds[ssds > 0]
    
    if len(ssd_values) > 0:
        ssd_min = ssd_values.min()
        ssd_max = ssd_values.max()
        ssd_mean = ssd_values.mean()
        ssd_std = ssd_values.std()
    else:
        ssd_min = ssd_max = ssd_mean = ssd_std = 0
    
    subject_stats.append({
        'subject': s,
        'total_stop': total_stop,
        'non_zero_ssd': non_zero_ssd,
        'ssd_min': ssd_min,
        'ssd_max': ssd_max,
        'ssd_mean': ssd_mean,
        'ssd_std': ssd_std
    })

df = pd.DataFrame(subject_stats)

print("\n" + "=" * 70)
print("STEP 1: Per-subject STOP trial counts")
print("=" * 70)

for _, row in df.iterrows():
    print(f"{row['subject']}: Total={row['total_stop']:>2}, NonZeroSSD={row['non_zero_ssd']:>2}, "
          f"SSD=[{row['ssd_min']:.3f}, {row['ssd_max']:.3f}, {row['ssd_mean']:.3f}±{row['ssd_std']:.3f}]")

print("\n" + "=" * 70)
print("STEP 2: SSD histogram (aggregated)")
print("=" * 70)

all_ssds = []
for _, row in df.iterrows():
    ev = pd.read_csv(f"{DATA_ROOT}/{row['subject']}/func/{row['subject']}_task-stopsignal_events.tsv", sep='\t')
    ssds = ev['StopSignalDelay'].values
    all_ssds.extend(ssds[ssds > 0])

all_ssds = np.array(all_ssds)
print(f"Total non-zero SSD values: {len(all_ssds)}")
print(f"Min: {all_ssds.min():.3f}s, Max: {all_ssds.max():.3f}s, Mean: {all_ssds.mean():.3f}s")

# Histogram
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
hist, _ = np.histogram(all_ssds, bins=bins)
print(f"\nSSD Distribution:")
for i in range(len(bins)-1):
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}s: {hist[i]} trials")

print("\n" + "=" * 70)
print("STEP 3: Subject eligibility")
print("=" * 70)

eligible = df[df['non_zero_ssd'] >= 30]
excluded = df[df['non_zero_ssd'] < 30]

print(f"Subjects with >= 30 valid SSD trials: {len(eligible)}")
print(f"Subjects excluded: {len(excluded)}")

if len(excluded) > 0:
    print("\nExcluded subjects:")
    for _, row in excluded.iterrows():
        print(f"  {row['subject']}: {row['non_zero_ssd']} trials")

print("\n" + "=" * 70)
print("STEP 4: Bin thresholds for USED subjects")
print("=" * 70)

# Recompute bin thresholds for eligible subjects
used_subjects = eligible['subject'].tolist()

all_ssd_for_used = []
for s in used_subjects:
    ev = pd.read_csv(f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv", sep='\t')
    ssds = ev['StopSignalDelay'].values
    all_ssd_for_used.extend(ssds[ssds > 0])

all_ssd_for_used = np.array(all_ssd_for_used)
overall_median = np.median(all_ssd_for_used)

print(f"Overall median SSD: {overall_median:.3f}s")

# Per-subject binning
bin_stats = []
for s in used_subjects:
    ev = pd.read_csv(f"{DATA_ROOT}/{s}/func/{s}_task-stopsignal_events.tsv", sep='\t')
    ssds = ev['StopSignalDelay'].values[ev['StopSignalDelay'] > 0]
    
    median = np.median(ssds)
    low_count = (ssds <= median).sum()
    high_count = (ssds > median).sum()
    
    bin_stats.append({
        'subject': s,
        'median_ssd': median,
        'low_count': low_count,
        'high_count': high_count
    })

bin_df = pd.DataFrame(bin_stats)

print(f"\n{'Subject':<12} {'Median SSD':>10} {'LOW':>6} {'HIGH':>6}")
print(f"{'-'*36}")
for _, row in bin_df.iterrows():
    print(f"{row['subject']:<12} {row['median_ssd']:>10.3f} {row['low_count']:>6} {row['high_count']:>6}")

print("\n" + "=" * 70)
print("STEP 5: FAIL checks")
print("=" * 70)

failures = []

# Check bin counts
for _, row in bin_df.iterrows():
    if row['low_count'] < 20:
        failures.append(f"{row['subject']}: LOW bin = {row['low_count']} < 20")
    if row['high_count'] < 20:
        failures.append(f"{row['subject']}: HIGH bin = {row['high_count']} < 20")
    if row['median_ssd'] < 0.01:
        failures.append(f"{row['subject']}: Zero median SSD")

# Check SSD variance
for _, row in df.iterrows():
    if row['ssd_std'] < 0.01:  # 10ms
        failures.append(f"{row['subject']}: SSD std = {row['ssd_std']*1000:.1f}ms < 10ms")

if failures:
    print("FAILURES FOUND:")
    for f in failures:
        print(f"  - {f}")
else:
    print("All validation checks PASSED")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)