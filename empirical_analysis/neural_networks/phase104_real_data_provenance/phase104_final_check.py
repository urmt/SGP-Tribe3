"""
PHASE 104 - REAL DATA PROVENANCE LOCK (FINAL CORRECTION)
"""

import os
import json

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase104_real_data_provenance"

with open(os.path.join(BASE_DIR, "dataset_manifest.json"), "r") as f:
    manifest = json.load(f)

print("="*70)
print("PHASE 104: REAL EEG VERIFICATION (FINAL)")
print("="*70)

# Patterns for non-EEG data
NON_NEURAL_PATTERNS = [
    "embedding", "embed", "model", "weight", "checkpoint",
    "projection", "latent", "feature", "mnist", "fashion",
    "ollama", "stimulus", "structured", "token", "vocab",
    "train-images", "train-labels", "t10k", "idx", "ubyte",
    "classifier", "network", "vae", "gan", "diffusion",
    "text", "language", "bert", "gpt", "llama",
    "results", "metrics", "parameters", "auc", "accuracy",
    "stability", "regime", "cohort", "subject", "task",
    "simulation", "synthetic", "phase", "analysis", "figure"
]

# Paths containing these are synthetic data
SYNTHETIC_PATH_PATTERNS = ["empirical_analysis", "simulation", "generated"]

filtered_records = []
excluded_count = 0

for rec in manifest["files"]:
    exclude = False
    
    # Exclude .csv files - results, not raw data
    if rec["filename"].endswith('.csv'):
        exclude = True
    
    # Exclude non-neural patterns in filename
    elif any(p in rec["filename"].lower() for p in NON_NEURAL_PATTERNS):
        exclude = True
    
    # Exclude synthetic data paths
    elif any(p in rec["path"] for p in SYNTHETIC_PATH_PATTERNS):
        exclude = True
    
    if exclude and rec["valid"]:
        rec["valid"] = False
        rec["exclusion_reason"] = "NOT_NEURAL_DATA"
        excluded_count += 1
    
    filtered_records.append(rec)

valid_files = [r for r in filtered_records if r["valid"]]
excluded = [r for r in filtered_records if not r["valid"]]

print(f"\n[CORRECTION]")
print(f"  Additional files excluded: {excluded_count}")
print(f"  Valid files remaining: {len(valid_files)}")

print(f"\n[Sample remaining valid files]")
for r in valid_files[:5]:
    print(f"  {r['filename']}")

print(f"\n[VERDICT]")
if len(valid_files) == 0:
    verdict = "NO_REAL_EEG_DATA_AVAILABLE"
    print(f"  **{verdict}**")
else:
    # Check if any are actually EEG recordings
    eeg_patterns = ["chb", "eegmmidb", "tuh", "physio", "sleep", "edf", "meg"]
    has_eeg = any(any(p in r["filename"].lower() for p in eeg_patterns) for r in valid_files)
    
    if has_eeg:
        verdict = "VALID_EEG_DATA_FOUND"
        print(f"  {verdict}")
    else:
        verdict = "NO_REAL_EEG_DATA_AVAILABLE"
        print(f"  **verdict** (files are not actual EEG recordings)")

# Update files
manifest["files"] = filtered_records
manifest["correction"] = {
    "non_neural_excluded": excluded_count,
    "verdict": verdict
}

with open(os.path.join(BASE_DIR, "dataset_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

report = {"verdict": verdict, "valid_files": len(valid_files), "excluded": len(excluded)}
with open(os.path.join(BASE_DIR, "provenance_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*70)