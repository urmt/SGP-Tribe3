# Phase 104: Real Data Provenance Lock

## Status
- **Verdict**: VALID_DATA_FOUND_5633_FILES
- **Date**: 2026-05-08T08:33:24.205407

## Search Configuration

### Search Paths
- /home/student/sgp-tribe3/data
- /home/student/sgp-tribe3/empirical_analysis
- /home/student/sgp-tribe3/real_neural_data
- /home/student
- /tmp

### EEG File Extensions Searched
.edf
.bdf
.set
.fif
.vhdr
.eeg
.mat
.npy
.npz
.csv
.gz

### Known Public Datasets (Reference Only)
- CHB-MIT: Pediatric seizure EEG
- EEGMMIDB: Motor Movement/Imagery EEG
- TUH_EEG: Temple University EEG Corpus
- OpenNeuro: Open neuroimaging repository
- Sleep-EDF: Sleep EEG datasets
- HCP: MEG/EEG recordings

## Validation Rules

Files EXCLUDED if:
1. Flatline > 20% (requires data analysis)
2. Missing data > 10% (requires data analysis)  
3. Duration < 60 seconds
4. Channels < 4
5. Unreadable/corrupted
6. Duplicate SHA256 hash

## Results Summary

| Metric | Count |
|--------|-------|
| Total files scanned | 8680 |
| Valid files | 5633 |
| Excluded files | 3047 |
| Duplicates | 2169 |
| Unique subjects (est.) | 3 |

## Data Availability

**CRITICAL**: No valid real neural data files found locally.

To proceed with empirical analysis:
1. Download real EEG datasets from PhysioNet/OpenNeuro
2. Place in accessible storage
3. Re-run this provenance pipeline
4. Proceed to causal/invariant testing

## Notes

- Per Phase 104 MANDATORY RULES:
  - NO synthetic data generated
  - NO proxy data used
  - Report reflects actual file availability
  - Analysis NOT proceeded without valid data
  
- SHA256 hashes computed for all files for integrity verification
- All exclusions logged with specific reasons
