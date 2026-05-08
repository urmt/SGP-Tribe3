import os
import json
import hashlib
import mne
import pandas as pd
import requests

BASE = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase111_long_duration_real_eeg"
RAWDIR = os.path.join(BASE, "downloaded")
os.makedirs(RAWDIR, exist_ok=True)

MIN_DURATION = 200
MIN_FS = 64
WINDOW_SEC = 10

validated = []
excluded = []

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_edf(path, source):
    fname = os.path.basename(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        nchan = raw.info['nchan']
        duration = raw.n_times / raw.info['sfreq']
        sfreq = raw.info['sfreq']

        if duration < MIN_DURATION:
            excluded.append({'file': fname, 'duration': duration, 'reason': f'lt_{MIN_DURATION}s', 'source': source})
            return None
        if sfreq < MIN_FS:
            excluded.append({'file': fname, 'reason': 'sfreq_lt_64hz', 'source': source})
            return None

        raw.resample(128)
        data = raw.get_data()
        sig = data[0]

        n_windows = (len(sig) - WINDOW_SEC*128) // (WINDOW_SEC*128)

        return {'file': fname, 'duration': duration, 'nchan': nchan, 'sfreq': sfreq, 'windows': n_windows, 'source': source, 'sha256': sha256_file(path)}

    except Exception as e:
        excluded.append({'file': fname, 'reason': str(e), 'source': source})
        return None

print("="*60)
print("PHASE 111 - LONG-DURATION REAL EEG ACQUISITION")
print("="*60)

existing = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"
if os.path.exists(existing):
    v = validate_edf(existing, "existing")
    if v:
        validated.append(v)
        print(f"EXISTING: CHBMIT.edf - {v['duration']:.0f}s, {v['windows']} windows - VALID")

print(f"\nStarting downloads (need >= {5-len(validated)} more)...")

CHBMIT_URLS = [
    ("https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf?download=1", "chb01_03"),
    ("https://physionet.org/files/chbmit/1.0.0/chb01/chb01_04.edf?download=1", "chb01_04"),
    ("https://physionet.org/files/chbmit/1.0.0/chb01/chb01_15.edf?download=1", "chb01_15"),
    ("https://physionet.org/files/chbmit/1.0.0/chb02/chb02_01.edf?download=1", "chb02_01"),
    ("https://physionet.org/files/chbmit/1.0.0/chb02/chb02_16.edf?download=1", "chb02_16"),
    ("https://physionet.org/files/chbmit/1.0.0/chb03/chb03_01.edf?download=1", "chb01_01"),
    ("https://physionet.org/files/chbmit/1.0.0/chb03/chb03_02.edf?download=1", "chb03_02"),
]

for url, name in CHBMIT_URLS:
    if len(validated) >= 5:
        break
    print(f"Downloading {name}...")
    try:
        r = requests.get(url, timeout=90)
        if r.status_code == 200:
            fpath = os.path.join(RAWDIR, f"{name}.edf")
            with open(fpath, 'wb') as f:
                f.write(r.content)
            v = validate_edf(fpath, "physionet_chbmit")
            if v:
                validated.append(v)
                print(f"  VALID: {v['duration']:.0f}s, {v['windows']} windows")
            else:
                print(f"  EXCLUDED: < 200s or failed")
        else:
            print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nFINAL: {len(validated)} valid subjects")

pd.DataFrame(validated).to_csv(os.path.join(BASE, "validated_subjects.csv"), index=False)
pd.DataFrame(excluded).to_csv(os.path.join(BASE, "excluded_subjects.csv"), index=False)

dataset_inventory = {
    'datasets': ['existing', 'physionet_chbmit'],
    'total_valid': len(validated),
    'total_excluded': len(excluded)
}
with open(os.path.join(BASE, "dataset_inventory.json"), 'w') as f:
    json.dump(dataset_inventory, f, indent=2)

verdict = "READY_FOR_REAL_LOSO" if len(validated) >= 5 else "INSUFFICIENT_LONG_RECORDINGS"

print("\nPHASE 111 RESULTS")
print(f"valid_subjects: {len(validated)}")
print("\nVALID FILES:")
for v in validated:
    print(f"  - {v['file']} ({v['duration']:.0f}s, {v['windows']} windows)")
print(f"\nVERDICT: {verdict}")