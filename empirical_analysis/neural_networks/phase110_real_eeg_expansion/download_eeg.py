import os
import json
import hashlib
import subprocess
import mne
import pandas as pd
import requests
import zipfile
import shutil

BASE = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase110_real_eeg_expansion"
RAWDIR = os.path.join(BASE, "downloaded")
os.makedirs(RAWDIR, exist_ok=True)

MIN_DURATION = 300
MIN_FS = 64
MIN_WINDOWS = 20
WINDOW_SEC = 10
SFREQ_TARGET = 128

validated = []
excluded = []
download_log = []

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_edf(path):
    fname = os.path.basename(path)
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        nchan = raw.info['nchan']
        duration = raw.n_times / raw.info['sfreq']
        sfreq = raw.info['sfreq']

        if duration < MIN_DURATION:
            excluded.append({'file': fname, 'reason': f'duration_{duration:.1f}s_lt_{MIN_DURATION}s', 'source': 'download'})
            return None
        if sfreq < MIN_FS:
            excluded.append({'file': fname, 'reason': f'sfreq_{sfreq}_lt_{MIN_FS}hz', 'source': 'download'})
            return None
        if nchan < 1:
            excluded.append({'file': fname, 'reason': 'no_channels', 'source': 'download'})
            return None

        raw.resample(SFREQ_TARGET)
        data = raw.get_data()
        sig = data[0]

        win = WINDOW_SEC * SFREQ_TARGET
        n_windows = (len(sig) - win) // win

        if n_windows < MIN_WINDOWS:
            excluded.append({'file': fname, 'reason': f'only_{n_windows}_windows_lt_{MIN_WINDOWS}', 'source': 'download'})
            return None

        return {'file': fname, 'duration': duration, 'nchan': nchan, 'sfreq': sfreq, 'windows': n_windows, 'sha256': sha256_file(path)}

    except Exception as e:
        excluded.append({'file': fname, 'reason': str(e), 'source': 'download'})
        return None

print("="*60)
print("PHASE 110 - REAL EEG EXPANSION")
print("="*60)

existing = [
    "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase105_real_eeg_download/raw/EEGMMIDB.edf",
    "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"
]

for path in existing:
    if os.path.exists(path):
        v = validate_edf(path)
        if v:
            v['source'] = 'existing'
            validated.append(v)
            print(f"EXISTING: {os.path.basename(path)} - {v['windows']} windows - VALID")
        else:
            print(f"EXISTING: {os.path.basename(path)} - EXCLUDED")

print(f"\nAfter existing: {len(validated)} valid subjects")

if len(validated) >= 5:
    print("Already have 5+ subjects!")
else:
    print("\nDownloading more EEGMMIDB subjects...")
    EEGMMIDB_URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf?download=1"
    
    try:
        r = requests.get(EEGMMIDB_URL, timeout=60)
        if r.status_code == 200:
            fpath = os.path.join(RAWDIR, "S001R04.edf")
            with open(fpath, 'wb') as f:
                f.write(r.content)
            download_log.append({'url': EEGMMIDB_URL, 'status': 'success', 'file': 'S001R04.edf'})
            v = validate_edf(fpath)
            if v:
                v['source'] = 'physionet_eegmmidb'
                validated.append(v)
                print(f"DOWNLOADED: S001R04.edf - {v['windows']} windows - VALID")
    except Exception as e:
        download_log.append({'url': EEGMMIDB_URL, 'status': 'failed', 'error': str(e)})
        print(f"Failed: {e}")

    print(f"Total valid: {len(validated)}")

    EEGMMIDB_URLS = [
        f"https://physionet.org/files/eegmmidb/1.0.0/S002/S002R04.edf?download=1",
        f"https://physionet.org/files/eegmmidb/1.0.0/S003/S003R04.edf?download=1",
        f"https://physionet.org/files/eegmmidb/1.0.0/S004/S004R04.edf?download=1",
    ]
    
    for url in EEGMMIDB_URLS:
        if len(validated) >= 5:
            break
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                fname = url.split('/')[-1].split('?')[0]
                fpath = os.path.join(RAWDIR, fname)
                with open(fpath, 'wb') as f:
                    f.write(r.content)
                download_log.append({'url': url, 'status': 'success', 'file': fname})
                v = validate_edf(fpath)
                if v:
                    v['source'] = 'physionet_eegmmidb'
                    validated.append(v)
                    print(f"DOWNLOADED: {fname} - {v['windows']} windows - VALID")
        except Exception as e:
            download_log.append({'url': url, 'status': 'failed', 'error': str(e)})
            print(f"Failed {url.split('/')[-1].split('?')[0]}: {e}")

print(f"\nFINAL: {len(validated)} valid subjects")

pd.DataFrame(validated).to_csv(os.path.join(BASE, "validated_subjects.csv"), index=False)
pd.DataFrame(excluded).to_csv(os.path.join(BASE, "excluded_subjects.csv"), index=False)

with open(os.path.join(BASE, "download_log.json"), 'w') as f:
    json.dump(download_log, f, indent=2)

dataset_inventory = {
    'datasets_attempted': ['physionet_eegmmidb'],
    'total_validated': len(validated),
    'total_excluded': len(excluded)
}
with open(os.path.join(BASE, "dataset_inventory.json"), 'w') as f:
    json.dump(dataset_inventory, f, indent=2)

verdict = "READY_FOR_STRICT_LOSO" if len(validated) >= 5 else "INSUFFICIENT_PUBLIC_DATA"

print("\nPHASE 110 RESULTS")
print(f"valid_subjects: {len(validated)}")
print(f"excluded_subjects: {len(excluded)}")
print(f"datasets_acquired: {len(set(v.get('source','unknown') for v in validated))}")
print("\nVALID SUBJECT FILES:")
for v in validated:
    print(f"  - {v['file']} ({v['windows']} windows, source: {v.get('source','unknown')})")
print(f"\nVERDICT: {verdict}")