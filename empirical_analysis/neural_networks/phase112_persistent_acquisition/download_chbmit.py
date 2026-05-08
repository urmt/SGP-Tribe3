import os
import json
import mne
import pandas as pd
import requests
import time

BASE = "empirical_analysis/neural_networks/phase112_persistent_acquisition"
RAWDIR = os.path.join(BASE, "downloaded")
os.makedirs(RAWDIR, exist_ok=True)

MIN_DURATION = 200
WINDOW_SEC = 10

TARGET_FILES = [
    ("chb01", "chb01_04"),
    ("chb01", "chb01_05"),
    ("chb02", "chb02_01"),
    ("chb02", "chb02_16"),
    ("chb03", "chb03_01"),
    ("chb03", "chb03_02"),
    ("chb04", "chb04_01"),
    ("chb04", "chb04_04"),
]

validated = []
excluded = []
download_log = []

existing = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"
if os.path.exists(existing):
    raw = mne.io.read_raw_edf(existing, preload=False, verbose=False)
    dur = raw.n_times / raw.info['sfreq']
    raw.resample(128)
    nw = (raw.get_data()[0].shape[0] - WINDOW_SEC*128) // (WINDOW_SEC*128)
    validated.append({"file": "CHBMIT.edf", "subject": "chb00", "duration": dur, "windows": nw, "source": "existing"})
    print(f"EXISTING: chb00 CHBMIT.edf = {nw} windows")

chb01_03 = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase111_long_duration_real_eeg/downloaded/chb01_03.edf"
if os.path.exists(chb01_03):
    raw = mne.io.read_raw_edf(chb01_03, preload=False, verbose=False)
    dur = raw.n_times / raw.info['sfreq']
    raw.resample(128)
    nw = (raw.get_data()[0].shape[0] - WINDOW_SEC*128) // (WINDOW_SEC*128)
    validated.append({"file": "chb01_03.edf", "subject": "chb01", "duration": dur, "windows": nw, "source": "phase111"})
    print(f"EXISTING: chb01 chb01_03.edf = {nw} windows")

print(f"\nStarting downloads (need >= {5-len(validated)} more unique subjects)...")

def download_with_retry(url, path, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return True
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(2)
    return False

for folder, name in TARGET_FILES:
    subject = folder
    if any(v["subject"] == subject for v in validated):
        print(f"SKIP {name}: subject {subject} already have")
        continue
        
    url = f"https://physionet.org/files/chbmit/1.0.0/{folder}/{name}.edf?download=1"
    fpath = os.path.join(RAWDIR, f"{name}.edf")
    
    print(f"Downloading {name}.edf...")
    
    if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
        print("  Already exists, skipping")
    else:
        if not download_with_retry(url, fpath):
            print(f"  FAILED after 3 retries")
            download_log.append({"file": name, "status": "failed"})
            continue
    
    try:
        raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
        dur = raw.n_times / raw.info['sfreq']
        
        if dur < MIN_DURATION:
            excluded.append({"file": name, "duration": dur, "reason": f"lt_{MIN_DURATION}s"})
            print(f"  EXCLUDED: {dur:.0f}s < {MIN_DURATION}s")
            continue
            
        raw.resample(128)
        nw = (raw.get_data()[0].shape[0] - WINDOW_SEC*128) // (WINDOW_SEC*128)
        
        validated.append({"file": f"{name}.edf", "subject": subject, "duration": dur, "windows": nw, "source": "download"})
        download_log.append({"file": name, "status": "success"})
        print(f"  VALID: subject={subject}, {dur:.0f}s, {nw} windows")
        
    except Exception as e:
        excluded.append({"file": name, "reason": str(e)})
        print(f"  ERROR: {e}")

unique_subjects = len(set(v["subject"] for v in validated))
print(f"\nFINAL: {len(validated)} files, {unique_subjects} unique subjects")

pd.DataFrame(validated).to_csv(os.path.join(BASE, "validated_subjects.csv"), index=False)
pd.DataFrame(excluded).to_csv(os.path.join(BASE, "excluded_subjects.json"), index=False)
with open(os.path.join(BASE, "download_manifest.json"), "w") as f:
    json.dump(download_log, f, indent=2)

verdict = "READY_FOR_STRICT_REAL_LOSO" if unique_subjects >= 5 else "INSUFFICIENT_REAL_SUBJECTS"

print(f"\nPHASE 112 RESULTS")
print(f"unique_valid_subjects: {unique_subjects}")
print("\nVALID SUBJECTS:")
for v in validated:
    print(f"  - {v['subject']}: {v['file']} ({v['duration']:.0f}s, {v['windows']}w)")
print(f"\nVERDICT: {verdict}")