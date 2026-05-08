import os
import json
import hashlib
import shutil
import requests
from pathlib import Path

ROOT = Path("/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase105_real_eeg_download")

RAW = ROOT / "raw"
VALID = ROOT / "validated"
REJECT = ROOT / "rejected"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

REPORTS.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "EEGMMIDB",
        "url": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
        "expected_ext": ".edf"
    },
    {
        "name": "CHBMIT",
        "url": "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf",
        "expected_ext": ".edf"
    }
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def validate_file(path):
    try:
        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb < 0.01:
            return False, "too_small"

        if path.suffix.lower() != ".edf":
            return False, "wrong_extension"

        with open(path, "rb") as f:
            header = f.read(8)

        if not header.startswith(b"0"):
            return False, "invalid_edf_header"

        return True, "valid"

    except Exception as e:
        return False, str(e)

results = []

for ds in DATASETS:
    out_path = RAW / f"{ds['name']}.edf"

    entry = {
        "dataset": ds["name"],
        "url": ds["url"],
        "downloaded": False,
        "validated": False,
        "reason": None,
        "sha256": None,
        "size_mb": None
    }

    try:
        r = requests.get(ds["url"], stream=True, timeout=120)

        if r.status_code != 200:
            entry["reason"] = f"http_{r.status_code}"
            results.append(entry)
            continue

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        entry["downloaded"] = True

        valid, reason = validate_file(out_path)

        if valid:
            final_path = VALID / out_path.name
            shutil.copy2(out_path, final_path)

            entry["validated"] = True
            entry["reason"] = "valid"
            entry["sha256"] = sha256(final_path)
            entry["size_mb"] = round(final_path.stat().st_size / (1024*1024), 3)
        else:
            reject_path = REJECT / out_path.name
            shutil.move(out_path, reject_path)

            entry["validated"] = False
            entry["reason"] = reason

    except Exception as e:
        entry["reason"] = str(e)

    results.append(entry)

valid_count = sum(r["validated"] for r in results)

summary = {
    "phase": 105,
    "valid_datasets": valid_count,
    "results": results
}

with open(REPORTS / "phase105_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPHASE 105 RESULTS\n")

for r in results:
    print(r)

if valid_count == 0:
    print("\nFINAL VERDICT: NO_REAL_DATA_ACQUIRED")
else:
    print(f"\nFINAL VERDICT: VALID_REAL_DATA_ACQUIRED ({valid_count})")