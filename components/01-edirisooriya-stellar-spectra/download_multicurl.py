"""
High-Speed Parallel Multi-Curl Downloader for SDSS APOGEE DR17 spectra.
Uses Windows native curl.exe subprocess workers in parallel.
Correctly extracts stellar parameters from HDU 9/10 BinTableHDU.
"""

import os, sys, re, time, subprocess, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

RAW_DIR = Path("raw_spectra")
RAW_DIR.mkdir(exist_ok=True)
MANIFEST = "dataset_manifest_labeled.csv"
LOG_FILE = "download_log.txt"
TARGET = 4000
CONCURRENCY = 6  # 6 simultaneous native curl streams
BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

FIELDS = [
    ("apo25m", "M15"),
    ("apo25m", "120+12"),
    ("apo25m", "M3"),
    ("apo25m", "180+04"),
    ("apo25m", "M67"),
    ("apo25m", "M92"),
    ("apo25m", "M13"),
    ("apo25m", "000+60"),
    ("apo25m", "000+02"),
]

def teff_to_label(t):
    if t >= 10000: return "B"
    elif t >= 7500: return "A"
    elif t >= 6000: return "F"
    elif t >= 5200: return "G"
    else: return "K"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

existing = set(f.name for f in RAW_DIR.glob("*.fits"))
log("=" * 65)
log("=== MULTI-CURL ACCELERATED ENGINE STARTED ===")
log(f"Current local spectra count: {len(existing)}")
log(f"Parallel native curl workers: {CONCURRENCY}")
log("=" * 65)

# Collect download queue across fields
tasks_to_download = []
for tel, field in FIELDS:
    dir_url = f"{BASE}/{tel}/{field}/"
    try:
        req = urllib.request.Request(dir_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            names = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', html)
            new_names = [n for n in names if n not in existing]
            for n in new_names:
                tasks_to_download.append((f"{dir_url}{n}", n))
            log(f"Field {tel}/{field}: {len(new_names)} queued for download")
    except Exception as e:
        log(f"Listing failed for {tel}/{field}: {e}")

log(f"Total candidate spectra in queue: {len(tasks_to_download)}")

manifest_lock = threading.Lock()
downloaded = 0
failed = 0
results = []
start_time = time.time()

def extract_stellar_params(filepath):
    try:
        with fits.open(filepath) as hdul:
            for idx in [10, 9]:
                if len(hdul) > idx and hdul[idx].data is not None:
                    d = hdul[idx].data
                    if 'teff' in d.names:
                        arr = d['teff']
                        teff_val = arr[-1] if len(arr) > 0 else 0
                        if not np.isnan(teff_val) and teff_val > 0:
                            logg = d['logg'][-1] if 'logg' in d.names else 0
                            feh = d['feh'][-1] if 'feh' in d.names else 0
                            return float(teff_val), float(logg), float(feh)
            # Fallback to primary header if present
            hdr = hdul[0].header
            t = float(hdr.get("TEFF", 0))
            if t > 0:
                return t, float(hdr.get("LOGG", 0)), float(hdr.get("FEH", 0))
    except Exception:
        pass
    return None

def fetch_single_spectrum(item):
    global downloaded, failed
    url, fname = item
    dest = RAW_DIR / fname

    if dest.exists() and dest.stat().st_size > 50000:
        return None

    cmd = [
        "curl.exe",
        "-s",
        "--connect-timeout", "20",
        "--max-time", "180",
        "--retry", "2",
        "--retry-delay", "3",
        "-o", str(dest),
        url
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=190)
        if proc.returncode != 0 or not dest.exists() or dest.stat().st_size < 50000:
            dest.unlink(missing_ok=True)
            with manifest_lock:
                failed += 1
            return None
    except Exception:
        dest.unlink(missing_ok=True)
        with manifest_lock:
            failed += 1
        return None

    # Verify and extract params
    params = extract_stellar_params(dest)
    if params is None:
        dest.unlink(missing_ok=True)
        with manifest_lock:
            failed += 1
        return None

    teff, logg, feh = params
    label = teff_to_label(teff)
    dt = time.time() - t0
    size_mb = dest.stat().st_size / (1024 * 1024)

    record = {
        "file_name": fname,
        "teff": teff,
        "logg": logg,
        "feh": feh,
        "label": label
    }

    with manifest_lock:
        downloaded += 1
        existing.add(fname)
        results.append(record)
        elapsed = time.time() - start_time
        speed = downloaded / (elapsed / 60) if elapsed > 0 else 0
        total_now = len(existing)
        log(f"  ✓ [{downloaded}/{TARGET}] {label} (TEFF={teff:.0f}K) | {size_mb:.1f}MB in {dt:.1f}s | Speed: {speed:.1f} spectra/min | Total: {total_now} | {fname}")

        # Checkpoint every 5 spectra
        if downloaded % 5 == 0:
            if Path(MANIFEST).exists():
                old = pd.read_csv(MANIFEST)
                combo = pd.concat([old, pd.DataFrame(results)], ignore_index=True).drop_duplicates("file_name")
            else:
                combo = pd.DataFrame(results)
            combo.to_csv(MANIFEST, index=False)

    return record

# Run concurrent native curl pool
with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = []
    for item in tasks_to_download:
        if downloaded >= TARGET:
            break
        futures.append(executor.submit(fetch_single_spectrum, item))

    for f in as_completed(futures):
        if downloaded >= TARGET:
            for fut in futures:
                fut.cancel()
            break

log("=" * 65)
log(f"Multi-curl finished: {downloaded} spectra saved in {(time.time()-start_time)/60:.1f} mins.")
if results:
    old = pd.read_csv(MANIFEST)
    combo = pd.concat([old, pd.DataFrame(results)], ignore_index=True).drop_duplicates("file_name")
    combo.to_csv(MANIFEST, index=False)
    log(f"Final Manifest updated: {len(combo)} total spectra.")
