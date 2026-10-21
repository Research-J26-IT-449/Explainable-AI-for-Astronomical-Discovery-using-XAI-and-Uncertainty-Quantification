"""
High-throughput, non-blocking downloader using urllib with hard timeouts.
Avoids requests connection pool hangs on Windows.
"""

import os, sys, re, time, urllib.request, socket
from datetime import datetime
from pathlib import Path
import pandas as pd
from astropy.io import fits

# Hard socket timeout so Windows sockets never freeze indefinitely
socket.setdefaulttimeout(35.0)

RAW_DIR = Path("raw_spectra")
RAW_DIR.mkdir(exist_ok=True)
MANIFEST = "dataset_manifest_labeled.csv"
LOG_FILE = "download_log.txt"
TARGET = 4000
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
log("=== HIGH-THROUGHPUT RELIABLE DOWNLOADER STARTED ===")
log(f"Current local dataset count: {len(existing)} files")
log(f"Targeting: +{TARGET} spectra")
log("=" * 65)

downloaded = 0
failed = 0
results = []
start_time = time.time()

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) APOGEE-Astrophysics/2.0')]
urllib.request.install_opener(opener)

for tel, field in FIELDS:
    if downloaded >= TARGET:
        break

    dir_url = f"{BASE}/{tel}/{field}/"
    log(f"\n--- Reading Field: {tel}/{field} ---")

    names = []
    try:
        req = urllib.request.Request(dir_url)
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            names = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', html)
    except Exception as e:
        log(f"  Field listing failed for {tel}/{field}: {e}")
        continue

    new_names = [n for n in names if n not in existing]
    log(f"  Available in {field}: {len(names)} total | {len(new_names)} new to download")

    for fname in new_names:
        if downloaded >= TARGET:
            break

        dest = RAW_DIR / fname
        file_url = f"{dir_url}{fname}"
        t0 = time.time()
        file_ok = False

        for attempt in range(3):
            try:
                urllib.request.urlretrieve(file_url, dest)
                if dest.exists() and dest.stat().st_size > 50000:
                    file_ok = True
                    break
                else:
                    dest.unlink(missing_ok=True)
                    time.sleep(2)
            except Exception as e:
                dest.unlink(missing_ok=True)
                time.sleep(3)

        if not file_ok:
            dest.unlink(missing_ok=True)
            failed += 1
            continue

        # Extract TEFF metadata from FITS
        try:
            with fits.open(dest) as hdul:
                hdr = hdul[0].header
                teff = float(hdr.get("TEFF", 0))
                logg = float(hdr.get("LOGG", 0))
                feh = float(hdr.get("FEH", 0))
            if teff <= 0:
                dest.unlink(missing_ok=True)
                failed += 1
                continue
        except Exception:
            dest.unlink(missing_ok=True)
            failed += 1
            continue

        label = teff_to_label(teff)
        dt = time.time() - t0
        size_mb = dest.stat().st_size / (1024 * 1024)
        elapsed = time.time() - start_time
        rate = downloaded / elapsed * 60 if elapsed > 0 else 0

        results.append({
            "file_name": fname,
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "label": label
        })
        existing.add(fname)
        downloaded += 1

        log(f"  ✓ [{downloaded}/{TARGET}] {label} (TEFF={teff:.0f}K) | {size_mb:.1f}MB in {dt:.1f}s | Total: {len(existing)} | {fname}")

        # Checkpoint every 5 files
        if downloaded % 5 == 0 and results:
            if Path(MANIFEST).exists():
                old = pd.read_csv(MANIFEST)
                new = pd.DataFrame(results)
                combo = pd.concat([old, new], ignore_index=True).drop_duplicates("file_name")
            else:
                combo = pd.DataFrame(results)
            combo.to_csv(MANIFEST, index=False)
            log(f"  [Checkpoint] Manifest saved ({len(combo)} total registered spectra)")

log("=" * 65)
log(f"Finished session: {downloaded} new files downloaded, {failed} failed in {(time.time()-start_time)/60:.1f} min.")
