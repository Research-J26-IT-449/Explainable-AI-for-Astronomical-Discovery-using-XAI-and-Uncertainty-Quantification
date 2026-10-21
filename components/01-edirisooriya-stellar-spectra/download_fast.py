"""
Simple, reliable sequential downloader.
Lists M15 field (1,178 new files available), downloads one at a time.
Prints progress every file. No threads, no queues, no deadlocks.
"""
import re, time, requests, sys
import pandas as pd
from pathlib import Path
from astropy.io import fits

RAW_DIR  = Path("raw_spectra")
MANIFEST = "dataset_manifest_labeled.csv"
TARGET   = 4000
BASE     = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

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

existing = set(f.name for f in RAW_DIR.glob("*.fits"))
print(f"Start: {len(existing)} files on disk. Target: +{TARGET} more.", flush=True)

s = requests.Session()
s.headers["User-Agent"] = "APOGEE-research/1.0"

downloaded = 0
failed = 0
results = []
start = time.time()

for tel, field in FIELDS:
    if downloaded >= TARGET:
        break

    # Get directory listing
    dir_url = f"{BASE}/{tel}/{field}/"
    print(f"\n[{tel}/{field}] Getting file list...", flush=True)
    try:
        r = s.get(dir_url, timeout=30)
        names = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', r.text)
    except Exception as e:
        print(f"  FAILED to list: {e}", flush=True)
        continue

    new_names = [n for n in names if n not in existing]
    print(f"  {len(names)} total, {len(new_names)} new to download", flush=True)

    for fname in new_names:
        if downloaded >= TARGET:
            break

        dest = RAW_DIR / fname
        file_url = f"{dir_url}{fname}"

        try:
            r = s.get(file_url, timeout=90, stream=True)
            if r.status_code != 200:
                failed += 1
                print(f"  HTTP {r.status_code} for {fname}", flush=True)
                continue
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        except Exception as e:
            failed += 1
            dest.unlink(missing_ok=True)
            print(f"  ERROR {fname}: {e}", flush=True)
            continue

        # Read metadata
        try:
            with fits.open(dest) as hdul:
                hdr = hdul[0].header
                teff = float(hdr.get("TEFF", 0))
                logg = float(hdr.get("LOGG", 0))
                feh  = float(hdr.get("FEH", 0))
            if teff <= 0:
                dest.unlink(missing_ok=True)
                failed += 1
                continue
        except Exception:
            dest.unlink(missing_ok=True)
            failed += 1
            continue

        label = teff_to_label(teff)
        results.append({"file_name": fname, "teff": teff, "logg": logg, "feh": feh, "label": label})
        existing.add(fname)
        downloaded += 1
        total_now = len(existing)
        elapsed = time.time() - start
        rate = downloaded / elapsed * 60  # files per minute
        eta_min = (TARGET - downloaded) / rate if rate > 0 else 999
        print(f"  [{downloaded}/{TARGET}] total={total_now} | {label} {teff:.0f}K | {rate:.1f} f/min | ETA ~{eta_min:.0f}min | {fname}", flush=True)

# Save manifest
print(f"\n{'='*60}", flush=True)
print(f"Downloaded: {downloaded} | Failed: {failed} | Time: {(time.time()-start)/60:.1f} min", flush=True)
if results:
    old = pd.read_csv(MANIFEST)
    new = pd.DataFrame(results)
    combo = pd.concat([old, new], ignore_index=True).drop_duplicates("file_name")
    combo.to_csv(MANIFEST, index=False)
    print(f"Manifest: {len(combo)} total spectra", flush=True)
    print(combo["label"].value_counts().to_string(), flush=True)
print("Done!", flush=True)
