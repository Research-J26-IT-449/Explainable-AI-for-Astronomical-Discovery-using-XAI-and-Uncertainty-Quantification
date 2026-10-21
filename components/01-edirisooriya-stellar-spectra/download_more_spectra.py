"""
Download 4,000 MORE SDSS APOGEE DR17 apStar spectra
Target: 5,000 total (already have 1,000)
Strategy: Pull from multiple APOGEE fields via direct HTTP from data.sdss.org
"""

import os
import requests
import time
import random
import pandas as pd
from pathlib import Path
from astropy.io import fits

RAW_DIR = Path("raw_spectra")
RAW_DIR.mkdir(exist_ok=True)
MANIFEST = "dataset_manifest_labeled.csv"
TARGET_NEW = 4000
BASE_URL = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

# ── Temperature → label mapping ──────────────────────────────────────────────
def teff_to_label(teff):
    if teff >= 10000:          return "B"
    elif teff >= 7500:         return "A"
    elif teff >= 6000:         return "F"
    elif teff >= 5200:         return "G"
    else:                      return "K"

# ── APOGEE DR17 fields — expanded list of real fields ────────────────────────
# Format: (hemisphere, field_name)
FIELDS = [
    # Galactic bulge & disk
    ("apo25m", "000+02"), ("apo25m", "000+04"), ("apo25m", "000+08"),
    ("apo25m", "004+02"), ("apo25m", "008+04"), ("apo25m", "012+04"),
    ("apo25m", "016+04"), ("apo25m", "020+04"), ("apo25m", "024+04"),
    ("apo25m", "028+04"), ("apo25m", "032+04"), ("apo25m", "036+04"),
    ("apo25m", "040+04"), ("apo25m", "044+04"), ("apo25m", "048+04"),
    ("apo25m", "052+04"), ("apo25m", "056+04"), ("apo25m", "060+04"),
    ("apo25m", "064+04"), ("apo25m", "068+04"),
    ("apo25m", "072+04"), ("apo25m", "076+04"), ("apo25m", "080+04"),
    ("apo25m", "084+04"), ("apo25m", "088+04"), ("apo25m", "092+04"),
    ("apo25m", "096+04"), ("apo25m", "100+04"), ("apo25m", "104+04"),
    ("apo25m", "108+04"), ("apo25m", "112+04"), ("apo25m", "116+04"),
    ("apo25m", "120+04"), ("apo25m", "120+12"), ("apo25m", "124+04"),
    ("apo25m", "128+04"), ("apo25m", "132+04"), ("apo25m", "136+04"),
    ("apo25m", "140+04"), ("apo25m", "144+04"), ("apo25m", "148+04"),
    ("apo25m", "152+04"), ("apo25m", "156+04"), ("apo25m", "160+04"),
    ("apo25m", "164+04"), ("apo25m", "168+04"), ("apo25m", "172+04"),
    ("apo25m", "176+04"), ("apo25m", "180+04"), ("apo25m", "184+04"),
    ("apo25m", "188+04"), ("apo25m", "192+04"), ("apo25m", "196+04"),
    ("apo25m", "200+04"), ("apo25m", "204+04"), ("apo25m", "208+04"),
    ("apo25m", "212+04"), ("apo25m", "216+04"), ("apo25m", "220+04"),
    ("apo25m", "224+04"), ("apo25m", "228+04"), ("apo25m", "232+04"),
    ("apo25m", "236+04"), ("apo25m", "240+04"), ("apo25m", "244+04"),
    ("apo25m", "248+04"), ("apo25m", "252+04"), ("apo25m", "256+04"),
    ("apo25m", "260+04"), ("apo25m", "264+04"), ("apo25m", "268+04"),
    ("apo25m", "272+04"), ("apo25m", "276+04"), ("apo25m", "280+04"),
    ("apo25m", "284+04"), ("apo25m", "288+04"), ("apo25m", "292+04"),
    ("apo25m", "296+04"), ("apo25m", "300+04"), ("apo25m", "304+04"),
    ("apo25m", "308+04"), ("apo25m", "312+04"), ("apo25m", "316+04"),
    ("apo25m", "320+04"), ("apo25m", "324+04"), ("apo25m", "328+04"),
    ("apo25m", "332+04"), ("apo25m", "336+04"), ("apo25m", "340+04"),
    ("apo25m", "344+04"), ("apo25m", "348+04"), ("apo25m", "352+04"),
    ("apo25m", "356+04"),
    # High-latitude halo fields
    ("apo25m", "000+60"), ("apo25m", "090+60"), ("apo25m", "180+60"),
    ("apo25m", "270+60"), ("apo25m", "000+30"), ("apo25m", "060+30"),
    ("apo25m", "120+30"), ("apo25m", "180+30"), ("apo25m", "240+30"),
    ("apo25m", "300+30"),
    # Globular clusters
    ("apo25m", "M13"), ("apo25m", "M15"), ("apo25m", "M92"),
    ("apo25m", "M3"),  ("apo25m", "M67"), ("apo25m", "M53"),
    ("apo25m", "M71"), ("apo25m", "M107"), ("apo25m", "NGC6791"),
    ("apo25m", "NGC6819"), ("apo25m", "47Tuc"), ("apo25m", "omeCen"),
    # Open clusters
    ("apo25m", "N188"), ("apo25m", "N6819"), ("apo25m", "Pleiades"),
    # Southern (LCO)
    ("lco25m", "000+02"), ("lco25m", "000+04"), ("lco25m", "060+04"),
    ("lco25m", "120+04"), ("lco25m", "180+04"), ("lco25m", "240+04"),
    ("lco25m", "300+04"), ("lco25m", "000-60"), ("lco25m", "SMC"),
    ("lco25m", "LMC"), ("lco25m", "47Tuc"), ("lco25m", "omeCen"),
]

# ── Get existing files ────────────────────────────────────────────────────────
existing = set(f.name for f in RAW_DIR.glob("*.fits"))
print(f"Already have: {len(existing)} FITS files")
print(f"Need to download: {TARGET_NEW} more\n")

# ── Try to get a file listing from each field ─────────────────────────────────
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 APOGEE-research"})

def list_field_files(telescope, field):
    """Get list of apStar file URLs from a field directory."""
    url = f"{BASE_URL}/{telescope}/{field}/"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        # Parse filenames from directory listing HTML
        import re
        names = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', r.text)
        return [f"{url}{n}" for n in names]
    except Exception:
        return []

def download_file(url, dest_path):
    """Download a single FITS file."""
    try:
        r = session.get(url, timeout=60, stream=True)
        if r.status_code != 200:
            return False
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception:
        return False

def extract_meta(fits_path):
    """Extract TEFF, LOGG, FEH from FITS header."""
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            teff = float(hdr.get("TEFF", 0))
            logg = float(hdr.get("LOGG", 0))
            feh  = float(hdr.get("FEH",  0))
            if teff <= 0:
                return None
            return teff, logg, feh
    except Exception:
        return None

# ── Main download loop ────────────────────────────────────────────────────────
new_records = []
downloaded  = 0
attempted   = 0
fields_tried = 0

random.shuffle(FIELDS)  # randomise field order for variety

for telescope, field in FIELDS:
    if downloaded >= TARGET_NEW:
        break

    fields_tried += 1
    urls = list_field_files(telescope, field)
    if not urls:
        print(f"  [{telescope}/{field}] No listing or unreachable — skip")
        continue

    random.shuffle(urls)
    field_count = 0

    for url in urls:
        if downloaded >= TARGET_NEW:
            break

        fname = url.split("/")[-1]
        if fname in existing:
            continue  # already have it

        dest = RAW_DIR / fname
        attempted += 1

        ok = download_file(url, dest)
        if not ok:
            continue

        meta = extract_meta(dest)
        if meta is None:
            dest.unlink(missing_ok=True)
            continue

        teff, logg, feh = meta
        label = teff_to_label(teff)
        new_records.append({
            "file_name": fname,
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "label": label
        })
        existing.add(fname)
        downloaded += 1
        field_count += 1

        if downloaded % 50 == 0:
            print(f"  ✓ Downloaded {downloaded}/{TARGET_NEW} | latest: {fname} ({label}, {teff:.0f}K)")
        time.sleep(0.05)  # polite delay

    if field_count:
        print(f"  [{telescope}/{field}] +{field_count} spectra")

# ── Merge with existing manifest ──────────────────────────────────────────────
print(f"\nDownload complete: {downloaded} new files ({attempted} attempts)")

if new_records:
    old_df = pd.read_csv(MANIFEST)
    new_df = pd.DataFrame(new_records)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.to_csv(MANIFEST, index=False)
    print(f"\nManifest updated: {len(combined)} total spectra")
    print("Class distribution:")
    print(combined["label"].value_counts())
else:
    print("No new records to add.")

print("\nDone!")
