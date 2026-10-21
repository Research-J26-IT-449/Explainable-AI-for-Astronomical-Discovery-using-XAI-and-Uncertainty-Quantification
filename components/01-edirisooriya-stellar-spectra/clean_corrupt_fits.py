import os
from astropy.io import fits

raw_dir = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\raw_spectra"
removed = 0
valid = 0

for fname in os.listdir(raw_dir):
    if not fname.endswith('.fits'):
        continue
    fpath = os.path.join(raw_dir, fname)
    try:
        if os.path.getsize(fpath) == 0:
            os.remove(fpath)
            removed += 1
            continue
        with fits.open(fpath) as hdul:
            _ = hdul[0].header
        valid += 1
    except Exception as e:
        try:
            os.remove(fpath)
            removed += 1
        except:
            pass

print(f"Cleanup complete! Removed {removed} corrupt files. Valid FITS files: {valid}")
