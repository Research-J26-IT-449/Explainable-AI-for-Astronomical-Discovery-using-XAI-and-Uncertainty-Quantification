from astropy.io import fits
import os

raw_dir = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\raw_spectra"

# Check a sample of files to confirm they are real SDSS APOGEE FITS
sample_files = [
    "apStar-dr17-2M06010117+3214538.fits",     # original (Oct 5)
    "apStar-dr17-2M17412127-2819140.fits",     # newly downloaded (Oct 18)
]

for fname in sample_files:
    fpath = os.path.join(raw_dir, fname)
    if not os.path.exists(fpath):
        print(f"NOT FOUND: {fname}")
        continue
    
    with fits.open(fpath) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        print(f"\n===== {fname} =====")
        print(f"  File size:        {os.path.getsize(fpath):,} bytes")
        print(f"  HDU count:        {len(hdul)}")
        print(f"  TELESCOPE:        {hdr0.get('TELESCOP', 'N/A')}")
        print(f"  SURVEY:           {hdr0.get('SURVEY', 'N/A')}")
        print(f"  INSTRUME:         {hdr0.get('INSTRUME', 'N/A')}")
        print(f"  APSTAR_ID:        {hdr0.get('APSTAR_ID', hdr0.get('OBJID', 'N/A'))}")
        print(f"  RA:               {hdr0.get('RA', 'N/A')}")
        print(f"  DEC:              {hdr0.get('DEC', 'N/A')}")
        print(f"  RV_TEFF:          {hdr0.get('RV_TEFF', 'N/A')} K")
        print(f"  NAXIS1 (pixels):  {hdr1.get('NAXIS1', 'N/A')}")
        print(f"  CRVAL1 (log-lam): {hdr1.get('CRVAL1', 'N/A')}")
