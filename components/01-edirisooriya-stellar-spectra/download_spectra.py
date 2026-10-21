# download_spectra.py
"""
Download APOGEE DR17 spectra (apStar FITS files) using verified direct URLs.
Files are stored in the `raw_spectra` folder (created if missing).
"""

import os
import argparse
import urllib.request
import urllib.error

# Base URL for field 000+02 (real files confirmed from SDSS SAS directory listing)
BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/000%2B02/"

FILENAMES = [
    "apStar-dr17-2M17335483-2753043.fits",
    "apStar-dr17-2M17335915-2742223.fits",
    "apStar-dr17-2M17340445-2800033.fits",
    "apStar-dr17-2M17340500-2808243.fits",
    "apStar-dr17-2M17340793-2746343.fits",
    "apStar-dr17-2M17341132-2806083.fits",
    "apStar-dr17-2M17341788-2757189.fits",
    "apStar-dr17-2M17342737-2745032.fits",
    "apStar-dr17-2M17343177-2749299.fits",
    "apStar-dr17-2M17343249-2817260.fits",
    "apStar-dr17-2M17343333-2818449.fits",
    "apStar-dr17-2M17343440-2810490.fits",
    "apStar-dr17-2M17343798-2823294.fits",
    "apStar-dr17-2M17344049-2821066.fits",
    "apStar-dr17-2M17344156-2801557.fits",
    "apStar-dr17-2M17344497-2817405.fits",
    "apStar-dr17-2M17344608-2724318.fits",
    "apStar-dr17-2M17344844-2807400.fits",
    "apStar-dr17-2M17345235-2722089.fits",
    "apStar-dr17-2M17345272-2736505.fits",
    "apStar-dr17-2M17345377-2746243.fits",
    "apStar-dr17-2M17345572-2753373.fits",
    "apStar-dr17-2M17345763-2815528.fits",
    "apStar-dr17-2M17350086-2808051.fits",
    "apStar-dr17-2M17350195-2720524.fits",
    "apStar-dr17-2M17350512-2802456.fits",
    "apStar-dr17-2M17350983-2739429.fits",
    "apStar-dr17-2M17351027-2830318.fits",
    "apStar-dr17-2M17351062-2832534.fits",
    "apStar-dr17-2M17351127-2733449.fits",
    "apStar-dr17-2M17351291-2810554.fits",
    "apStar-dr17-2M17351647-2752123.fits",
    "apStar-dr17-2M17351654-2827218.fits",
    "apStar-dr17-2M17351741-2730541.fits",
    "apStar-dr17-2M17351858-2744278.fits",
    "apStar-dr17-2M17352222-2833101.fits",
    "apStar-dr17-2M17352252-2809573.fits",
    "apStar-dr17-2M17352593-2723384.fits",
    "apStar-dr17-2M17352605-2721516.fits",
    "apStar-dr17-2M17353172-2718506.fits",
    "apStar-dr17-2M17353200-2816293.fits",
    "apStar-dr17-2M17353215-2759106.fits",
    "apStar-dr17-2M17353239-2732368.fits",
    "apStar-dr17-2M17353467-2818243.fits",
    "apStar-dr17-2M17353542-2835576.fits",
    "apStar-dr17-2M17353666-2753317.fits",
    "apStar-dr17-2M17353909-2800469.fits",
    "apStar-dr17-2M17354074-2726492.fits",
    "apStar-dr17-2M17354253-2807096.fits",
    "apStar-dr17-2M17354431-2758434.fits",
    "apStar-dr17-2M17354499-2810410.fits",
    "apStar-dr17-2M17354575-2821241.fits",
    "apStar-dr17-2M17354627-2752106.fits",
    "apStar-dr17-2M17354760-2742056.fits",
    "apStar-dr17-2M17355017-2732011.fits",
    "apStar-dr17-2M17355137-2756486.fits",
    "apStar-dr17-2M17355503-2758205.fits",
    "apStar-dr17-2M17355669-2751098.fits",
    "apStar-dr17-2M17355687-2805005.fits",
    "apStar-dr17-2M17360107-2710573.fits",
    "apStar-dr17-2M17360667-2822112.fits",
]

# Also include the one file that already worked from the previous run
EXTRA = [
    ("https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/180+04/",
     "apStar-dr17-2M06010117+3214538.fits"),
]


def download_file(url, out_dir):
    fname = os.path.basename(url)
    dst = os.path.join(out_dir, fname)
    if os.path.exists(dst):
        print(f"[SKIP]  {fname} already exists")
        return
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"[OK]    {fname}")
    except urllib.error.HTTPError as e:
        print(f"[FAIL]  {fname}: HTTP {e.code}")
    except Exception as e:
        print(f"[FAIL]  {fname}: {e}")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = len(FILENAMES) + len(EXTRA)
    print(f"Downloading {total} spectra into '{out_dir}' ...")
    for fname in FILENAMES:
        download_file(BASE + fname, out_dir)
    for base_url, fname in EXTRA:
        download_file(base_url + fname, out_dir)
    fits_count = len([f for f in os.listdir(out_dir) if f.endswith(".fits")])
    print(f"\nDone. {fits_count} FITS files in '{out_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download APOGEE DR17 spectra")
    parser.add_argument("--out_dir", default="raw_spectra",
                        help="Directory to store FITS files")
    args = parser.parse_args()
    main(args.out_dir)
