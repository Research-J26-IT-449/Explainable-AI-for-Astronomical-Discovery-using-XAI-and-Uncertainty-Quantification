import urllib.request
from astropy.io import fits

test_url = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/0266/spec-0266-51602-0028.fits"
print(f"Testing direct download from: {test_url}")
req = urllib.request.Request(test_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data_bytes = resp.read()
    print(f"Success! Downloaded {len(data_bytes)} bytes.")
    with open("test_spec.fits", "wb") as f:
        f.write(data_bytes)
    with fits.open("test_spec.fits") as hdul:
        hdul.info()
        hdr = hdul[2].data
        print("Classes/Subclasses in HDU 2:", hdr['CLASS'], hdr['SUBCLASS'])
except Exception as e:
    print(f"Failed: {e}")
