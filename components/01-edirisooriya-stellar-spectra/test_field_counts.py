import urllib.request
import re

base = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/"
test_fields = [
    "M13", "NGC6819", "000%2B14", "000%2B45", "000%2B60", 
    "001%2B01", "002%2B00", "002%2B04", "003%2B02", "003%2B05",
    "120%2B12", "060%2B10", "180%2B04"
]

for f in test_fields:
    url = f"{base}{f}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', html)
        print(f"{f}: {len(matches)} spectra found")
    except Exception as e:
        print(f"{f}: failed ({e})")
