import urllib.request
import re

base = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/"
candidate_fields = [
    "000%2B08", "000%2B12", "000%2B16", "000%2B30", "000%2B75", 
    "001-01", "002%2B06", "002-04", "003%2B04", "004%2B00",
    "004%2B02", "004%2B04", "005%2B00", "005%2B02", "005%2B04",
    "M15", "M92", "M3", "M5", "M67", "M71", "M35",
    "030%2B00", "045%2B00", "090%2B00", "150%2B00", "210%2B00"
]

valid = []
for f in candidate_fields:
    url = f"{base}{f}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as r:
            html = r.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', html)
        if len(matches) > 0:
            print(f"{f}: {len(matches)} spectra")
            valid.append((f, len(matches)))
    except Exception as e:
        pass

print(f"\nTotal viable fields found: {len(valid)}")
for name, cnt in valid:
    print(f"  {name}: {cnt}")
