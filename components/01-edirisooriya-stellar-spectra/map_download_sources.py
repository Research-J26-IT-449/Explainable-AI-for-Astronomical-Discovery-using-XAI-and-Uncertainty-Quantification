import urllib.request
import re
import os

base_url = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/"
raw_files = set([f for f in os.listdir("raw_spectra") if f.endswith(".fits")])

fields_checked = [
    "000%2B02", "000%2B04", "000%2B14", "000%2B45", "000%2B60", 
    "001%2B01", "002%2B00", "002%2B04", "003%2B05", "120%2B12", 
    "180%2B04", "M13", "M15", "M92", "M3", "M67", "030%2B00", "000%2B75", "000%2B08"
]

field_summary = {}

for f in fields_checked:
    url = f"{base_url}{f}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        matches = set(re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', html))
        downloaded = matches.intersection(raw_files)
        if downloaded:
            field_summary[f] = {
                'url': url,
                'count': len(downloaded),
                'sample': list(downloaded)[:2]
            }
    except Exception as e:
        pass

print(f"Total local FITS files currently: {len(raw_files)}\n")
print(f"{'Field':<12} | {'Count':<5} | {'Exact SDSS SAS Directory URL'}")
print("-" * 80)
total_matched = 0
for f, info in field_summary.items():
    print(f"{f:<12} | {info['count']:<5} | {info['url']}")
    total_matched += info['count']

print("-" * 80)
print(f"Total matched across these fields: {total_matched}")
