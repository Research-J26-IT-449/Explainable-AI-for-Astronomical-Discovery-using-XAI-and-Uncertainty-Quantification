import urllib.request
import re

url = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/"
print(f"Fetching fields from {url} ...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8')
    fields = re.findall(r'href="([^"]+)/"', html)
    fields = [f for f in fields if not f.startswith('?') and not f.startswith('/')]
    print(f"Found {len(fields)} fields in apo25m! First 10: {fields[:10]}")
except Exception as e:
    print(f"Failed to fetch fields: {e}")
