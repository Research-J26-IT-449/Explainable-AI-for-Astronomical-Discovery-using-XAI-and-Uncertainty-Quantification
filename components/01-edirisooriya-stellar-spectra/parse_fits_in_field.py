import re

with open("field_000_04.html", "r", encoding="utf-16", errors="ignore") as f:
    text = f.read()

fits_files = re.findall(r'href="(apStar-[^"]+\.fits)"', text)
print(f"Total apStar FITS files in field 000+04: {len(fits_files)}")
print("Sample files:", fits_files[:5])
