import os
import re
import urllib.request
import urllib.error
import time
import socket

RAW_DIR = r"C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\raw_spectra"
BASE_URL = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/"

# Fields with hundreds/thousands of available spectra
TARGET_FIELDS = [
    "M15", "M13", "M92", "M3", "M67", "030%2B00", "120%2B12", 
    "180%2B04", "000%2B02", "000%2B04", "000%2B60", "000%2B08"
]

def get_field_fits_list(field):
    url = f"{BASE_URL}{field}/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8', errors='ignore')
            files = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', text)
            return list(set(files))
        except Exception as e:
            time.sleep(2)
    return []

def download_batch(target_total=1000, max_per_field=80):
    os.makedirs(RAW_DIR, exist_ok=True)
    existing = set([f for f in os.listdir(RAW_DIR) if f.endswith('.fits')])
    print(f"Current local spectra count: {len(existing)}")
    
    needed = target_total - len(existing)
    if needed <= 0:
        print("Target 1000 spectra already reached!")
        return

    print(f"Goal: Download {needed} more spectra to reach {target_total} spectra total...\n")

    for field in TARGET_FIELDS:
        if len(existing) >= target_total:
            break
        print(f"\n--- Field: {field} ---")
        fits_list = get_field_fits_list(field)
        print(f"Found {len(fits_list)} spectra in {field}")
        
        field_count = 0
        for fname in fits_list:
            if len(existing) >= target_total or field_count >= max_per_field:
                break
            if fname in existing:
                continue
            
            file_url = f"{BASE_URL}{field}/{fname}"
            dst = os.path.join(RAW_DIR, fname)
            
            success = False
            for retry in range(3):
                try:
                    req = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=40) as response, open(dst, 'wb') as out_file:
                        out_file.write(response.read())
                    
                    # Verify file is not empty
                    if os.path.getsize(dst) > 10000:
                        existing.add(fname)
                        field_count += 1
                        print(f"[{len(existing)}/{target_total}] Downloaded: {fname} (from {field})")
                        success = True
                        break
                    else:
                        if os.path.exists(dst): os.remove(dst)
                except Exception as e:
                    time.sleep(1.5)
            
            time.sleep(0.3)

    print(f"\nCompleted! Total FITS files now in raw_spectra: {len(existing)}")

if __name__ == '__main__':
    download_batch(target_total=1000, max_per_field=80)
