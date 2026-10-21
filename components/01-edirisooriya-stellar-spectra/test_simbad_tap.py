import urllib.request
import urllib.parse
import glob
import astropy.io.fits as fits
import pandas as pd

files = sorted(glob.glob('module1_stellar_spectra/raw_spectra/*.fits'))
objids = [fits.getheader(f, 0)['OBJID'] for f in files]

# 2MASS IDs in SIMBAD typically have format '2MASS J...'
simbad_ids = [f"2MASS J{oid[2:]}" for oid in objids]
id_list_str = ", ".join([f"'{sid}'" for sid in simbad_ids[:30]])

query = f"SELECT main_id, sp_type FROM basic WHERE main_id IN ({id_list_str})"
url = 'http://simbad.u-strasbg.fr/simbad/sim-tap/sync?request=doQuery&lang=adql&format=csv&query=' + urllib.parse.quote(query)
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode('utf-8')
        print("SIMBAD response:")
        print(content)
except Exception as e:
    print('HTTP error:', e)
