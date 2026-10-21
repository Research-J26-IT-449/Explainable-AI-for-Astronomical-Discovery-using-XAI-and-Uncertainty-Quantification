import time
from astroquery.sdss import SDSS

print("Testing SDSS.get_spectra via SAS (data.sdss.org)...")
try:
    # Plate 266, MJD 51602, FiberID 28 (from our earlier successful query)
    sp = SDSS.get_spectra(plate=266, mjd=51602, fiberID=28, data_release=17)
    print(f"Success! Retrieved {len(sp)} spectrum:")
    sp[0].info()
    data = sp[0][1].data
    print(f"Flux length: {len(data['flux'])}, loglam range: {data['loglam'].min()} - {data['loglam'].max()}")
except Exception as e:
    print(f"Failed to retrieve: {e}")
