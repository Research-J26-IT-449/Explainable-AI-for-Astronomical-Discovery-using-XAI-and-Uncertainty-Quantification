import requests
from astroquery.sdss import SDSS

print("Testing SDSS query connection...")
query = """
SELECT TOP 5 s.specobjid, s.plate, s.mjd, s.fiberid, s.class, s.subclass, s.snMedian
FROM SpecObjAll s
WHERE s.class = 'STAR' AND s.snMedian > 30 AND s.zwarning = 0
"""
try:
    res = SDSS.query_sql(query)
    print("SDSS query result:")
    print(res)
except Exception as e:
    print(f"Error querying SDSS: {e}")
