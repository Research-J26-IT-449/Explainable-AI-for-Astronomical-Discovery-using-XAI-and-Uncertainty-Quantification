from astroquery.sdss import SDSS

classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
for c in classes:
    q = f"""
    SELECT COUNT(*) as cnt
    FROM SpecObjAll s
    WHERE s.class = 'STAR' AND s.subclass LIKE '{c}%' AND s.snMedian > 25 AND s.zwarning = 0
    """
    res = SDSS.query_sql(q)
    print(f"Class {c}: {res['cnt'][0]} stars available")
