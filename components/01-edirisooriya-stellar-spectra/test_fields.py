import pandas as pd, re, requests

df = pd.read_csv('dataset_manifest_labeled.csv')
print('Current:', len(df), 'files')

s = requests.Session()

known_fields = [
    ('apo25m','000+02'),
    ('apo25m','M13'),
    ('apo25m','M15'),
    ('apo25m','M92'),
    ('apo25m','M3'),
    ('apo25m','M67'),
    ('apo25m','000+60'),
    ('apo25m','120+12'),
    ('apo25m','180+04'),
]

for tel, field in known_fields:
    url = f'https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/{tel}/{field}/'
    r = s.get(url, timeout=15)
    names = re.findall(r'href="(apStar-dr17-[^"]+\.fits)"', r.text)
    print(f'{tel}/{field}: HTTP {r.status_code} | {len(names)} files listed')
