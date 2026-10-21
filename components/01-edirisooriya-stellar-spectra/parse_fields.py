import re

with open("apo25m_index.html", "r", encoding="utf-16", errors="ignore") as f:
    text = f.read()

fields = re.findall(r'href="([^"/]+)/"', text)
fields = [f for f in fields if not f.startswith('?') and not f.startswith('/') and f != '..']
print(f"Total fields available in apo25m: {len(fields)}")
print("Sample fields:", fields[:20])
