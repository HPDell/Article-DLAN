"""Validate direct and glossary-driven citation keys against the Zotero export."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
source=(root/'main.tex').read_text();bib=(root/'zotero.bib').read_text()
entries=set(re.findall(r'^@\w+\s*\{\s*([^,\s]+)\s*,',bib,re.M))
body=source.split(r'\begin{document}',1)[1]
keys=set()
for m in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',body):
    keys.update(k.strip()for k in m[1].split(',')if k.strip())
glossary={m[2]:m[1] for m in re.finditer(r'\\newacronym\[user1=\{([^}]+)\}\]\{([^}]+)\}',source)}
for m in re.finditer(r'\\[gG]lscite\{([^}]+)\}',body):
    keys.update(k.strip()for k in glossary.get(m[1],'').split(',')if k.strip())
missing=sorted(keys-entries)
assert r'\bibitem'not in source and r'\bibliography{zotero}'in source
print(f'{len(keys)} cited keys; {len(keys)-len(missing)} found; {len(missing)} missing.')
for key in missing:print('MISSING:',key)
(root/'evidence/current_citation_keys.txt').write_text('\n'.join(sorted(keys))+'\n')
if missing:raise SystemExit(1)
