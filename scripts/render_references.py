"""Validate citation keys against zotero.bib; never generate inline bibitems."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
source = (root / 'main.tex').read_text()
bibliography = (root / 'zotero.bib').read_text()
entries = set(re.findall(r'^@\w+\s*\{\s*([^,\s]+)\s*,', bibliography, re.M))
citations = set()
for match in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}', source):
    citations.update(key.strip() for key in match[1].split(','))
missing = sorted(citations - entries)
assert r'\bibitem' not in source
assert r'\bibliography{zotero}' in source
print(f'{len(citations)} cited keys; {len(citations) - len(missing)} found; {len(missing)} missing.')
for key in missing:
    print(f'MISSING: {key}')
if missing:
    print('Titles and original sources: evidence/missing_references.md')
