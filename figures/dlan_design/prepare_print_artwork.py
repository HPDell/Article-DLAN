#!/usr/bin/env python3
"""Prepare a font-independent vector SVG at the manuscript's 34 pc width."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

OUT = Path(__file__).resolve().parent
SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)


def run_inkscape(source: Path, target: Path, *args: str) -> None:
    subprocess.run(['inkscape', str(source), f'--export-filename={target}', *args], check=True)


def main() -> None:
    for stem, width_mm in [('dlan_architecture', 34*12*25.4/72),
                           ('dlan_attention_detail', 34*12*25.4/72),
                           ('dlan_film_detail', 104)]:
        source = OUT / f'{stem}.svg'
        target = OUT / f'{stem}_print.svg'
        run_inkscape(source, target, '--export-type=svg', '--export-text-to-path', '--export-plain-svg')
        tree = ET.parse(target)
        root = tree.getroot()
        assert not root.findall(f'.//{{{SVG_NS}}}text'), 'Print artwork still depends on fonts'
        viewbox = list(map(float, root.attrib['viewBox'].split()))
        root.set('width', f'{width_mm:.6f}mm')
        root.set('height', f'{width_mm * viewbox[3] / viewbox[2]:.6f}mm')
        tree.write(target, encoding='utf-8', xml_declaration=True)
    target = OUT / 'dlan_architecture_print.svg'
    run_inkscape(target, OUT/'dlan_architecture.png', '--export-type=png', '--export-width=1500')
    run_inkscape(target, OUT/'dlan_architecture_144mm.png', '--export-type=png', '--export-dpi=150')
    with Image.open(OUT/'dlan_architecture_144mm.png') as image:
        image.convert('L').save(OUT/'dlan_architecture_grayscale.png', dpi=(150, 150))


if __name__ == '__main__':
    main()
