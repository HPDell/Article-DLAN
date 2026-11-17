#!/usr/bin/env python3
"""Audit deliverable integrity and publication-size typography, not model validity."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
NS = 'http://www.w3.org/2000/svg'


def semantic(element):
    attrs = dict(element.attrib)
    if element.tag in {'mxGeometry', 'mxPoint'}:
        # Draw.io omits zero coordinates and canonicalises numeric strings.
        for key in ['x', 'y']:
            attrs.setdefault(key, '0')
        for key in ['x', 'y', 'width', 'height']:
            if key in attrs:
                attrs[key] = f'{float(attrs[key]):.10f}'
    return (element.tag, sorted(attrs.items()), (element.text or '').strip(),
            [semantic(child) for child in element])


def inherited_sizes(element, inherited=16):
    size = float(str(element.attrib.get('font-size', inherited)).removesuffix('px'))
    style_size = re.search(r'(?:^|;)\s*font-size:\s*([\d.]+)', element.attrib.get('style', ''))
    if style_size:
        size = float(style_size.group(1))
    result = [size] if element.tag == f'{{{NS}}}text' else []
    for child in element:
        result.extend(inherited_sizes(child, size))
    return result


def audit_joint_attention(cells):
    """Verify the depicted merge, unique softmax and shared-weight fan-out."""
    roles = {}
    for cell in cells:
        match = re.search(r'(?:^|;)dlanRole=([^;]+)', cell.get('style', ''))
        if match:
            assert match.group(1) not in roles, match.group(1)
            roles[match.group(1)] = cell.get('id')
    edges = {(c.get('source'), c.get('target')) for c in cells if c.get('edge') == '1'}
    fusion, weights = roles['score_fusion'], roles['joint_softmax']
    assert (fusion, weights) in edges
    assert sum(t == weights for s,t in edges) == 1
    expected_pools = set()
    for axis in ['1', '2', 'D']:
        qk, score = roles[f'dimension_qk_{axis}'], roles[f'dimension_score_{axis}']
        value, pool = roles[f'dimension_value_{axis}'], roles[f'dimension_pool_{axis}']
        assert (qk, score) in edges and (score, fusion) in edges
        assert (weights, pool) in edges and (value, pool) in edges
        assert (pool, roles['attention_output']) in edges
        assert {s for s,t in edges if t == pool} == {weights, value}
        expected_pools.add(pool)
    assert {t for s,t in edges if s == weights} == expected_pools
    return {'displayed_dimensions': ['1', '2', 'D'], 'one_softmax_after_score_fusion': True,
            'same_attention_weights_feed_all_dimension_pools': True,
            'dimension_specific_qk_and_value_projections': True}


def main():
    report = {'drawio_version': '32.6.10', 'checks': {}}
    for stem in ['dlan_architecture', 'dlan_mechanism', 'dlan_retrieval', 'dlan_attention_detail', 'dlan_film_detail']:
        model = ET.parse(OUT/f'{stem}.drawio')
        svg = ET.parse(OUT/f'{stem}.svg').getroot()
        embedded = ET.fromstring(svg.attrib['content'])
        cells = model.findall('.//mxCell')
        assert [semantic(c) for c in cells] == [semantic(c) for c in embedded.findall('.//mxCell')]
        ids = {c.attrib['id'] for c in cells}
        for cell in cells:
            if cell.get('edge') == '1':
                assert cell.get('source') in ids and cell.get('target') in ids
        assert not svg.findall(f'.//{{{NS}}}foreignObject')
        assert not svg.findall(f'.//{{{NS}}}image')
        attention_audit = audit_joint_attention(cells) if stem in {'dlan_architecture', 'dlan_attention_detail'} else None
        report['checks'][stem] = {
            'editable_cells_match_embedded_model': True,
            'shapes': sum(c.get('vertex') == '1' for c in cells),
            'edges': sum(c.get('edge') == '1' for c in cells),
            'raster_images': 0, 'foreign_objects': 0,
            'sha256': hashlib.sha256((OUT/f'{stem}.svg').read_bytes()).hexdigest(),
        }
        if attention_audit:
            report['checks'][stem]['joint_attention_topology'] = attention_audit
    root = ET.parse(OUT/'dlan_architecture.svg').getroot()
    viewbox = list(map(float, root.attrib['viewBox'].split()))
    pt_per_unit = 408 / viewbox[2]
    min_font_pt = min(inherited_sizes(root)) * pt_per_unit
    line_widths = [float(e.get('stroke-width', '1')) for e in root.iter()
                   if e.get('stroke') not in {None, 'none'}]
    min_line_pt = min(line_widths) * pt_per_unit
    assert viewbox[3]/viewbox[2]*408*25.4/72 + 35 < 650*25.4/72
    assert min_font_pt >= 8
    assert min_line_pt >= .3
    # Check explicitly wrapped labels in their native Draw.io boxes.
    for stem in ['dlan_architecture', 'dlan_attention_detail']:
        for cell in ET.parse(OUT/f'{stem}.drawio').findall('.//mxCell[@vertex="1"]'):
            value, style = cell.get('value', ''), cell.get('style', '')
            if not value:
                continue
            geom = cell.find('mxGeometry')
            size = int(re.search(r'fontSize=(\d+)', style).group(1))
            bold = 'fontStyle=1;' in style
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf', size)
            lines = value.splitlines()
            assert len(lines)*size*1.2 <= float(geom.get('height')), (value, 'height')
            assert max(font.getlength(line) for line in lines) <= float(geom.get('width')), (value, 'width')
    print_svg = ET.parse(OUT/'dlan_architecture_print.svg').getroot()
    assert not print_svg.findall(f'.//{{{NS}}}text')
    assert not print_svg.findall(f'.//{{{NS}}}image')
    assert abs(float(print_svg.get('width').removesuffix('mm')) - 408*25.4/72) < 1e-5
    manifest = json.loads((OUT/'source_manifest.json').read_text())
    for path, checksum in manifest['sources'].items():
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest() == checksum, path
    report['publication_profile'] = {
        'manuscript_width_pt': 408, 'manuscript_width_mm': 408*25.4/72,
        'height_mm': viewbox[3]/viewbox[2]*408*25.4/72,
        'caption_space_reserved_mm': 35,
        'minimum_font_pt': min_font_pt, 'minimum_stroke_pt': min_line_pt,
        'minimum_font_target_pt': 8, 'publisher_minimum_stroke_pt': .3,
        'text_outlined_in_print_svg': True, 'source_files_unchanged': True,
        'scope': 'Quantitative artwork audit; structural fidelity and visual quality reviewed separately in PUBLICATION_REVIEW.md',
    }
    report['detail_profiles'] = {}
    for stem, width_mm in [('dlan_attention_detail', 408*25.4/72), ('dlan_film_detail', 104)]:
        detail = ET.parse(OUT/f'{stem}.svg').getroot()
        vb = list(map(float, detail.attrib['viewBox'].split()))
        font_pt = min(inherited_sizes(detail)) * width_mm/25.4*72/vb[2]
        height_mm = width_mm*vb[3]/vb[2]
        outlined = ET.parse(OUT/f'{stem}_print.svg').getroot()
        assert font_pt >= 8
        assert height_mm + 35 < 650*25.4/72
        assert not outlined.findall(f'.//{{{NS}}}text')
        assert not outlined.findall(f'.//{{{NS}}}image')
        strokes = [float(e.get('stroke-width', '1')) for e in detail.iter()
                   if e.get('stroke') not in {None, 'none'}]
        stroke_pt = min(strokes)*width_mm/25.4*72/vb[2]
        assert stroke_pt >= .3
        report['detail_profiles'][stem] = dict(width_mm=width_mm, height_mm=height_mm, minimum_font_pt=font_pt, minimum_stroke_pt=stroke_pt, caption_space_reserved_mm=35, text_outlined=True)
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['publication_profile'], indent=2))


if __name__ == '__main__':
    main()
