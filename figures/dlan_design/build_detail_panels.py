#!/usr/bin/env python3
"""Extract connected mechanism panels into page-sized editable diagrams."""
import copy
import xml.etree.ElementTree as ET
from build_dlan_drawio import Diagram, OUT
from dimension_attention_panel import attention_detail


def panel(stem, title, xmin, xmax, compact=False):
    original = ET.parse(OUT/'dlan_mechanism.drawio')
    cells = original.findall('.//mxCell')
    selected = []
    for cell in cells:
        if cell.get('vertex') != '1':
            continue
        geo = cell.find('mxGeometry')
        x = float(geo.get('x', 0))
        if xmin <= x < xmax:
            if compact and (cell.get('value', '').startswith('FiLM acts on hidden features') or
                            cell.get('value', '').startswith('× = elementwise')):
                continue
            selected.append(cell)
    ids = {cell.get('id') for cell in selected}
    selected += [cell for cell in cells if cell.get('edge') == '1' and
                 cell.get('source') in ids and cell.get('target') in ids]
    diagram = Diagram(title, xmax-xmin, 900 if compact else 1160)
    transformed = []
    for cell in selected:
        cell = copy.deepcopy(cell)
        geo = cell.find('mxGeometry')
        for element in geo.iter():
            if element.get('x') is not None:
                element.set('x', str(float(element.get('x'))-xmin))
            if element.get('y') is not None:
                y = float(element.get('y'))-120
                element.set('y', str(y*.72+20 if compact else y*.85))
        if cell.get('vertex') == '1' and 'verticalAlign=top' in cell.get('style', ''):
            factor = .72 if compact else .85
            geo.set('height', str(float(geo.get('height'))*factor + (20 if compact and not cell.get('value') else 0)))
        if not compact and cell.get('value', '').startswith('l = 1'):
            geo.set('y', str(float(geo.get('y'))+10))
        # Labels refer to the main figure rather than standalone panel lettering.
        if cell.get('value', '').startswith('(b) Dimension-specific'):
            cell.set('value', 'Dimension-specific joint attention')
        elif cell.get('value', '').startswith('(c) FiLM-modulated'):
            cell.set('value', 'FiLM-modulated predictor')
        transformed.append(cell)
    by_id = {c.get('id'): c for c in transformed}
    for cell in transformed:
        if cell.get('edge') == '1' and cell.find('mxGeometry/Array') is not None:
            points = cell.find('mxGeometry/Array')
            src = by_id[cell.get('source')].find('mxGeometry')
            dst = by_id[cell.get('target')].find('mxGeometry')
            cy_src = float(src.get('y')) + float(src.get('height'))/2
            cy_dst = float(dst.get('y')) + float(dst.get('height'))/2
            if 'exitY=0.5;' in cell.get('style', ''):
                points[0].set('y', str(cy_src))
            if 'entryY=0.5;' in cell.get('style', ''):
                points[-1].set('y', str(cy_dst))
    for cell in transformed:
        diagram.root.append(cell)
    diagram.save(OUT/f'{stem}.drawio')


if __name__ == '__main__':
    attention_detail().save(OUT/'dlan_attention_detail.drawio')
    panel('dlan_film_detail', 'DLAN FiLM mechanism', 1345, 1875, compact=True)
