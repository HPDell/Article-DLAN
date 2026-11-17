"""Explicit dimension-wise scores and a single joint attention distribution."""
from build_dlan_drawio import Diagram, OUT


def mark(diagram: Diagram, ident: str, role: str) -> str:
    cell = diagram.root.find(f"mxCell[@id='{ident}']")
    cell.set('style', cell.get('style') + f'dlanRole={role};')
    return ident


def add_attention_panel(diagram: Diagram, x: float = 370, y: float = 0,
                        detailed: bool = False, label: str = '(b)') -> None:
    """Draw one block/head; 1, 2, D are representative coordinate branches."""
    d = diagram
    d.text(f'{label} Dimension-wise scores → joint attention'.strip(),
           x+5, y+12, 715, 40, 24, True, align='left')
    d.text('One block r and head h; all D coordinate branches are used',
           x+5, y+65, 715, 34, 22)
    out = mark(d, d.node('Concatenate all l,h → output projection',
                         x+45, y+125, 620, 60, 'blue', 22), 'attention_output')
    softmax = mark(d, d.node('ONE joint softmax over shared candidate IDs j\nMask → softmax → attention dropout: α̃ᵢₕⱼ',
                             x+15, y+420, 685, 80, 'purple', 22, True), 'joint_softmax')
    fusion = mark(d, d.node('Combine scores across all coordinates\ns = (1/D)Σₗ cₗ + Σₗ gₗ + Σₗ pₗ',
                            x+50, y+535, 615, 80, 'gold', 22), 'score_fusion')
    d.edge(fusion, softmax)
    for index, axis in enumerate(['1', '2', 'D']):
        left = x+15 + 235*index
        width = 215
        sub = ['₁', '₂', 'ₗ'][index]
        pool = mark(d, d.node(f'Context l = {axis}\nΣⱼ α̃ᵢₕⱼ vⱼ{ sub }ₕ',
                             left, y+230, width, 70, 'green', 22), f'dimension_pool_{axis}')
        values = mark(d, d.node(f'V{ sub } projection', left, y+330, width, 55,
                               'green', 22), f'dimension_value_{axis}')
        d.edge(values, pool)
        # Weight rails run in the free lane on the left of each dimension.
        rail = left-10
        branch_center = left+width/2
        d.edge(softmax, pool, points=[(branch_center, y+405), (rail, y+405), (rail, y+265)],
               exit=((branch_center-(x+15))/685, 0), entry=(0, .5),
               color='#6B5096')
        d.edge(pool, out, entry=((branch_center-(x+45))/620, 1))
        if detailed:
            score_label = (f'l = {axis}\nc{sub} = q{sub}ᵀk{sub}/√dₕ\n'
                           f'g{sub} = MLP{sub}(δ{sub},…)\np{sub} = −½(δ{sub}/b{sub})²')
            score_top, score_height = y+665, 135
        else:
            score_label = f'l = {axis}\nContent c{sub}\nBias g{sub}, prior p{sub}'
            score_top, score_height = y+685, 95
        score = mark(d, d.node(score_label, left, score_top, width, score_height,
                              'gold', 22), f'dimension_score_{axis}')
        proj = mark(d, d.node(f'Q{sub}, K{sub}\nprojections', left, y+835, width, 60,
                             'blue', 22), f'dimension_qk_{axis}')
        d.edge(proj, score)
        d.edge(score, fusion, entry=((branch_center-(x+50))/615, 1))
    d.text('⋯', x+460, y+190, 32, 36, 26)
    d.text('⋯', x+460, y+625, 32, 36, 26)
    d.text('Shared reference IDs; independent dimension projections',
           x+5, y+910, 715, 34, 22)


def attention_detail():
    diagram = Diagram('Dimension-wise scores and joint attention', 745, 1030)
    add_attention_panel(diagram, x=0, y=0, detailed=True, label='')
    return diagram


if __name__ == '__main__':
    attention_detail().save(OUT/'dlan_attention_detail.drawio')
