#!/usr/bin/env python3
"""Build editable draw.io diagrams using native cells and connected edges.

Export SVG with the official draw.io desktop CLI; see README.md.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
C = dict(ink='#283747', blue='#DCEAF6', green='#DDEEDC', gold='#FFF0C5',
         purple='#E9E0F2', coral='#F7DDD5', gray='#F2F4F5', white='#FFFFFF')

class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.model = ET.Element('mxGraphModel', dx='1880', dy='1320', grid='1', gridSize='10',
            guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1',
            pageScale='1', pageWidth=str(w), pageHeight=str(h), math='0', shadow='0')
        self.root=ET.SubElement(self.model,'root')
        ET.SubElement(self.root,'mxCell',id='0')
        ET.SubElement(self.root,'mxCell',id='1',parent='0')
        self.ids=0
    def node(self, label, x,y,w,h, color='white', size=18, bold=False, style=''):
        # Plain SVG text avoids foreignObject, including in journal converters.
        if 'verticalAlign=top' not in style:
            font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf'
            font = ImageFont.truetype(font_path, size)
            wrapped = []
            for line in label.split('\n'):
                words=line.split(' '); current=''
                for word in words:
                    candidate=(current+' '+word).strip()
                    if current and font.getlength(candidate)>w-18:
                        wrapped.append(current); current=word
                    else: current=candidate
                wrapped.append(current)
            label='\n'.join(wrapped)
        self.ids+=1; ident=f'n{self.ids}'
        st=(f'rounded=1;arcSize=8;whiteSpace=none;html=0;fillColor={C[color]};'
            f'strokeColor={C["ink"]};strokeWidth=1.4;fontColor={C["ink"]};'
            f'fontFamily=Arial;fontSize={size};fontStyle={1 if bold else 0};'
            'spacing=8;align=center;verticalAlign=middle;'+style)
        cell=ET.SubElement(self.root,'mxCell',id=ident,value=label,style=st,vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        return ident
    def text(self,label,x,y,w,h,size=17,bold=False,align='center'):
        return self.node(label,x,y,w,h,size=size,bold=bold,
            style=f'fillColor=none;strokeColor=none;rounded=0;align={align};spacing=0;')
    def group(self,label,x,y,w,h,color='gray'):
        return self.node(label,x,y,w,h,color,size=17,bold=True,
            style='dashed=1;dashPattern=5 4;verticalAlign=top;align=left;spacingTop=12;spacingLeft=14;')
    def edge(self,src,tgt,label='',points=None,dash=False,color=None,exit=(.5,0),entry=(.5,1)):
        self.ids+=1
        st=('edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=0;'
            f'endArrow=block;endFill=1;strokeWidth=1.5;strokeColor={color or C["ink"]};'
            'fontFamily=Arial;fontSize=15;labelBackgroundColor=#FFFFFF;'
            f'exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;'
            f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
            +('dashed=1;dashPattern=5 4;' if dash else ''))
        c=ET.SubElement(self.root,'mxCell',id=f'e{self.ids}',value=label,style=st,edge='1',parent='1',source=src,target=tgt)
        g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
        if points:
            a=ET.SubElement(g,'Array',**{'as':'points'})
            for x,y in points: ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
    def save(self,path):
        doc=ET.Element('mxfile',host='app.diagrams.net',agent='DLAN architecture builder',version='32.6.10',type='device',compressed='false')
        d=ET.SubElement(doc,'diagram',id=self.name.lower().replace(' ','-'),name=self.name)
        d.append(self.model)
        ET.indent(doc)
        ET.ElementTree(doc).write(path,encoding='utf-8',xml_declaration=True)


def architecture():
    d=Diagram('DLAN detailed architecture',1900,1360)
    d.text('DLAN  |  Dimension-specific Local Attention Network',30,20,1840,44,29,True)
    d.text('Bandwidth-adaptive training-reference retrieval → joint geographical attention → conditionally modulated prediction',30,69,1840,30,19)
    d.group('(a) Overall architecture',25,120,535,1160,'white')
    d.group('(b) Dimension-specific joint attention  ·  block ℓ, head h',580,120,745,1160,'white')
    d.group('(c) FiLM-modulated predictor',1345,120,530,1160,'white')

    # Panel (a): classic bottom-up stack, two residual bypasses.
    q=d.node('Query\nxᵢ ∈ ℝᵖ, zᵢ ∈ ℝᴰ',70,1125,190,70,'blue')
    r=d.node('Training reference bank\n{(xⱼ, zⱼ, yⱼ)}ⱼ∈train',310,1125,205,70,'green')
    ret=d.node('Bandwidth-adaptive retrieval\nd² = Σₐ (δₐ / bℓₐₕ)²\nTop-K per (ℓ,h) → ID union',285,1000,245,85,'gold',16)
    d.edge(r,ret)
    d.edge(q,ret,'zᵢ',[(275,1160),(275,1047)],exit=(1,.5),entry=(0,.5))
    qe=d.node('Shared encoders\nEₓ(xᵢ) + E_z(zᵢ)\nLayerNorm → eᵢ',70,875,190,100,'blue',17)
    re=d.node('Shared encoders\neⱼ = LN[Eₓ(xⱼ) + E_z(zⱼ)]\nvⱼⁱⁿ = eⱼ + Eᵧ(yⱼ)',285,875,245,100,'green',16)
    d.edge(q,qe)
    d.edge(ret,re,'gather aligned IDs')
    tk=d.node('Query state\nhᵢ⁽⁰⁾ = eᵢ',85,775,165,55,'blue',17)
    kv=d.node('Fixed reference tokens\nK input: eⱼ   |   V input: vⱼⁱⁿ',285,775,245,55,'green',16)
    d.edge(qe,tk);d.edge(re,kv)
    d.group('',100,410,385,325,'gray')
    d.text('× L blocks',110,417,145,30,17,True)
    att=d.node('Joint geographical cross-attention\nExpanded in panel (b)',140,655,305,55,'gold',17)
    n1=d.node('Add & Norm',170,590,245,40,'purple',17)
    ff=d.node('Feed-forward  ·  d → 4d → d',140,520,305,45,'blue',17)
    n2=d.node('Add & Norm',170,455,245,40,'purple',17)
    d.edge(tk,att,'Q');d.edge(kv,att,'K, V',[(500,750),(500,682)],entry=(1,.5))
    d.edge(att,n1);d.edge(n1,ff);d.edge(ff,n2)
    d.edge(tk,n1,points=[(115,750),(115,610)],exit=(0,.5),entry=(0,.5))
    d.edge(n1,n2,points=[(465,610),(465,475)],exit=(1,.5),entry=(1,.5))
    h=d.node('Local condition  hᵢ = hᵢ⁽ᴸ⁾',135,345,315,50,'purple',19)
    d.edge(n2,h)
    film=d.node('FiLM-modulated nonlinear predictor\nExpanded in panel (c)',135,245,315,65,'coral',17)
    d.edge(h,film)
    d.edge(q,film,'xᵢ',[(50,1160),(50,277)],exit=(0,.5),entry=(0,.5))
    out=d.node('Prediction  ŷᵢ',190,170,205,44,'coral',20,True);d.edge(film,out)
    d.text('Training: exclude the focal row by ID.  Inference: retain the labelled training bank.',50,1200,485,28,14)

    # Panel (b): shared IDs, score assembly, one softmax, shared weights.
    state=d.node('hᵢ⁽ℓ⁻¹⁾\nquery state',610,1140,180,60,'blue',17)
    keys=d.node('eⱼ\nreference key token',840,1140,200,60,'green',17)
    vals=d.node('vⱼⁱⁿ = eⱼ + Eᵧ(yⱼ)\nreference value token',1090,1140,210,60,'green',16)
    qp=d.node('Qₐₕ projection',610,1045,180,50,'blue',17)
    kp=d.node('Kₐₕ projection',840,1045,200,50,'green',17)
    vp=d.node('Vₐₕ projection',1090,1045,210,50,'green',17)
    d.edge(state,qp);d.edge(keys,kp);d.edge(vals,vp)
    content=d.node('Content match · average over D coordinates\nmᵢₕⱼ = (1/D) Σₐ (qᵢₐₕᵀ kⱼₐₕ)/√dₕ',610,930,430,75,'blue',17)
    d.edge(qp,content,entry=(.25,1));d.edge(kp,content,entry=(.75,1))
    delta=d.node('Relative coordinate\nδᵢⱼₐ = zⱼₐ − zᵢₐ',1090,930,210,75,'gray',17)
    geo=d.node('Directional bias\nGᵢₕⱼ = Σₐ gₐₕ(δ, |δ|, sgn δ)\nMLP: 3 → 32 → H',645,800,285,85,'gold',16)
    gau=d.node('Gaussian product prior\nPᵢₕⱼ = −½ Σₐ (δᵢⱼₐ / bℓₐₕ)²\nbℓₐₕ = softplus(ρℓₐₕ) + ε',1000,800,300,85,'gold',16)
    d.edge(delta,geo,points=[(1070,967),(1070,908),(785,908)],exit=(0,.5))
    d.edge(delta,gau)
    score=d.node('Add scores + padding mask\nsᵢₕⱼ = mᵢₕⱼ + Gᵢₕⱼ + Pᵢₕⱼ',760,680,430,70,'gold',18)
    d.edge(content,score,points=[(600,967),(600,715)],exit=(0,.5),entry=(0,.5))
    d.edge(geo,score,entry=(.25,1));d.edge(gau,score,entry=(.75,1))
    sm=d.node('ONE joint Softmax over reference IDs j\nαᵢₕⱼ = softmaxⱼ(sᵢₕⱼ)\nAttention dropout during training',760,560,430,80,'purple',18,True)
    d.edge(score,sm)
    pool=d.node('Weighted sum for each coordinate a\ncᵢₐₕ = Σⱼ Dropout(αᵢₕⱼ) vⱼₐₕ',760,445,430,70,'green',18)
    d.edge(sm,pool)
    d.edge(vp,pool,'V',[(1310,1070),(1310,480)],exit=(1,.5),entry=(1,.5))
    cat=d.node('Concatenate all coordinate–head contexts\nD × H × dₕ = Dd',760,335,430,65,'green',18)
    d.edge(pool,cat)
    proj=d.node('Output projection  Wₒ : Dd → d\nGeographical context  cᵢ',760,230,430,65,'blue',18)
    d.edge(cat,proj)
    d.text('To residual stack in (a)',790,178,370,30,17)
    d.text('a = 1,…,D   ·   h = 1,…,H   ·   dₕ = d/H\nAll coordinates use the same candidate IDs and the same α within each head.',620,1206,675,43,15)

    # Panel (c): explicit FiLM operations, a side condition stream.
    x=d.node('Query covariates\ntᵢ⁽⁰⁾ = xᵢ',1380,1135,220,65,'blue',18)
    hc=d.node('Local condition\nhᵢ ∈ ℝᵈ',1645,1135,195,65,'purple',18)
    d.group('',1370,355,475,735,'gray')
    d.text('R hidden layers\n(layer r shown)',1625,375,200,55,17,True)
    linear=d.node('Linear\naᵢ⁽ʳ⁾ = Wᵣ tᵢ⁽ʳ⁻¹⁾ + cᵣ',1390,980,215,70,'blue',17)
    cond=d.node('FiLM linear  d → 2mᵣ\nSplit → γᵣ(hᵢ), βᵣ(hᵢ)\nZero-initialised weights + bias',1625,980,200,90,'purple',15)
    d.edge(x,linear);d.edge(hc,cond)
    scale=d.node('Bounded scale\nsᵢ⁽ʳ⁾ = 1 + tanh γᵣ(hᵢ)',1625,825,200,70,'purple',16)
    d.edge(cond,scale)
    mul=d.node('×',1460,830,65,60,'coral',29,True,'shape=ellipse;')
    d.edge(linear,mul);d.edge(scale,mul,exit=(0,.5),entry=(1,.5))
    add=d.node('+',1460,695,65,60,'coral',29,True,'shape=ellipse;')
    d.edge(mul,add)
    d.edge(cond,add,'βᵣ(hᵢ)',[(1835,1025),(1835,725)],exit=(1,.5),entry=(1,.5))
    gelu=d.node('GELU',1390,570,215,55,'coral',19)
    drop=d.node('Dropout',1390,465,215,55,'gray',18)
    d.edge(add,gelu);d.edge(gelu,drop)
    d.text('tᵢ⁽ʳ⁾ ∈ ℝᵐʳ\nRepeat with the same hᵢ',1625,455,200,70,16)
    read=d.node('After R layers: linear readout\nŷᵢ = wᵀ tᵢ⁽ᴿ⁾ + b',1390,250,430,70,'coral',19)
    d.edge(drop,read)
    d.text('FiLM acts on hidden features.\nInitial scale = 1; initial shift = 0.\nNo response yᵢ enters the query path.',1385,165,440,65,16)
    d.text('× = elementwise multiplication\n+ = elementwise addition\nSolid arrow: forward data flow\nhᵢ is reused at each hidden layer',1380,1207,460,65,14,align='left')
    d.text('Implementation: attention_mode="gtdr" · product_knn · use_context_y=True · use_gaussian=True   |   Main experiments: L = 1, H = 1, dropout = 0.1',30,1300,1830,30,16)
    return d


def retrieval():
    d=Diagram('Bandwidth-adaptive reference retrieval',1500,870)
    d.text('DLAN  |  Bandwidth-adaptive reference retrieval',30,20,1440,50,28,True)
    d.text('A shared candidate support for every coordinate projection',30,78,1440,32,20)
    q=d.node('Query coordinates\nzᵢ ∈ ℝᴰ',45,180,240,85,'blue',20)
    r=d.node('Eligible training coordinates\n{zⱼ : j ∈ train, j ≠ i}',45,335,240,85,'green',19)
    b=d.node('Learnable scale parameters\nbℓₐₕ = softplus(ρℓₐₕ) + 10⁻⁶\nFor each block ℓ, coordinate a, head h',45,535,320,110,'gold',18)
    metric=d.node('Joint coordinate-scaled distance\nd²ℓₕ(i,j) = Σₐ [(zⱼₐ − zᵢₐ) / bℓₐₕ]²',420,250,470,100,'gold',20)
    d.edge(q,metric,exit=(1,.5),entry=(0,.3));d.edge(r,metric,exit=(1,.5),entry=(0,.7))
    d.edge(b,metric,'detach(b) for discrete retrieval',[(390,590),(390,375),(655,375)],dash=True,exit=(1,.5))
    top=d.node('Top-K nearest under each (ℓ,h) metric\nStable sorting; focal training row excluded by ID',975,250,470,100,'gold',19)
    d.edge(metric,top,exit=(1,.5),entry=(0,.5))
    union=d.node('Deduplicate the union over blocks and heads\nCᵢ = ⋃ℓ,ₕ TopKⱼ{−d²ℓₕ(i,j)}\nKeep aligned reference IDs; mask padding',975,445,470,110,'green',19)
    d.edge(top,union,exit=(.5,1),entry=(.5,0))
    gather=d.node('Gather xⱼ, zⱼ, yⱼ on the same IDs\nAll coordinate Q/K/V branches share Cᵢ\nONE softmax per query and head over Cᵢ',975,650,470,115,'purple',19)
    d.edge(union,gather,exit=(.5,1),entry=(.5,0))
    prior=d.node('Differentiable attention score\nGaussian term: −½ Σₐ (δᵢⱼₐ/bℓₐₕ)²\nBackpropagation updates the scales',420,535,470,110,'gold',19)
    d.edge(b,prior,'b',exit=(1,.5),entry=(0,.5))
    d.edge(prior,gather,exit=(1,.5),entry=(0,.5))
    d.text('L = H = 1: |Cᵢ| ≤ K.   General case: |Cᵢ| ≤ min(Neligible, LHK).\nReference tokens remain fixed across attention blocks; only the query state is updated.',40,785,1410,60,18)
    return d

if __name__=='__main__':
    for name, builder in [('dlan_mechanism',architecture),('dlan_retrieval',retrieval)]:
        builder().save(OUT/f'{name}.drawio')
        print(OUT/f'{name}.drawio')
