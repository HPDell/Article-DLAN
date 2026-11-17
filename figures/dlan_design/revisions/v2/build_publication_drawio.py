#!/usr/bin/env python3
"""Publication main figure: all labels >=22 units at a width of 1100 units."""
from build_dlan_drawio import Diagram, OUT


def publication():
    d=Diagram('DLAN publication architecture',1100,940)
    # Standalone panel labels; no in-figure title or implementation footnote.
    d.text('(a) Model architecture',10,12,345,36,22,True,align='left')
    d.text('(b) Joint attention',380,12,350,36,22,True,align='left')
    d.text('(c) FiLM predictor',765,12,325,36,22,True,align='left')
    pred=d.node('Prediction ŷᵢ',90,82,210,48,'coral',20)
    head=d.node('FiLM predictor\nsee (c)',65,175,245,64,'coral',20)
    cond=d.node('Local condition hᵢ',65,265,245,45,'purple',20)
    d.group('',45,325,285,295,'gray')
    d.text('× L',52,330,60,28,20,True)
    n2=d.node('Add & Norm',95,365,185,40,'purple',20)
    ffn=d.node('Feed-forward',65,430,245,42,'blue',20)
    n1=d.node('Add & Norm',95,495,185,40,'purple',20)
    att=d.node('Joint attention\nsee (b)',65,560,245,55,'gold',20)
    qt=d.node('Query\neᵢ',25,665,110,65,'blue',20)
    rt=d.node('Reference\nK: eⱼ\nV: eⱼ+Eᵧ(yⱼ)',165,650,180,88,'green',20)
    ret=d.node('Top-K union\nusing bℓₐₕ',165,760,180,60,'gold',20)
    q=d.node('Query\n(xᵢ, zᵢ)',25,840,110,65,'blue',20)
    r=d.node('Training bank\n(xⱼ, zⱼ, yⱼ)',165,840,180,65,'green',20)
    d.edge(q,qt);d.edge(r,ret);d.edge(ret,rt)
    d.edge(q,ret,'zᵢ',[(150,872),(150,790)],exit=(1,.5),entry=(0,.5))
    d.edge(qt,att)
    d.edge(rt,att,points=[(355,694),(355,587)],exit=(1,.5),entry=(1,.5))
    d.edge(att,n1);d.edge(n1,ffn);d.edge(ffn,n2);d.edge(n2,cond);d.edge(cond,head);d.edge(head,pred)
    d.edge(qt,n1,points=[(80,640),(55,640),(55,515)],exit=(.5,0),entry=(0,.5))
    d.edge(n1,n2,points=[(320,515),(320,385)],exit=(1,.5),entry=(1,.5))
    d.edge(q,head,'xᵢ',[(10,872),(10,207)],exit=(0,.5),entry=(0,.5))

    out=d.node('Concat + output projection',395,82,320,60,'blue',20)
    pool=d.node('Shared-weight pooling\ncᵢₐₕ = Σⱼ α̃ᵢₕⱼ vⱼₐₕ',395,200,320,68,'green',20)
    sm=d.node('Masked joint softmax\nthen attention dropout',395,325,320,68,'purple',20)
    score=d.node('sᵢₕⱼ = Cᵢₕⱼ + Gᵢₕⱼ + Pᵢₕⱼ\nPᵢₕⱼ = −½Σₐ(δₐ/bℓₐₕ)²',395,445,320,90,'gold',20)
    content=d.node('Content mean\nCᵢₕⱼ',380,595,160,70,'blue',20)
    geo=d.node('Geometry sum\nGᵢₕⱼ + Pᵢₕⱼ',560,595,165,70,'gold',20)
    qk=d.node('Qₐₕ, Kₐₕ\nprojections',380,735,160,70,'blue',20)
    rel=d.node('Offsets δₐ\nScales bℓₐₕ',560,735,165,85,'gold',20)
    v=d.node('Projected values Vₐₕ vⱼⁱⁿ',395,850,320,48,'green',20)
    d.edge(qk,content);d.edge(rel,geo);d.edge(content,score);d.edge(geo,score)
    d.edge(score,sm);d.edge(sm,pool);d.edge(pool,out)
    d.edge(v,pool,points=[(740,874),(740,234)],exit=(1,.5),entry=(1,.5))

    read=d.node('Linear readout',785,82,290,60,'coral',20)
    d.group('',770,220,310,610,'gray')
    d.text('× R layers',955,235,125,55,20,True)
    drop=d.node('Dropout',785,305,150,48,'gray',20)
    gelu=d.node('GELU',785,400,150,48,'coral',20)
    add=d.node('+',835,505,50,50,'coral',26,False,'shape=ellipse;')
    mul=d.node('×',835,625,50,50,'coral',26,False,'shape=ellipse;')
    lin=d.node('Linear',785,750,150,48,'blue',20)
    film=d.node('FiLM\nγ, β',950,735,125,70,'purple',20)
    scale=d.node('Scale\n1+tanh(γ)',950,615,125,75,'purple',20)
    xin=d.node('xᵢ',785,850,150,48,'blue',20)
    hi=d.node('hᵢ',965,850,105,48,'purple',20)
    d.edge(xin,lin);d.edge(hi,film);d.edge(film,scale);d.edge(lin,mul)
    d.edge(scale,mul,exit=(0,.5),entry=(1,.5))
    d.edge(film,add,'β',[(1090,770),(1090,530)],exit=(1,.5),entry=(1,.5))
    d.edge(mul,add);d.edge(add,gelu);d.edge(gelu,drop)
    d.edge(drop,read,points=[(860,190),(930,190)])
    # At the manuscript's 34 pc width, 22 drawing units exceed 8 pt.
    for c in d.root.findall('mxCell'):
        if c.get('vertex')=='1' and c.get('value'):
            style=c.get('style')
            if 'fontSize=20;' in style:style=style.replace('fontSize=20;','fontSize=22;')
            elif 'fontSize=22;' in style:style=style.replace('fontSize=22;','fontSize=24;')
            c.set('style',style)
    # Use an exact minimum text/line scale throughout the main figure.
    for c in d.root.findall('mxCell'):
        if c.get('edge')=='1':
            c.set('style',c.get('style').replace('fontSize=15;','fontSize=22;'))
    return d

if __name__=='__main__':
    publication().save(OUT/'dlan_architecture.drawio')
    print(OUT/'dlan_architecture.drawio')
