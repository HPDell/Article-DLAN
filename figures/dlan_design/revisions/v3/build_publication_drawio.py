#!/usr/bin/env python3
"""Publication main figure: all labels >=22 units at a width of 1100 units."""
from build_dlan_drawio import Diagram, OUT
from dimension_attention_panel import add_attention_panel


def publication():
    d=Diagram('DLAN publication architecture',1100,1220)
    # Standalone panel labels; no in-figure title or implementation footnote.
    d.text('(a) Model architecture',10,12,345,36,22,True,align='left')
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

    # Promote the unchanged overview before adding the wider mechanism panel.
    # At the manuscript's 34 pc width, 22 drawing units exceed 8 pt.
    for c in d.root.findall('mxCell'):
        if c.get('vertex')=='1' and c.get('value'):
            style=c.get('style')
            if 'fontSize=20;' in style:style=style.replace('fontSize=20;','fontSize=22;')
            elif 'fontSize=22;' in style:style=style.replace('fontSize=22;','fontSize=24;')
            c.set('style',style)
    add_attention_panel(d)

    # The prediction head moves below the model/attention pair.
    d.text('(c) FiLM predictor',10,970,340,40,24,True,align='left')
    d.group('',145,1010,735,190,'gray')
    d.text('× R layers',665,1020,205,36,22,True)
    xin=d.node('xᵢ',10,1120,105,55,'blue',22)
    linear=d.node('Linear',165,1120,135,55,'blue',22)
    modulation=d.node('(1+tanh γᵣ) ⊙ aᵣ + βᵣ',345,1120,305,55,'purple',22)
    activation=d.node('GELU + dropout',695,1120,165,55,'coral',22)
    readout=d.node('Linear readout',920,1120,165,55,'coral',22)
    condition=d.node('hᵢ → FiLM: γᵣ, βᵣ',345,1030,305,55,'purple',22)
    d.edge(xin,linear,exit=(1,.5),entry=(0,.5))
    d.edge(linear,modulation,exit=(1,.5),entry=(0,.5))
    d.edge(modulation,activation,exit=(1,.5),entry=(0,.5))
    d.edge(activation,readout,exit=(1,.5),entry=(0,.5))
    d.edge(condition,modulation,exit=(.5,1),entry=(.5,0))
    for cell in d.root.findall('mxCell'):
        if cell.get('edge')=='1':
            cell.set('style',cell.get('style').replace('fontSize=15;','fontSize=22;'))
    return d

if __name__=='__main__':
    publication().save(OUT/'dlan_architecture.drawio')
    print(OUT/'dlan_architecture.drawio')
