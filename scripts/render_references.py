"""Embed an audited author-date bibliography in the single manuscript source."""
from pathlib import Path
from html import unescape
import re,json,unicodedata
R=Path(__file__).resolve().parents[1];p=R/'main.tex';s=p.read_text();data=json.loads((R/'evidence/references_crossref.json').read_text())
keys=set(','.join(re.findall(r'\\cite[pt]?\{([^}]+)\}',s)).split(','))
def clean(x):return re.sub(r'\s+',' ',unescape(re.sub('<[^>]+>','',str(x)))).strip()
def tex(x):
 x=clean(x).replace('–','--').replace('—','---').replace('‐','-').replace('−','-').replace('’',"'").replace('‘',"'").replace('“','``').replace('”',"''")
 x=''.join(c for c in unicodedata.normalize('NFKD',x) if not unicodedata.combining(c))
 return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$'}.get(c,c)for c in x)
def name(a):return a.get('literal') or (a.get('given','')+' '+a.get('family','')).strip()
def family(a):return a.get('family') or a['literal'].split()[-1]
records=[]
for k in keys:
 m=data[k]['metadata'];authors=m.get('author',[]);assert authors,(k,'missing authors')
 # Prefer print/issue year over online-first year in journal records.
 date=m.get('published-print',m.get('published',m.get('issued')))
 year=date['date-parts'][0][0];title=tex(m['title'][0]);families=[family(a)for a in authors]
 short=families[0] if len(authors)==1 else (' and '.join(families)if len(authors)==2 else families[0]+' et al.')
 auth='; '.join(name(a)for a in authors)
 container=tex((m.get('container-title')or [''])[0]);vol=tex(m.get('volume',''));issue=tex(m.get('issue',''));pages=tex(m.get('page',m.get('article-number',''))).replace('-','--')
 loc=(r'\textit{'+container+'}'if container else tex(m.get('publisher','')))
 if vol:loc+=' '+r'\textbf{'+vol+'}'+('('+issue+')'if issue else '')
 if pages:loc+=', '+pages
 doi=m.get('DOI');url='https://doi.org/'+doi if doi else m.get('URL',data[k]['source'])
 link=r'\href{'+url.replace('%',r'\%')+'}{'+('doi:'+doi.replace('_',r'\_')if doi else 'Publisher record')+'}'
 item=r'\bibitem['+tex(short)+'('+str(year)+')]{'+k+'}\n'+tex(auth)+'. '+str(year)+'. '+title+'. '+loc+'. '+link+'.'
 records.append((families[0].lower(),year,title,k,item))
records.sort(); assert len(records)>=50
text=r'\begin{thebibliography}{99}'+'\n'+'\n\n'.join(x[-1]for x in records)+'\n'+r'\end{thebibliography}'
s=re.sub(r'% REFERENCES_BEGIN.*?% REFERENCES_END',lambda _: '% REFERENCES_BEGIN\n'+text+'\n% REFERENCES_END',s,flags=re.S);p.write_text(s)
(R/'evidence/reference_audit.csv').write_text('key,year,source\n'+'\n'.join(f'{k},{year},{data[k]["source"]}'for _,year,_,k,_ in records)+'\n')
print('Embedded',len(records),'unique cited references. All have a retained metadata source.')
