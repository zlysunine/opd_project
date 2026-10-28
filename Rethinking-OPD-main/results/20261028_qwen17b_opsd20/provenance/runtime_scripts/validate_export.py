"""Validate the actual portable publication bundle, without executing report JavaScript."""
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/20261028_qwen17b_opsd20'


class Document(HTMLParser):
    def __init__(self):
        super().__init__();self.scripts=[];self.active=False;self.current=[];self.external=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for key in ('src','href'):
            if attrs.get(key) and not attrs[key].startswith(('data:','#')):
                self.external.append((tag,key,attrs[key]))
        if tag=='script':self.active=True;self.current=[]
    def handle_data(self,data):
        if self.active:self.current.append(data)
    def handle_endtag(self,tag):
        if tag=='script' and self.active:
            self.scripts.append(''.join(self.current));self.active=False


pages={}
for name,steps in [('OPSD_report.html',[0,1,5,10,20]),('acceptance_2step.html',[0,1,2])]:
    page=OUT/name;reader=Document();reader.feed(page.read_text())
    assert not reader.external,reader.external
    scripts=[v.strip() for v in reader.scripts if v.strip().startswith('window.REPORT_DATA=')]
    assert len(scripts)==1
    payload=json.loads(scripts[0][len('window.REPORT_DATA='):].removesuffix(';'))
    assert payload['meta']['monitor_steps']==steps,payload['meta']
    assert payload['meta']['standalone'] is True
    expected=list(range(1,steps[-1]+1));assert payload['meta']['training_steps']==expected
    assert all(g['path'].startswith('data:image/') for g in payload['gallery'])
    pages[name]=dict(bytes=page.stat().st_size,monitor_steps=steps,metrics=len(payload['metrics']),
                     embedded_rsm_images=len(payload['gallery']),offline_assets=True)
summary=json.loads((OUT/'summary.json').read_text());assert summary['passed']
assert [r['step'] for r in summary['training']]==list(range(1,21))
assert summary['reuse']['generation_calls']==0
assert all(r['extraction_success_rate']>=.7 and r['complete_extractable_rate']>=.7 for r in summary['checkpoints'])
with (OUT/'data/behavior.csv').open(encoding='utf-8-sig') as f:behavior=list(csv.DictReader(f))
assert len(behavior)==54*5
assert sum(r['split']=='heldout_probe' for r in behavior)==50*5
for name in ['data/checkpoint_metrics.csv.gz','data/distributions.csv.gz','data/curvature.csv.gz']:
    with gzip.open(OUT/name,'rt',encoding='utf-8-sig') as f:
        assert sum(1 for row in csv.DictReader(f))>0
for path in OUT.rglob('*'):
    if path.is_file():
        assert path.suffix.lower() not in {'.pt','.pth','.bin','.safetensors','.ckpt','.npz'}
        assert path.stat().st_size<50*1024*1024
for link in ['OPSD_report.html','acceptance_2step.html','HANDOFF_zh.md','summary.json']:
    assert (OUT/link).is_file()
result=dict(passed=True,html=pages,all_20_training_steps=True,all_5_checkpoints=True,
            behavior_rows=len(behavior),weights_exported=False,all_data_streams_readable=True,
            limitations=['Node runtime control test skipped; HTML structure, embedded assets and data validated.'])
(OUT/'provenance/export_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
