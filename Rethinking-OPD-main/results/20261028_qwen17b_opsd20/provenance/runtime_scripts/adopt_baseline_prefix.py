"""Audited transfer of a fixed baseline prefix; never choose rows by outcome."""
import copy
import json
import os
from pathlib import Path
import time

from analysis.common import source_hashes
from analysis.evaluate import enforce_quality
from scripts.research.common import digest, file_hash, write_json
from scripts.research.job_control import stop_session
from scripts.research.preparation_cache import FILES, save_selection

ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
LOG=ROOT/'research_runs/server_handoff_20261028'
SOURCE=ROOT/'research_runs/base_opsd_20261028_acceptance_2step_q70_n100'
TARGET=ROOT/'research_runs/base_opsd_20261028_acceptance_2step_q70_n50'
SOURCE_SELECTION=LOG/'acceptance_selection_cache_q70_n100'


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def ensure_fifty_selection():
    c=json.loads((TARGET/'config.json').read_text())
    destination=Path(c['dataset']['preparation_cache'])
    if destination.exists():return destination
    source_receipt=json.loads((SOURCE_SELECTION/'selection_complete.json').read_text())
    for name,expected in source_receipt['files'].items():
        assert file_hash(SOURCE_SELECTION/name)==expected,name
    heldout=read_rows(SOURCE_SELECTION/'heldout_probe.jsonl')
    assert len(heldout)==100 and c['monitor']['heldout_probe_n']==50
    assert file_hash(SOURCE_SELECTION/'heldout_probe.jsonl')==c['monitor']['heldout_selection']['source_heldout_sha256']
    stage=LOG/'selection_prefix50_stage';stage.mkdir(exist_ok=False)
    for name in FILES:
        if name=='heldout_probe.jsonl':
            with (stage/name).open('w') as out:
                for row in heldout[:50]:out.write(json.dumps(row,ensure_ascii=False)+'\n')
        else:os.link(SOURCE_SELECTION/name,stage/name)
    save_selection(stage,c,destination)
    write_json(destination/'subset_lineage.json',dict(policy='fixed_prefix',source_n=100,target_n=50,
        source_manifest_hash=file_hash(SOURCE_SELECTION/'selection_complete.json'),
        selected_ids=[r['problem_id'] for r in heldout[:50]],selection_uses_outcomes=False,
        train_validation_train_probe_preserved=True,
        authorization='User reduced baseline to50 during evaluation; all first50 failures retained'))
    return destination


def wait_and_adopt():
    receipt_path=LOG/'baseline50_transfer.json'
    if receipt_path.is_file():
        report=json.loads(receipt_path.read_text())
        destination=TARGET/'shared/generation_cache/baseline_live'
        assert file_hash(destination/'identity.json')==report['target_identity_sha256']
        current=json.loads((destination/'identity.json').read_text())
        assert current['sources']==source_hashes()
        assert current['monitor']==json.loads((TARGET/'config.json').read_text())['monitor']
        for item in report['transferred']:
            assert file_hash(destination/f'{item["index"]:06d}.json')==item['sha256']
        return report
    selection=ensure_fifty_selection()
    c=json.loads((TARGET/'config.json').read_text())
    source_config=json.loads((SOURCE/'config.json').read_text())
    source_cache=SOURCE/'shared/generation_cache/baseline_live'
    original=json.loads((source_cache/'identity.json').read_text())
    assert original['sources']==source_hashes(),'Generation code changed; cannot transfer answers'
    assert original['model_adapter_hash']==file_hash(ROOT/'scripts/research/models.py')
    assert original['monitor']==source_config['monitor']
    allowed={'heldout_probe_n','heldout_selection'}
    assert {k:v for k,v in original['monitor'].items() if k not in allowed}=={k:v for k,v in c['monitor'].items() if k not in allowed}
    expected=read_rows(SOURCE_SELECTION/'heldout_probe.jsonl')+read_rows(SOURCE_SELECTION/'train_probe.jsonl')
    assert digest(expected)==original['rows_hash']
    target_rows=read_rows(selection/'heldout_probe.jsonl')+read_rows(selection/'train_probe.jsonl')
    assert target_rows[:50]==expected[:50]
    while True:
        done=sum((source_cache/f'{i:06d}.json').is_file() for i in range(50))
        write_json(LOG/'baseline_subset_progress.json',dict(phase='waiting_for_fixed50',completed=done,total=50,updated_unix=time.time()))
        if done==50:break
        d=SOURCE/'.session/attempts'/json.loads((SOURCE/'.session/current.json').read_text())['attempt']
        s=json.loads((d/'state.json').read_text())
        if s['status'] not in ('running','starting'):raise RuntimeError('Source baseline ended before first50 completed')
        time.sleep(1)
    # The user reduced the budget. Stop the old100-row worker at the50-row boundary.
    stop_session(SOURCE)
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        d=SOURCE/'.session/attempts'/json.loads((SOURCE/'.session/current.json').read_text())['attempt']
        if json.loads((d/'state.json').read_text())['status'] in ('stopped','completed','failed'):break
        time.sleep(.2)
    else:raise RuntimeError('Old100-row worker did not exit; refusing concurrent GPU jobs')
    destination=TARGET/'shared/generation_cache/baseline_live'
    destination.mkdir(parents=True,exist_ok=False)
    new_identity=copy.deepcopy(original)
    new_identity.update(rows_hash=digest(target_rows),monitor=c['monitor'])
    write_json(destination/'identity.json',new_identity)
    source_index={row['problem_id']:i for i,row in enumerate(expected)}
    transferred=[]
    for i,row in enumerate(target_rows):
        src=source_cache/f'{source_index[row["problem_id"]]:06d}.json'
        if not src.exists():continue
        saved=json.loads(src.read_text())
        assert saved['row_hash']==digest(row) and saved['result_hash']==digest(saved['result'])
        result=saved['result']
        assert result['source_optimizer_step']==0 and result['pool']=='live_rollout'
        assert result['decoding']['greedy']==c['monitor']['live_greedy']
        assert result['decoding']['max_new_tokens']==c['monitor']['behavior_max_response_tokens']
        assert result['seed']==int(digest([c['monitor']['seed'],row['problem_id'],'live_rollout'])[:8],16)
        os.link(src,destination/f'{i:06d}.json')
        transferred.append(dict(index=i,problem_id=row['problem_id'],source=str(src),sha256=file_hash(src)))
    assert len([r for r in transferred if r['index']<50])==50
    precheck=[json.loads((destination/f'{i:06d}.json').read_text())['result'] for i in range(50)]
    enforce_quality(precheck,c['monitor'],LOG/'quality_first50_precheck')
    report=dict(source_identity_sha256=file_hash(source_cache/'identity.json'),target_identity_sha256=file_hash(destination/'identity.json'),
        source_n=104,target_n=len(target_rows),heldout_n=50,transferred=transferred,
        generation_settings_unchanged=True,source_hashes_unchanged=True,selection_uses_outcomes=False,
        quality_precheck=str(LOG/'quality_first50_precheck/quality.json'))
    write_json(LOG/'baseline50_transfer.json',report)
    print('Fixed50 baseline passed; reused50 validated answers, old100-row worker stopped.',flush=True)
    return report
