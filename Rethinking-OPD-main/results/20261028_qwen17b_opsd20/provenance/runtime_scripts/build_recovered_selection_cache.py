"""Import preserved, audited eligibility into a newly configured selection cache."""
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from analysis.prepare_probe import split_rows
from scripts.research.common import file_hash, checkpoint_identity, opsd, write_json, write_jsonl
from scripts.research.answer_protocol import answer_metadata
from scripts.research.preparation_cache import save_selection

ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
LOG=ROOT/'research_runs/server_handoff_20261028'
SOURCE=ROOT/'research_runs/base_opsd_20261028_acceptance_2step'
AUDIT=SOURCE/'shared/preparation_audit'
TARGET=ROOT/'research_runs/base_opsd_20261028_acceptance_2step_q70_n100'
c=json.loads((TARGET/'config.json').read_text())
old=json.loads((SOURCE/'config.json').read_text())
recovered=json.loads((AUDIT/'accepted_pool.recovery_manifest.json').read_text())
identity=json.loads((SOURCE/'shared/generation_cache/baseline_live/identity.json').read_text())
for name in ['scripts/research/answer_protocol.py','scripts/val/eval/utils.py','verl/verl/utils/opsd.py']:
    assert file_hash(ROOT/name)==identity['sources'][name],f'Scientific answer/prompt rules changed: {name}'
for name in ['max_prompt_length','max_teacher_prompt_length','max_response_length']:
    assert c['training'][name]==old['training'][name]
assert c['model']==old['model']
for name in ['input','revision','answer_protocol','reference_field','difficulty_min','difficulty_max','require_verified_reference','topic_level']:
    assert c['dataset'][name]==old['dataset'][name]
assert file_hash(c['dataset']['input'])==recovered['native_sha256']
assert checkpoint_identity(c['model']['path'])==identity['model']['content_hash']
assert file_hash(AUDIT/'eligibility.jsonl')==recovered['decision_sha256']
assert file_hash(AUDIT/'accepted_pool.recovered.jsonl')==recovered['recovered_sha256']
checks={r['problem_id']:r for r in map(json.loads,(AUDIT/'eligibility.jsonl').open())}
rows=[json.loads(line) for line in (AUDIT/'accepted_pool.recovered.jsonl').open()]
assert {r['problem_id'] for r in rows}=={pid for pid,r in checks.items() if r['exclusion_reason'] is None}
tokenizer=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
accepted=[]
for start in range(0,len(rows),64):
    batch=rows[start:start+64]
    student=[opsd.plain_messages(r['question'])[0]['content'] for r in batch]
    teacher=[opsd.privileged_messages(opsd.plain_messages(r['question']),r['privilege_text'],'plain_completion')[0]['content'] for r in batch]
    sp=tokenizer(student,add_special_tokens=False)['input_ids']
    tp=tokenizer(teacher,add_special_tokens=False)['input_ids']
    for row,p,q in zip(batch,sp,tp):
        check=checks[row['problem_id']]
        assert check['reference_verified'] and len(p)==check['student_tokens'] and len(q)==check['teacher_tokens']
        assert len(p)<=c['training']['max_prompt_length'] and len(q)<=c['training']['max_teacher_prompt_length']
        meta=answer_metadata(row);assert meta['answer_encoding_status']=='ok'
        accepted.append({**row,**meta,'reference_correctness':True,'messages':opsd.plain_messages(row['question'])})
    print('Validated recovered candidates:',min(start+64,len(rows)),'/',len(rows),flush=True)
splits,duplicates=split_rows(accepted,c)
stage=LOG/'recovered_selection_stage_q70_n100';stage.mkdir(exist_ok=False)
os.link(AUDIT/'accepted_pool.recovered.jsonl',stage/'eligible.jsonl')
write_json(stage/'eligible.jsonl.manifest.json',dict(dataset_id='zwhe99/DeepMath-103K',dataset_revision=c['dataset']['revision'],
    original_raw_count=103022,original_difficulty_selected_count=64942,after=len(accepted),
    scope='Recovered accepted candidates; all original decisions remain in eligibility.jsonl',
    source_manifest=str(AUDIT/'accepted_pool.recovery_manifest.json')))
for name in ['eligibility.jsonl','answer_encoding_quarantine.jsonl']:os.link(AUDIT/name,stage/name)
write_jsonl(stage/'duplicates.jsonl',duplicates)
for name,values in splits.items():write_jsonl(stage/f'{name}.jsonl',values)
cache=Path(c['dataset']['preparation_cache'])
save_selection(stage,c,cache)
write_json(cache/'recovery_provenance.json',dict(source_decision_sha256=recovered['decision_sha256'],
    source_pool_sha256=recovered['recovered_sha256'],source_model_identity=identity['model'],
    source_protocol_hashes={name:identity['sources'][name] for name in ['scripts/research/answer_protocol.py','scripts/val/eval/utils.py','verl/verl/utils/opsd.py']},
    all_accepted_ids_preserved=True,all_accepted_prompt_lengths_revalidated=True,
    references_reverified=False,reused_reference_verification=True,
    quality_threshold_changed_by_user=True,heldout_count_changed_by_user=True,
    pilot_generated_answers_reused=False,
    counts={name:len(values) for name,values in splits.items()}))
print('Selection cache ready:',cache,flush=True)
