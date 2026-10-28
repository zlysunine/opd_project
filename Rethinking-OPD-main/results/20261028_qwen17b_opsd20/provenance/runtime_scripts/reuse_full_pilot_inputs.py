"""Reuse audited pilot probes and step0 answers; never generate them again."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoConfig, AutoTokenizer
from analysis.common import source_hashes
from analysis.prepare_probe import freeze, normalized_question, split_rows, stratified_take
from scripts.research.common import checkpoint_identity, digest, file_hash, opsd, read_jsonl, write_json, write_jsonl
from scripts.research.compare import configure, load, prepare
from scripts.research.preparation_cache import save_selection

ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
LOG=ROOT/'research_runs/server_handoff_20261028'
PILOT=ROOT/'research_runs/base_opsd_20261028_acceptance_2step_q70_n50'
FORMAL=ROOT/'research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814'
CONFIG=LOG/'formal_config_20step_reuse_pilot.json'
CACHE=LOG/'formal_selection_reuse_pilot'


class CachedOnlyModel:
    """Authenticated Base metadata for importing already generated responses."""
    def __init__(self,c,identity):
        self.identity=identity
        self.tokenizer=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
        self.model=SimpleNamespace(config=AutoConfig.from_pretrained(c['model']['path'],local_files_only=True))
        assert digest(self.tokenizer.get_vocab())==identity['tokenizer_hash']
    def prompt(self,row,privileged=False):
        messages=opsd.plain_messages(row['question'])
        if privileged:messages=opsd.privileged_messages(messages,row['privilege_text'],'plain_completion')
        return self.tokenizer.encode(messages[0]['content'],add_special_tokens=False)
    def layers(self):return range(self.model.config.num_hidden_layers)
    def generate(self,*args,**kwargs):raise RuntimeError('Unexpected generation: must reuse pilot inputs')
    def generate_batch(self,*args,**kwargs):raise RuntimeError('Unexpected generation: must reuse pilot inputs')


def prepare_formal_from_pilot():
    source_config=load(PILOT)
    # Sync the visible template only after the pilot is fully accepted, avoiding
    # changes to its training implementation identity while it is still active.
    accepted=LOG/'acceptance_verification.json'
    if accepted.is_file():
        verdict=json.loads(accepted.read_text())
        if verdict.get('workdir')==str(PILOT) and verdict.get('verified_steps')==[0,1,2] and verdict.get('quality_gates_passed'):
            for template in [ROOT/'configs/analysis_deepmath.json',LOG/'formal_template_20step.json']:
                visible=json.loads(template.read_text())
                visible['monitor'].update(heldout_probe_n=source_config['monitor']['heldout_probe_n'],
                                          train_probe_n=source_config['monitor']['train_probe_n'])
                write_json(template,visible)
    probe=PILOT/'shared/probe'
    receipt=json.loads((probe/'complete.json').read_text())
    assert receipt['identity']['config_hash']==digest(source_config)
    assert receipt['identity']['sources']==source_hashes()
    assert receipt['identity']['input_hash']==file_hash(source_config['dataset']['input'])
    assert receipt['identity']['checkpoint_hash']==checkpoint_identity(source_config['model']['path'])
    for name,expected in receipt['files'].items():assert file_hash(probe/name)==expected,name
    source_model=json.loads((probe/'manifest.json').read_text())['model']
    assert source_model['content_hash']==receipt['identity']['checkpoint_hash']
    held=read_jsonl(probe/'heldout_probe.jsonl');train_probe=read_jsonl(probe/'train_probe.jsonl')
    old_train=read_jsonl(probe/'train.jsonl');old_val=read_jsonl(probe/'validation.jsonl')
    assert len(held)==50 and len(train_probe)==4 and len(old_train)==16
    raw=read_jsonl(probe/'eligible.jsonl')
    checks={row['problem_id']:row for row in read_jsonl(probe/'eligibility.jsonl')}
    accepted=[row for row in raw if checks[row['problem_id']]['exclusion_reason'] is None]
    unique_n=len({digest(normalized_question(row['question']))[:20] for row in accepted})
    all_config=copy.deepcopy(source_config)
    all_config['dataset'].update(train_n=unique_n-2,dev_n=1)
    all_config['monitor'].update(heldout_probe_n=1,train_probe_n=1)
    all_splits,duplicates=split_rows(accepted,all_config)
    pool=all_splits['heldout_probe']+all_splits['validation']+all_splits['train']
    protected={row['problem_id'] for row in held+old_train+old_val}
    remaining=[row for row in pool if row['problem_id'] not in protected]
    config=json.loads((LOG/'formal_template_20step.json').read_text())
    config['dataset']['train_n']=2814
    config['monitor'].update(heldout_probe_n=len(held),train_probe_n=len(train_probe),
        heldout_selection=copy.deepcopy(source_config['monitor']['heldout_selection']))
    config['monitor']['reuse_pilot_inputs']=dict(source_probe_sha256=file_hash(probe/'complete.json'),
        policy='identical heldout/train_probe/fixed_pool/baseline; larger training subset from the same screened pool')
    config['dataset']['preparation_cache']=str(CACHE)
    val_extra,remaining=stratified_take(remaining,config['dataset']['dev_n']-len(old_val),config['monitor']['seed']+1)
    train_extra,unused=stratified_take(remaining,config['dataset']['train_n']-len(old_train),config['monitor']['seed']+2)
    splits=dict(heldout_probe=held,train_probe=train_probe,
        validation=[{**row,'split':'validation'} for row in old_val+val_extra],
        train=[{**row,'split':'train'} for row in old_train+train_extra])
    for name in splits:splits[name].sort(key=lambda row:(row['topic_label'],row['difficulty'],row['problem_id']))
    assert splits['heldout_probe']==held and splits['train_probe']==train_probe
    write_json(CONFIG,config)
    args=SimpleNamespace(workdir=str(FORMAL),config=str(CONFIG),model_dir=source_config['model']['path'],
        model_revision=source_config['model']['revision'],input=source_config['dataset']['input'],
        dataset_revision=source_config['dataset']['revision'],monitor_steps=None)
    c=configure(args)
    if not CACHE.exists():
        stage=LOG/'formal_reuse_selection_stage';stage.mkdir(exist_ok=False)
        for name in ['eligible.jsonl','eligible.jsonl.manifest.json','eligibility.jsonl','answer_encoding_quarantine.jsonl']:
            os.link(probe/name,stage/name)
        write_jsonl(stage/'duplicates.jsonl',duplicates)
        for name,rows in splits.items():write_jsonl(stage/f'{name}.jsonl',rows)
        save_selection(stage,c,CACHE)
    records=[]
    target_rows=held+train_probe
    for pool_name,source_name,frozen_name in [('live_rollout','baseline_live','baseline_live_rollouts.jsonl'),
                                               ('fixed_base_rollout','fixed_base_rollout','fixed_pool.jsonl')]:
        source= PILOT/'shared/generation_cache'/source_name
        original=json.loads((source/'identity.json').read_text())
        assert original['sources']==source_hashes() and original['model']==source_model
        assert original['rows_hash']==digest(target_rows) and original['model_adapter_hash']==file_hash(ROOT/'scripts/research/models.py')
        different={k for k in set(original['monitor'])|set(c['monitor']) if original['monitor'].get(k)!=c['monitor'].get(k)}
        assert different <= {'optimizer_steps','teacher_geometry_steps','plot_full_rsm_steps','reuse_pilot_inputs'},different
        destination=FORMAL/'shared/generation_cache'/source_name
        new_identity=copy.deepcopy(original);new_identity['monitor']=c['monitor']
        frozen=read_jsonl(probe/frozen_name)
        if destination.exists():
            assert json.loads((destination/'identity.json').read_text())==new_identity
        else:
            destination.mkdir(parents=True)
            write_json(destination/'identity.json',new_identity)
        for i,row in enumerate(target_rows):
            src=source/f'{i:06d}.json';saved=json.loads(src.read_text())
            assert saved['row_hash']==digest(row) and saved['result_hash']==digest(saved['result'])
            assert all(frozen[i].get(k)==v for k,v in saved['result'].items())
            dst=destination/f'{i:06d}.json'
            if not dst.exists():os.link(src,dst)
            assert file_hash(dst)==file_hash(src)
        records.append(dict(pool=pool_name,answers_reused=len(frozen),source=str(source)))
    # All per-question generation receipts must exist. Missing inputs raise rather
    # than loading weights or producing replacement responses.
    freeze(c,model=CachedOnlyModel(c,source_model))
    prepare(c)  # Official config publication; freeze now verifies its completed receipt.
    for name in ['heldout_probe.jsonl','train_probe.jsonl','fixed_pool.jsonl','baseline_live_rollouts.jsonl']:
        assert file_hash(FORMAL/'shared/probe'/name)==file_hash(probe/name),name
    result=dict(formal=str(FORMAL),config=str(CONFIG),source_probe=str(probe),
        source_probe_sha256=file_hash(probe/'complete.json'),counts={name:len(rows) for name,rows in splits.items()},
        screened_unique=unique_n,unused_screened_candidates=len(unused),generation_calls=0,
        exact_files_reused=['heldout_probe.jsonl','train_probe.jsonl','fixed_pool.jsonl','baseline_live_rollouts.jsonl'],
        larger_training_subset_preserves_pilot_train_ids=True,validation_preserves_pilot_validation_ids=True,
        no_probe_training_leakage=True,pools=records)
    write_json(LOG/'formal_full_input_reuse.json',result)
    return result
