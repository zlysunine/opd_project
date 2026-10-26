"""Ordered entry points for the Qwen3-1.7B / DeepMath OPSD experiment.

configure -> check -> prepare -> evaluate(initial) -> train -> plot -> merge -> evaluate(trained)
Each command does one explicit stage. It never downloads models or starts a matrix.
The run stage performs the full sequence; without --execute it only prints the plan.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .common import ROOT, digest, file_hash, opsd, read_jsonl, write_json, write_jsonl
from .data import load_questions, prepare_deepmath, partition_questions
from .launch import resolve, plan, validate_experiment, validate_dataset


def config_at(workdir):
    path=Path(workdir).resolve()/'config.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing; run opsd_pipeline.sh configure first')
    c=json.loads(path.read_text())
    validate_experiment(c)
    return c


def configure(a):
    work=Path(a.workdir).resolve()
    model=Path(a.model_dir).resolve()
    native=Path(a.input).resolve()
    if not model.is_dir() or not (model/'config.json').is_file():
        raise FileNotFoundError(f'Local model config.json missing: {model}')
    if not native.exists(): raise FileNotFoundError(f'Native DeepMath input missing: {native}')
    for name,value in [('model revision',a.model_revision),('dataset revision',a.dataset_revision)]:
        if not value or value in ('main','master') or value.startswith(('FILL_','YOUR_')):
            raise ValueError(f'Supply an immutable {name}')
    c=resolve(ROOT/'configs/research'/('paper.json' if a.profile=='paper' else 'pilot_1p7b.json'))
    c.update(dataset_revision=a.dataset_revision,registry=str(work/'models.json'),
             train_data=str(work/'data/solution/train.jsonl'),validation_data=str(work/'data/solution/validation.jsonl'),
             native_input=str(native),pipeline_workdir=str(work),seed=a.seed,data_seed=a.data_seed)
    for field in ('steps','gpus','batch','max_prompt_length','max_response_length','max_teacher_prompt_length'):
        value=getattr(a,field,None)
        if value is not None:
            if value<1: raise ValueError(f'{field} must be positive')
            c[field]=value
    for field in ('lora_rank','lora_alpha','loss_type'):
        if getattr(a,field,None) is not None: c[field]=getattr(a,field)
    if a.template: c['chat_template_path']=str(Path(a.template).resolve())
    registry={'qwen3_1p7b_nonthinking':{'model_id':'Qwen/Qwen3-1.7B','path':str(model),
               'revision':a.model_revision,'initialization':'posttrained_nonthinking','size':'1.7B'}}
    if (work/'config.json').exists():
        if (json.loads((work/'config.json').read_text())!=c or not (work/'models.json').is_file()
                or json.loads((work/'models.json').read_text())!=registry):
            raise ValueError('Workspace configuration/model differs; use a new --workdir to retain the previous run')
        print('Configuration already matches:',work/'config.json')
        return c
    validate_experiment(c)
    write_json(work/'models.json',registry)
    write_json(work/'config.json',c)
    print('Configured:',work/'config.json')
    return c


def tokenizer_for(c):
    from transformers import AutoTokenizer
    info=json.loads(Path(c['registry']).read_text())[c['student']]
    tokenizer=AutoTokenizer.from_pretrained(info['path'],local_files_only=True)
    if c.get('chat_template_path'): tokenizer.chat_template=Path(c['chat_template_path']).read_text()
    if not tokenizer.chat_template or tokenizer.pad_token_id is None:
        raise ValueError('Verified chat template and PAD token are required')
    return tokenizer


def source_files(path):
    p=Path(path)
    files=sorted(p.rglob('*.parquet')) if p.is_dir() else [p]
    if not files: raise ValueError('No parquet shards in native data directory')
    return files


def prepare(c, overlength='error', max_problems=None, tokenizer=None):
    work=Path(c['pipeline_workdir'])
    tokenizer=tokenizer or tokenizer_for(c)
    identities={str(p):file_hash(p) for p in source_files(c['native_input'])}
    identity={'config':c,'source_files':identities,'overlength':overlength,'max_problems':max_problems,
              'tokenizer_hash':digest(tokenizer.get_vocab()),'template_hash':digest(tokenizer.chat_template),
              'privilege_template_hash':digest(opsd.PRIVILEGE_TEMPLATE)}
    key=digest(identity)
    manifest=work/'data/prepare_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['cache_key']!=key:
            raise ValueError('Prepared data identity changed; use a new workspace')
        if any(not Path(path).exists() or file_hash(path)!=expected for path,expected in old['output_hashes'].items()):
            raise ValueError('Prepared data were modified or are incomplete; restore or use a new workspace')
        print('Prepared data cache verified:',manifest)
        return old
    started=time.monotonic()
    native=load_questions(c['native_input'])
    candidates=prepare_deepmath(native,work/'data/candidates.jsonl',c['dataset_revision'],
                               solution_field=c['reference_field'],tokenizer=tokenizer)
    eligible=[]
    lengths=[]
    kwargs={'tokenize':True,'add_generation_prompt':True,'enable_thinking':c['enable_thinking']}
    for row in candidates:
        sp=tokenizer.apply_chat_template(row['messages'],**kwargs)
        tp=tokenizer.apply_chat_template(opsd.privileged_messages(row['messages'],row['privilege_text']),**kwargs)
        valid=len(sp)<=c['max_prompt_length'] and len(tp)<=c['max_teacher_prompt_length']
        lengths.append({'problem_id':row['problem_id'],'difficulty':row['difficulty'],
                        'student_prompt_tokens':len(sp),'teacher_prompt_tokens':len(tp),
                        'reference_tokens':row['reference_token_length'],'length_eligible':valid})
        if valid: eligible.append(row)
    write_jsonl(work/'data/lengths.jsonl',lengths)
    if overlength=='error' and len(eligible)!=len(candidates):
        raise ValueError(f'{len(candidates)-len(eligible)} overlong prompts/references; increase lengths or explicitly use prepare --overlength drop. See data/lengths.jsonl')
    if not eligible: raise ValueError('No eligible questions; increase prompt/context limits')
    before_limit=len(eligible)
    if max_problems is not None:
        if max_problems<3: raise ValueError('--max-problems must allow train/validation/test')
        if len(eligible)>max_problems:
            eligible=random.Random(c.get('data_seed',42)).sample(sorted(eligible,key=lambda r:r['problem_id']),max_problems)
    empty=[{**row,'privilege_type':'empty','privilege_text':'','reference_token_length':0,
            'reference_character_length':0,'reference_source_field':None,'reference_correctness':None} for row in eligible]
    partition_questions(eligible,work/'data/solution',c.get('data_seed',42))
    partition_questions(empty,work/'data/empty',c.get('data_seed',42))
    train_count=len(read_jsonl(work/'data/solution/train.jsonl'))
    available=(train_count//c['batch'])*c['epochs']
    if c.get('steps') is not None and available<c['steps']:
        raise ValueError(f'Prepared train split has {train_count} questions, supports {available} complete updates; '
                         f'{c["steps"]} steps require at least {c["steps"]*c["batch"]} prompt visits. '
                         'Increase --max-problems/data or explicitly change steps in a new workspace')
    outputs=[work/f'data/{condition}/{split}.jsonl' for condition in ('solution','empty') for split in ('train','validation','test')]
    report={'cache_key':key,'identity':identity,'raw_questions':len(native),'level_5_7_questions':len(candidates),
            'length_eligible_questions':before_limit,'selected_questions':len(eligible),
            'level_coverage':len(candidates)/len(native),'length_coverage':before_limit/len(candidates),
            'train_questions':train_count,'complete_updates_available':available,
            'max_problems':max_problems,'selected_ids':sorted(r['problem_id'] for r in eligible),
            'prepare_wall_seconds':time.monotonic()-started,'output_hashes':{str(p):file_hash(p) for p in outputs}}
    write_json(manifest,report)
    print(json.dumps({k:report[k] for k in ('raw_questions','level_5_7_questions','length_eligible_questions','selected_questions')}))
    return report


def selected_config(c, condition):
    c=copy.deepcopy(c)
    if condition=='empty':
        c.update(privilege_type='empty',reference_source=None,
                 train_data=str(Path(c['pipeline_workdir'])/'data/empty/train.jsonl'),
                 validation_data=str(Path(c['pipeline_workdir'])/'data/empty/validation.jsonl'))
    return c


def train_directory(c, condition):
    return Path(c['pipeline_workdir'])/('train' if condition=='solution' else 'train_empty')


def checkpoint_actor(c, condition, step=None):
    checkpoints=train_directory(c,condition)/'checkpoints'
    tracker=checkpoints/'latest_checkpointed_iteration.txt'
    if not tracker.is_file(): raise FileNotFoundError(f'No completed checkpoint tracker: {tracker}')
    last=int(tracker.read_text().strip())
    step=last if step is None else step
    if step<1 or step>last: raise ValueError('Requested step is not a completed checkpoint')
    actor=checkpoints/f'global_step_{step}/actor'
    if not (actor/'huggingface/config.json').is_file(): raise FileNotFoundError(f'Missing HF config: {actor}')
    for rank in range(c['gpus']):
        if not (actor/f'model_world_size_{c["gpus"]}_rank_{rank}.pt').is_file():
            raise FileNotFoundError(f'Incomplete FSDP checkpoint at {actor}, rank {rank}')
    return step,actor


def merge(c, condition, step=None):
    step,actor=checkpoint_actor(c,condition,step)
    target=Path(c['pipeline_workdir'])/'hf'/condition/f'global_step_{step}'
    identity={'step':step,'source':str(actor),'shards':{p.name:file_hash(p) for p in sorted(actor.glob('model_*.pt'))},
              'training_mode':c.get('training_mode','full'),'lora_rank':c.get('lora_rank',0),
              'lora_alpha':c.get('lora_alpha')}
    marker=target/'merge_info.json'
    if marker.exists():
        if json.loads(marker.read_text())!=identity: raise ValueError('Merged checkpoint source changed')
        if c.get('training_mode')=='lora' and not (target/'lora_merge.json').is_file():
            raise ValueError('Merged LoRA checkpoint marker is missing; refusing base-only weights')
        if not list(target.glob('*.safetensors')) and not list(target.glob('pytorch_model*.bin')):
            raise ValueError('Merged checkpoint weights are missing')
        print('Verified merged checkpoint:',target)
        return target
    if target.exists() and any(target.iterdir()): raise ValueError('Incomplete merge directory exists; choose another workspace or inspect it first')
    command=[sys.executable,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(actor),'--target_dir',str(target)]
    subprocess.run(command,cwd=ROOT,check=True)
    if c.get('training_mode')=='lora':
        from .lora import finalize_lora_merge
        finalize_lora_merge(target,c)
    write_json(marker,identity)
    return target


def evaluate(c, condition, which='trained', split='validation', step=None, samples=16, max_tokens=None, device='cuda'):
    from .run import main as research_main
    info=json.loads(Path(c['registry']).read_text())[c['student']]
    if which=='initial': model,revision=info['path'],info['revision']
    else:
        step,_=checkpoint_actor(c,condition,step)
        model=str(Path(c['pipeline_workdir'])/'hf'/condition/f'global_step_{step}')
        if not (Path(model)/'merge_info.json').exists(): raise FileNotFoundError('Run pipeline merge before trained evaluation')
        if c.get('training_mode')=='lora' and not (Path(model)/'lora_merge.json').exists():
            raise ValueError('LoRA weights were not merged; refusing base-only evaluation')
        revision=f'local:{condition}:step{step}'
    output=Path(c['pipeline_workdir'])/'eval'/f'{condition}_{which}_{step if which=="trained" else 0}_{split}.jsonl'
    command=['evaluate','--model',str(model),'--revision',revision,
             '--input',str(Path(c['pipeline_workdir'])/f'data/solution/{split}.jsonl'),'--split',split,
             '--samples',str(samples),'--pass-k','1',str(samples),'--max-tokens',str(max_tokens or c['max_response_length']),
             '--temperature',str(c['sampling_temperature']),'--seed',str(c['seed']),
             '--device',device,'--dtype','bfloat16' if device.startswith('cuda') else 'float32','--output',str(output)]
    if c.get('chat_template_path'): command+=['--template',c['chat_template_path']]
    # Test scores must never enter the validation learning curve.
    if which=='trained' and split=='validation': command+=['--training-run',str(train_directory(c,condition)),'--step',str(step)]
    research_main(command)
    if which=='initial' and split=='validation':
        summary=json.loads(Path(str(output)+'.summary.json').read_text())
        initial_record={'step':0,'budget':{'optimized_tokens':0,'rollout_tokens':0,'teacher_forward_tokens':0,'wall_seconds':0,'updates':0,'questions':0},
                        'metrics':{'eval/'+k:v for k,v in summary.items() if isinstance(v,(float,int))},'split':'validation'}
        target=train_directory(c,condition)/'evaluations.jsonl'
        old=read_jsonl(target) if target.exists() else []
        write_jsonl(target,[initial_record]+[r for r in old if r['step']!=0])
    return output


def workflow_plan(a):
    """Build commands without preparing files, importing GPU dependencies or training."""
    if a.samples<1: raise ValueError('--samples must be positive')
    if a.max_problems is not None and a.max_problems<3:
        raise ValueError('--max-problems must allow train/validation/test')
    work=str(Path(a.workdir).resolve())
    effective=resolve(ROOT/'configs/research'/('paper.json' if a.profile=='paper' else 'pilot_1p7b.json'))
    for key in ('steps','batch','gpus','lora_rank','lora_alpha','loss_type',
                'max_prompt_length','max_response_length','max_teacher_prompt_length'):
        if getattr(a,key,None) is not None:effective[key]=getattr(a,key)
    validate_experiment(effective)
    commands=[['check','--workdir',work],
              ['prepare','--workdir',work,'--overlength',a.overlength]]
    if a.max_problems is not None: commands[-1]+=['--max-problems',str(a.max_problems)]
    for condition in (['solution','empty'] if a.with_empty else ['solution']):
        common=['--workdir',work,'--condition',condition]
        commands+=[['evaluate',*common,'--which','initial','--split','validation','--samples',str(a.samples)],
                   ['train',*common,'--execute'],['merge',*common],
                   ['evaluate',*common,'--which','trained','--split','validation','--samples',str(a.samples)],
                   ['plot',*common]]
        if a.with_test:
            commands.append(['evaluate',*common,'--which','trained','--split','test','--samples',str(a.samples)])
    return {'workdir':work,'profile':a.profile,'student':'Qwen/Qwen3-1.7B',
            'training':{key:effective[key] for key in ('training_mode','loss_type','lora_rank','lora_alpha',
                       'learning_rate','batch','rollouts','steps','gpus')},
            'self_teacher':'current_batch_stop_gradient; same actor plus offline r1_solution_1',
            'configure':{k:v for k,v in vars(a).items() if k not in
                         ('stage','execute','overlength','max_problems','samples','with_empty','with_test')},
            'stages':commands}


def run_workflow(a):
    workflow=workflow_plan(a)
    if not a.execute:
        print(json.dumps({'mode':'plan_only','workflow':workflow},ensure_ascii=False,indent=2))
        print('Plan only. Add --execute to configure and run all stages.')
        return
    c=configure(a)
    work=Path(c['pipeline_workdir'])
    journal=work/'workflow.json'
    record={'workflow':workflow,'status':'running','attempts':[]}
    if journal.exists():
        previous=json.loads(journal.read_text())
        if previous['workflow']!=workflow:
            raise ValueError('Workflow options changed; use a new --workdir')
        if previous['status']=='passed':
            raise ValueError('Workflow already completed; use individual stages to inspect it or a new --workdir')
        record['attempts']=previous['attempts']
    attempt={'status':'running','started_unix':time.time(),'stages':[]}
    record['attempts'].append(attempt)
    write_json(journal,record)
    try:
        for command in workflow['stages']:
            stage={'command':command,'status':'running','started_unix':time.time()}
            attempt['stages'].append(stage)
            write_json(journal,record)
            print('OPSD stage:', ' '.join(command),flush=True)
            main(command)
            stage.update(status='passed',finished_unix=time.time())
            write_json(journal,record)
    except BaseException as error:
        status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        stage.update(status=status,finished_unix=time.time(),error=f'{type(error).__name__}: {error}')
        record['status']=attempt['status']=status
        write_json(journal,record)
        raise
    record['status']=attempt['status']='passed'
    attempt['finished_unix']=time.time()
    write_json(journal,record)
    print('OPSD workflow completed:',work)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    for stage in ('run','configure','check','prepare','train','plot','merge','evaluate','smoke'):
        s=sub.add_parser(stage)
        s.add_argument('--workdir',default=os.environ.get('OPSD_WORKDIR','research_runs/consistency_lora_fkl_500'))
        if stage in ('configure','run'):
            s.add_argument('--model-dir',required=True);s.add_argument('--model-revision',required=True)
            s.add_argument('--input',required=True);s.add_argument('--dataset-revision',required=True)
            s.add_argument('--profile',choices=['pilot','paper'],default='pilot');s.add_argument('--seed',type=int,default=42)
            s.add_argument('--data-seed',type=int,default=42)
            s.add_argument('--template')
            s.add_argument('--lora-rank',type=int);s.add_argument('--lora-alpha',type=int)
            s.add_argument('--loss-type',choices=['forward_kl','reverse_kl'])
            for name in ('steps','gpus','batch','max-prompt-length','max-response-length','max-teacher-prompt-length'):
                s.add_argument('--'+name,type=int)
            if stage=='run':
                s.add_argument('--execute',action='store_true')
                s.add_argument('--overlength',choices=['error','drop'],default='error')
                s.add_argument('--max-problems',type=int)
                s.add_argument('--samples',type=int,default=16)
                s.add_argument('--with-empty',action='store_true')
                s.add_argument('--with-test',action='store_true')
        elif stage=='check': s.add_argument('--cpu-only',action='store_true')
        elif stage=='prepare':
            s.add_argument('--overlength',choices=['error','drop'],default='error');s.add_argument('--max-problems',type=int)
        elif stage=='train':
            s.add_argument('--execute',action='store_true');s.add_argument('--config')
        elif stage=='evaluate':
            s.add_argument('--which',choices=['initial','trained'],default='trained')
            s.add_argument('--split',choices=['validation','test'],default='validation')
            s.add_argument('--step',type=int);s.add_argument('--samples',type=int,default=16)
            s.add_argument('--max-tokens',type=int);s.add_argument('--device',default='cuda')
        elif stage=='merge': s.add_argument('--step',type=int)
        if stage in ('train','plot','merge','evaluate'):
            s.add_argument('--condition',choices=['solution','empty'],default='solution')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.stage=='run': run_workflow(a);return
    if a.stage=='configure': configure(a);return
    if a.stage=='smoke':
        from .smoke import smoke
        smoke(Path(a.workdir)/'smoke',resolve(ROOT/'configs/research/smoke.json'));return
    c=resolve(a.config) if a.stage=='train' and a.config else config_at(a.workdir)
    if a.stage=='check':
        from .preflight import check_environment
        result=check_environment(c['gpus'],not a.cpu_only)
        write_json(Path(a.workdir)/'preflight.json',result)
        print(json.dumps(result,indent=2))
        if not result['ok']: raise SystemExit(1)
        tokenizer_for(c)
        from .preflight import compose_training_config
        _,command=plan(c,Path(a.workdir).resolve()/'check_plan',False)
        compose_training_config(command,os.environ.copy(),Path(a.workdir).resolve()/'check_plan')
    elif a.stage=='prepare': prepare(c,a.overlength,a.max_problems)
    elif a.stage=='train':
        if a.condition=='empty' and not c.get('pipeline_workdir'):
            raise ValueError('--condition empty requires a configured pipeline workspace; run configure first')
        current=selected_config(c,a.condition) if c.get('pipeline_workdir') else c
        out=train_directory(c,a.condition) if c.get('pipeline_workdir') else None
        out,_=plan(current,out,a.execute)
        print('Run directory:',out)
        if not a.execute: print('Plan only. Add --execute to start this condition.')
    elif a.stage=='plot':
        from .plots import plot_directory
        source=train_directory(c,a.condition)
        if not (source/'metrics.jsonl').exists(): raise FileNotFoundError(f'No training metrics: {source}/metrics.jsonl')
        print('Figures:',len(plot_directory(source,source/'plots')))
    elif a.stage=='merge': print(merge(c,a.condition,a.step))
    elif a.stage=='evaluate': print(evaluate(c,a.condition,a.which,a.split,a.step,a.samples,a.max_tokens,a.device))


if __name__=='__main__': main()
