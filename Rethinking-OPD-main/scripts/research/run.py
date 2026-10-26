"""Unified CLI. Run from repository root: python -m scripts.research.run --help."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch

from .common import ROOT, digest, provenance, read_jsonl, write_json, write_jsonl, seed_all
from .data import (canonical, controlled_dataset, load_questions, reference_intersection, export_verl, verify)
from .distributions import distribution_metrics, aggregate, evaluate_samples, gap_recovery
from .models import Model, trajectory_ids


def model_args(p):
    p.add_argument('--model', required=True, help='Existing local HF checkpoint directory')
    p.add_argument('--revision', required=True, help='Immutable source commit or explicit local checkpoint version')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','bfloat16','float16'], default='float32')
    p.add_argument('--template', help='Verified chat-template file, required if checkpoint lacks one')
    p.add_argument('--thinking', action='store_true')


def get_model(a):
    return Model(a.model,a.revision,a.device,a.dtype,a.template,a.thinking)


def generate_references(a):
    rows = load_questions(a.input,a.split)
    model = None if a.type=='empty' or a.oracle else get_model(a)
    if model:
        generator = {**model.identity,'model_id':a.generator_id or a.model}
    else:
        generator = {'path':'controlled_oracle' if a.oracle else None,'revision':'algorithm_v1' if a.oracle else None}
    config = {'type':a.type,'generator':generator,'seed':a.seed,'max_tokens':a.max_tokens,
              'temperature':a.temperature,'top_p':a.top_p,'data_hash':digest(rows),'oracle':a.oracle}
    cache_key = digest(config)
    previous = {}
    if Path(a.output).exists():
        previous = {r['problem_id']:r for r in read_jsonl(a.output)}
        if any(r.get('reference_cache_key') != cache_key for r in previous.values()):
            raise ValueError('Reference cache identity mismatch; use a new output')
    results = []
    for row in rows:
        if row['problem_id'] in previous:
            results.append(previous[row['problem_id']])
            continue
        row_seed = int(digest([a.seed,row['problem_id']])[:8],16)
        seed_all(row_seed)
        if a.type=='empty':
            text,correct,truncated = '',None,False
        elif a.oracle:
            if row.get('benchmark')!='controlled_modular' or 'oracle_solution' not in row:
                raise ValueError('--oracle is restricted to generated controlled-skill data')
            if a.type=='hint':
                text = 'Reduce modulo the modulus after every operation.'
            elif a.type=='final_answer':
                text = '\\boxed{'+row['ground_truth']+'}'
            else:
                text = row['oracle_solution']
            correct,truncated = (None if a.type=='hint' else True),False
        else:
            request = copy.deepcopy(row)
            instruction = {'hint':' Give a useful hint without the final answer.',
                           'final_answer':' Give only the final answer in \\boxed{}.',
                           'solution':' Provide a worked solution and final answer in \\boxed{}.'}[a.type]
            request['messages'][-1]['content'] += instruction
            ids,text,truncated = model.generate(request,a.max_tokens,a.temperature,a.top_p)
            correct = None if a.type=='hint' else verify(text,row['ground_truth'],row['benchmark'])
        length = len((model.tokenizer if model else __import__('scripts.research.models',fromlist=['ByteTokenizer']).ByteTokenizer()).encode(text))
        item = {**row,'privilege_text':text,'privilege_type':a.type,
                'generator_model':generator.get('model_id',generator['path']),
                'generator_revision':generator['revision'],'generation_seed':row_seed,
                'reference_correctness':correct,'reference_token_length':length,
                'reference_length_tokenizer':model.identity['tokenizer_hash'] if model else 'byte_smoke',
                'reference_truncated':truncated,'generation_config':config,'reference_cache_key':cache_key}
        results.append(item)
        # Atomic incremental cache; resumable without repeating completed questions.
        write_jsonl(a.output,results)
    write_json(str(a.output)+'.manifest.json',provenance(config,[a.input]))
    return results


def evaluate(a):
    seed_all(a.seed)
    model = get_model(a)
    rows = load_questions(a.input,a.split)
    if a.limit: rows = rows[:a.limit]
    output = []
    for row in rows:
        for sample in range(a.samples):
            seed = int(digest([a.seed,row['problem_id'],sample])[:8],16)
            seed_all(seed)
            ids,text,truncated = model.generate(row,a.max_tokens,a.temperature,a.top_p,a.privileged)
            output.append({**row,'sample_index':sample,'response_ids':ids,'response_length':len(ids),'response':text,
                           'correct':verify(text,row['ground_truth'],row['benchmark']), 'truncated':truncated,
                           'seed':seed, 'tokenizer_hash':model.identity['tokenizer_hash'], 'checkpoint':model.identity,
                           'trajectory_source':'fixed_diagnostic' if a.fixed else 'on_policy_evaluation',
                           'privileged':a.privileged,'sampling_temperature':a.temperature,'top_p':a.top_p,
                           'optimized_tokens':0,'prefix_hash':digest([row['messages'],ids])})
    write_jsonl(a.output,output)
    summary = evaluate_samples(output,tuple(a.pass_k))
    summary['controlled_skill_calibration'] = 'all_incorrect_adjust_difficulty' if summary['avg_accuracy']==0 else ('all_correct_adjust_difficulty' if summary['avg_accuracy']==1 else 'nondegenerate')
    write_json(str(a.output)+'.summary.json',summary)
    if a.training_run:
        if a.step is None:
            raise ValueError('--training-run requires --step for checkpoint accounting')
        training = [r for r in read_jsonl(Path(a.training_run)/'metrics.jsonl') if r['step']==a.step]
        if not training:
            raise ValueError('No matching training step for evaluation')
        base = training[-1]
        namespace = 'eval_privileged/' if a.privileged else 'eval/'
        entry = {'step':a.step,'budget':base['budget'], 'checkpoint':model.identity,
                 'split':a.split,'metrics':{namespace+k:v for k,v in summary.items() if isinstance(v,(int,float))}}
        target = Path(a.training_run)/'evaluations.jsonl'
        with target.open('a') as f: f.write(json.dumps(entry)+'\n')
    write_json(str(a.output)+'.manifest.json',provenance({**vars(a),'model_identity':model.identity},[a.input]))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_question','benchmarks')},ensure_ascii=False))
    return output


def diagnose(a):
    from .common import opsd
    student = get_model(a)
    teacher = student if a.teacher_model is None else Model(a.teacher_model,a.teacher_revision,a.device,a.dtype,a.teacher_template,a.teacher_thinking)
    opsd.assert_tokenizer_compatible(student.tokenizer,teacher.tokenizer,student.model.config.vocab_size,teacher.model.config.vocab_size)
    rows = read_jsonl(a.input)
    values = []
    for row in rows[:a.limit or len(rows)]:
        response = trajectory_ids(row,student.tokenizer)
        if row.get('tokenizer_hash') != student.identity['tokenizer_hash']:
            raise ValueError('Diagnostic prefix tokenizer mismatch')
        sp, tp = student.prompt(row), teacher.prompt(row,a.privileged)
        # Hidden forward + chunked LM head: never store/log [batch,sequence,vocab].
        with torch.no_grad():
            def states(m,prompt):
                batch = m.pack(prompt,response)
                hidden = m.model.base_model(**{k:v for k,v in batch.items() if k!='responses'},use_cache=False).last_hidden_state
                return hidden[0,len(prompt)-1:len(prompt)-1+len(response)]
            sh,th = states(student,sp), states(teacher,tp)
            for start in range(0,len(response),a.chunk):
                stop = min(start+a.chunk,len(response))
                sl = student.model.get_output_embeddings()(sh[start:stop]).float()
                tl = teacher.model.get_output_embeddings()(th[start:stop]).float()
                metrics = distribution_metrics(sl,tl,a.ks,a.diagnostic_temperature)
                pl,ql = (sl/a.diagnostic_temperature).log_softmax(-1),(tl/a.diagnostic_temperature).log_softmax(-1)
                labels = torch.tensor(response[start:stop],device=sl.device)[:,None]
                metrics['kl/sampled_token_estimate_p_to_q'] = (pl.gather(-1,labels)-ql.gather(-1,labels)).squeeze(-1).cpu().numpy()
                for name, array in metrics.items():
                    for offset,value in enumerate(array):
                        index = start+offset
                        values.append({'problem_id':row['problem_id'],'sample_index':row.get('sample_index',0),
                            'token_index':index,'relative_bucket':min(a.bins-1,index*a.bins//len(response)),
                            'metric':name,'value':float(value) if math.isfinite(value) else None,
                            'diagnostic_temperature':a.diagnostic_temperature,
                            'sampling_temperature':row.get('sampling_temperature'),
                            'student_checkpoint_hash':student.identity['content_hash'],
                            'teacher_checkpoint_hash':teacher.identity['content_hash'],
                            'teacher_privileged':a.privileged,
                            'teacher_role':'external_teacher' if a.teacher_model else ('opsd_self_teacher' if a.privileged else 'unprivileged_self_control'),
                            'prefix_hash':digest(response[:index])})
    write_jsonl(a.output,values)
    write_json(str(a.output)+'.summary.json',aggregate(values))
    write_json(str(a.output)+'.manifest.json',provenance({**vars(a),'student':student.identity,'teacher':teacher.identity},[a.input]))
    return values


def select_checkpoint(a):
    """Select validation-only checkpoints nearest target accuracy under a budget cap."""
    rows = read_jsonl(a.input)
    eligible = []
    for row in rows:
        if row.get('split') != 'validation':
            raise ValueError('Checkpoint selection input must contain validation results only')
        if a.max_tokens is None or row['optimized_tokens'] <= a.max_tokens:
            eligible.append(row)
    if not eligible:
        raise ValueError('No eligible validation checkpoint')
    # Tie-breaking does not inspect test scores.
    chosen = min(eligible,key=lambda r:(abs(r['accuracy']-a.target),r['optimized_tokens'],r['step']))
    write_json(a.output,{'selected':chosen,'target_accuracy':a.target,'selection_split':'validation',
                         'absolute_gap':abs(chosen['accuracy']-a.target),'within_tolerance':abs(chosen['accuracy']-a.target)<=a.tolerance})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command',required=True)
    s = subs.add_parser('prepare')
    s.add_argument('--output',required=True); s.add_argument('--input'); s.add_argument('--split')
    s.add_argument('--count',type=int,default=64); s.add_argument('--seed',type=int,default=42)
    s.add_argument('--modulus',type=int,default=11); s.add_argument('--steps',type=int,default=3)
    s = subs.add_parser('prepare-deepmath')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True)
    s.add_argument('--revision',required=True); s.add_argument('--level-min',type=float,default=5.)
    s.add_argument('--level-max',type=float,default=7.); s.add_argument('--solution-field',default='r1_solution_1')
    s.add_argument('--privilege-type',choices=['solution','final_answer','empty'],default='solution')
    s.add_argument('--tokenizer',help='Optional local tokenizer for reference token lengths')
    s = subs.add_parser('partition')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True)
    s.add_argument('--seed',type=int,default=42); s.add_argument('--validation-fraction',type=float,default=.1)
    s.add_argument('--test-fraction',type=float,default=.1); s.add_argument('--group-by',default='instance_id')
    s = subs.add_parser('reference')
    # Model optional only for empty/oracle; missing regular model fails explicitly in get_model.
    s.add_argument('--model'); s.add_argument('--revision'); s.add_argument('--device',default='cpu')
    s.add_argument('--dtype',default='float32'); s.add_argument('--template'); s.add_argument('--thinking',action='store_true')
    s.add_argument('--generator-id'); s.add_argument('--oracle',action='store_true')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True); s.add_argument('--split')
    s.add_argument('--type',choices=['empty','solution','hint','final_answer'],default='solution')
    s.add_argument('--seed',type=int,default=42); s.add_argument('--max-tokens',type=int,default=1024)
    s.add_argument('--temperature',type=float,default=1.); s.add_argument('--top-p',type=float,default=1.)
    s = subs.add_parser('shuffle-reference')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True); s.add_argument('--seed',type=int,default=42)
    s.add_argument('--type',choices=['shuffled_reference','mismatched_reference'],default='shuffled_reference')
    s = subs.add_parser('intersect')
    s.add_argument('--inputs',nargs='+',required=True); s.add_argument('--output',required=True); s.add_argument('--correct-only',action='store_true')
    s = subs.add_parser('export')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True); s.add_argument('--split')
    s = subs.add_parser('train')
    s.add_argument('--config',required=True); s.add_argument('--output'); s.add_argument('--execute',action='store_true')
    s.add_argument('--method',choices=['opd','opsd','rl','sft','baseline']); s.add_argument('--external-teacher')
    s.add_argument('--student'); s.add_argument('--dataset-revision'); s.add_argument('--train-data'); s.add_argument('--validation-data')
    s = subs.add_parser('matrix')
    s.add_argument('--config',required=True); s.add_argument('--output',required=True)
    s.add_argument('--groups',nargs='+',choices=['exp1_opsd_consistency','exp2'],default=['exp1_opsd_consistency','exp2']); s.add_argument('--seeds',nargs='+',type=int)
    for command in ('evaluate','extract','diagnose'):
        s = subs.add_parser(command); model_args(s)
        s.add_argument('--input',required=True); s.add_argument('--output',required=True)
        s.add_argument('--privileged',action='store_true'); s.add_argument('--limit',type=int)
        if command=='evaluate':
            s.add_argument('--samples',type=int,default=16); s.add_argument('--max-tokens',type=int,default=7168)
            s.add_argument('--temperature',type=float,default=1.); s.add_argument('--top-p',type=float,default=1.)
            s.add_argument('--seed',type=int,default=42); s.add_argument('--pass-k',type=int,nargs='+',default=[1,16])
            s.add_argument('--split'); s.add_argument('--fixed',action='store_true')
            s.add_argument('--training-run'); s.add_argument('--step',type=int)
        elif command=='diagnose':
            s.add_argument('--teacher-model'); s.add_argument('--teacher-revision'); s.add_argument('--teacher-template')
            s.add_argument('--teacher-thinking',action='store_true')
            s.add_argument('--diagnostic-temperature',type=float,default=1.)
            s.add_argument('--ks',type=int,nargs='+',default=[16]); s.add_argument('--chunk',type=int,default=16)
            s.add_argument('--bins',type=int,default=5)
        else:
            s.add_argument('--layers',type=int,nargs='+',required=True)
            s.add_argument('--hook',choices=['attention_output','block_output','post_mlp'],default='block_output')
            s.add_argument('--positions',type=int,nargs='+',default=[0])
            s.add_argument('--strategy',choices=['last_prompt_token','fixed','relative','annotated'],default='relative')
            s.add_argument('--bins',type=int,default=5); s.add_argument('--cap',type=int,default=1024)
            s.add_argument('--semantics',choices=['predict_current','after_current'],default='predict_current')
    s = subs.add_parser('analyze')
    s.add_argument('--a',required=True); s.add_argument('--b',required=True); s.add_argument('--output',required=True)
    s.add_argument('--distance',choices=['cosine','euclidean'],default='cosine')
    s.add_argument('--neighbors',type=int,default=5); s.add_argument('--cap',type=int,default=2048)
    s.add_argument('--clusters',type=int,default=4); s.add_argument('--seed',type=int,default=42)
    s.add_argument('--baseline-a'); s.add_argument('--baseline-b'); s.add_argument('--rank',type=int,default=2)
    s.add_argument('--centered-pca',action='store_true'); s.add_argument('--cross-scale',action='store_true'); s.add_argument('--probe',action='store_true')
    s = subs.add_parser('select-checkpoint')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True); s.add_argument('--target',type=float,required=True)
    s.add_argument('--max-tokens',type=int); s.add_argument('--tolerance',type=float,default=0.02)
    s = subs.add_parser('gap-recovery')
    s.add_argument('--before',type=float,required=True); s.add_argument('--after',type=float,required=True)
    s.add_argument('--teacher',type=float,required=True); s.add_argument('--teacher-checkpoint',required=True)
    s.add_argument('--privileged',action='store_true'); s.add_argument('--output',required=True)
    s = subs.add_parser('plot')
    s.add_argument('--input',required=True); s.add_argument('--output',required=True)
    s = subs.add_parser('smoke')
    s.add_argument('--output',required=True); s.add_argument('--config',default='configs/research/smoke.json')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command=='prepare':
        if a.input: write_jsonl(a.output,load_questions(a.input,a.split))
        else: controlled_dataset(a.output,a.count,a.seed,a.modulus,a.steps)
    elif a.command=='prepare-deepmath':
        from .data import prepare_deepmath
        tokenizer=None
        if a.tokenizer:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        prepare_deepmath(load_questions(a.input),a.output,a.revision,a.level_min,a.level_max,
                         a.privilege_type,a.solution_field,tokenizer)
    elif a.command=='partition':
        from .data import partition_questions
        partition_questions(load_questions(a.input),a.output,a.seed,a.validation_fraction,a.test_fraction,a.group_by)
    elif a.command=='reference': generate_references(a)
    elif a.command=='shuffle-reference':
        rows = load_questions(a.input)
        # Shuffle within split only: references cannot cross the data split boundary.
        groups = {}
        for row in rows: groups.setdefault(row['split'],[]).append(row)
        rng = random.Random(a.seed)
        output = []
        for group in groups.values():
            if len(group)<2: raise ValueError('Shuffled reference requires at least 2 rows per split')
            order = list(range(len(group))); rng.shuffle(order)
            targets = order[1:]+order[:1]
            for i,j in zip(order,targets):
                source,target = group[i],group[j]
                output.append({**source,'privilege_text':target['privilege_text'],'privilege_type':a.type,
                               'reference_problem_id':target['problem_id'],'reference_token_length':target['reference_token_length'],
                               'reference_correctness':verify(target['privilege_text'],source['ground_truth'],source['benchmark']),
                               'shuffle_seed':a.seed})
        write_jsonl(a.output,output)
    elif a.command=='intersect': reference_intersection(a.inputs,a.output,a.correct_only)
    elif a.command=='export': export_verl(load_questions(a.input,a.split),a.output)
    elif a.command=='train':
        from .launch import resolve,plan
        c = resolve(a.config)
        for key in ('method','external_teacher','student','train_data','validation_data','dataset_revision'):
            if getattr(a,key,None) is not None: c[key] = getattr(a,key)
        if a.method and a.method!='opsd':
            c['teacher_update_policy'] = 'frozen_external' if a.method=='opd' else 'none'
            if a.method in ('rl','baseline'):
                c['reference_source']=None; c['privilege_type']='empty'
            elif a.method=='opd':
                c['reference_source']=c.get('external_teacher'); c['privilege_type']='empty'
        out,command = plan(c,a.output,a.execute)
        print('Run artifacts:',out)
        if not a.execute: print('Plan only; add --execute to launch one run. Command saved in plan_command.sh')
    elif a.command=='matrix':
        from .launch import resolve,matrix
        print('Planned runs:',len(matrix(resolve(a.config),a.output,tuple(a.groups),a.seeds)))
    elif a.command=='evaluate': evaluate(a)
    elif a.command=='diagnose': diagnose(a)
    elif a.command=='extract':
        from .artifacts import extract
        model = get_model(a)
        rows = read_jsonl(a.input)
        if a.limit: rows = rows[:a.limit]
        extract(model,rows,a.output,a.layers,a.hook,a.privileged,a.strategy,tuple(a.positions),a.bins,a.semantics,a.cap)
    elif a.command=='analyze':
        from .artifacts import analyze
        analyze(a.a,a.b,a.output,a.neighbors,a.cap,a.clusters,a.seed,a.baseline_a,a.baseline_b,a.rank,a.centered_pca,a.cross_scale,a.probe,a.distance)
    elif a.command=='select-checkpoint': select_checkpoint(a)
    elif a.command=='gap-recovery':
        write_json(a.output,gap_recovery(a.before,a.after,a.teacher,teacher_checkpoint=a.teacher_checkpoint,privileged=a.privileged))
    elif a.command=='plot':
        from .plots import plot_directory
        plot_directory(a.input,a.output)
    elif a.command=='smoke':
        from .smoke import smoke
        smoke(a.output,json.loads(Path(a.config).read_text()))


if __name__=='__main__':
    main()
