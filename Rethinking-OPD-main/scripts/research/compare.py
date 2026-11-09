"""OPSD training and checkpoint monitoring by default; explicit GRPO comparison remains available."""
import argparse
import copy
import json
from pathlib import Path
import sys

from .common import ROOT, file_hash, write_json
from .launch import plan, resolve, validate_experiment
from .job_control import record_progress


def configure(args):
    root = Path(args.workdir).resolve()
    c = json.loads(Path(args.config).read_text())
    model = Path(args.model_dir).resolve()
    native = Path(args.input).resolve()
    if not (model/'config.json').is_file() or not native.is_file():
        raise FileNotFoundError('Supply an existing local Base model directory and native DeepMath JSONL/Parquet file')
    for revision in (args.model_revision,args.dataset_revision):
        if not revision or revision in ('main','master') or revision.startswith(('FILL_','YOUR_')):
            raise ValueError('Explicit immutable model and dataset revisions required')
    if c.get('controlled_comparison'):
        for section,revision in [('model',args.model_revision),('dataset',args.dataset_revision)]:
            if c[section].get('revision')!=revision:
                raise ValueError(f'Controlled comparison must retain the locked {section} revision')
    model_config = json.loads((model/'config.json').read_text())
    if model_config.get('model_type')!='qwen3':
        raise ValueError('Production comparison requires an actual Qwen3 model config; tiny fixtures have a separate smoke CLI')
    declared = str(model_config.get('_name_or_path') or '')
    if 'Qwen3-1.7B' in declared and 'Base' not in declared:
        raise ValueError('Model config identifies the posttrained Qwen3-1.7B; supply Qwen3-1.7B-Base')
    c['model'].update(path=str(model),revision=args.model_revision)
    c['dataset'].update(input=str(native),revision=args.dataset_revision)
    if getattr(args,'reuse_probe',None):
        c['dataset']['frozen_probe_source'] = str(Path(args.reuse_probe).resolve())
    for name in ('steps','gpus','seed'):
        if getattr(args,name,None) is not None:
            c['training'][name] = getattr(args,name)
    for flag,section,field in [('train_n','dataset','train_n'),('dev_n','dataset','dev_n'),
                                ('probe_n','monitor','heldout_probe_n'),('train_probe_n','monitor','train_probe_n')]:
        if getattr(args,flag,None) is not None:
            c[section][field] = getattr(args,flag)
    if args.monitor_steps:
        c['monitor']['optimizer_steps'] = sorted({int(s) for s in args.monitor_steps.split(',')})
    steps = c['monitor']['optimizer_steps']
    if not steps or steps[0]!=0 or min(steps)<0 or max(steps)>c['training']['steps']:
        raise ValueError('Monitor steps must include 0 and remain within the configured training steps')
    if c.get('controlled_comparison'):
        from .controlled_data import validate_controlled
        c['monitor']['teacher_geometry_steps'] = steps
        c['monitor']['plot_full_rsm_steps'] = [0,c['training']['steps']]
        c['training']['save_every'] = min(c['training']['save_every'],c['training']['steps'])
        validate_controlled(c)
    if c['monitor'].get('heavy_metrics'):
        raise ValueError('Gride/ADP/ARI are not implemented in this P0/P1 delivery; leave heavy_metrics empty')
    c.update(workdir=str(root),output_root=str(root/'shared'),run_id=root.name)
    from analysis.common import load_config
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        write_json(Path(tmp)/'config.json',c)
        load_config(Path(tmp)/'config.json')
    if (root/'config.json').exists():
        if json.loads((root/'config.json').read_text())!=c:
            raise ValueError('Comparison configuration differs; use a new --workdir')
        return c
    write_json(root/'models.json',{'qwen3_1p7b_base':dict(model_id='Qwen/Qwen3-1.7B-Base',path=str(model),
               revision=args.model_revision,initialization='base',size='1.7B')})
    write_json(root/'config.json',c)
    return c


def load(workdir):
    return json.loads((Path(workdir).resolve()/'config.json').read_text())


def prepare(c):
    from analysis.prepare_probe import freeze
    available = (c['dataset']['train_n']//c['training']['batch'])*c['training']['epochs']
    if available < c['training']['steps']:
        raise ValueError(f'Not enough train questions: {available} complete updates available')
    if c.get('controlled_comparison'):
        from .controlled_data import prepare_shared
        probe = prepare_shared(c)
    else:
        probe = freeze(c)
    root = Path(c['workdir'])
    manifest = probe/'probe_manifest.jsonl'
    configs = {}
    for method in (('opsd','sft','grpo') if c.get('controlled_comparison') else ('opsd','grpo')):
        current = resolve(ROOT/c['training']['profile'])
        current.update(c['training'])
        current.pop('profile',None)
        current.update(group='exp_base_geometry',student='qwen3_1p7b_base',initialization='base',
                       prompt_mode='plain_completion',method={'grpo':'rl','sft':'sft','opsd':'opsd'}[method],
                       registry=str(root/'models.json'),dataset_revision=c['dataset']['revision'],
                       reference_field=c['dataset']['reference_field'],train_data=str(probe/('train_schedule.jsonl' if c.get('controlled_comparison') else 'train.jsonl')),
                       validation_data=str(probe/'validation.jsonl'),pipeline_workdir=str(root/method),
                       reference_source='deepmath_dataset' if method in ('opsd','sft') else None,
                       teacher_update_policy='current_batch_stop_gradient' if method=='opsd' else 'none',
                       loss_type={'opsd':'forward_kl','sft':'sft','grpo':'grpo'}[method],
                       privilege_type='solution' if method in ('opsd','sft') else 'empty')
        if c.get('controlled_comparison'):
            current.update(controlled_comparison=True,shuffle=False,
                           question_schedule=str(probe/'question_schedule.json'),
                           sft_target_repeats=c['training']['rollouts'],
                           notes=f"Controlled {method}: {c['training']['steps']} updates; shared prompts/LoRA/optimizer/probes/question order; method-specific supervision and compute.")
        analysis = copy.deepcopy(c)
        analysis.update(run_id=f'{root.name}__{method}',method=method,probe_root=str(probe),
                        output_root=str(root/method/'analysis'),training=copy.deepcopy(current))
        analysis['training'].pop('monitor',None)
        path = root/method/'analysis_config.json'
        current['monitor'] = dict(optimizer_steps=c['monitor']['optimizer_steps'],probe_manifest=str(manifest),
                                 probe_manifest_hash=file_hash(manifest),analysis_config=str(path),method=method,
                                 job_dir=str(root/method/'jobs'))
        if c.get('controlled_comparison'):
            current['monitor']['controlled_comparison']=True
        if current.get('checkpoint_retention')=='latest':
            current['monitor']['checkpoint_retention']='latest'
        validate_experiment(current)
        write_json(root/method/'training_config.json',current)
        write_json(path,analysis)
        configs[method] = current
    write_json(root/'design_manifest.json',dict(model=c['model'],dataset=c['dataset'],
               controlled_comparison=c.get('controlled_comparison',False),
               shared_training=c['training'],
               question_schedule_sha256=file_hash(probe/'question_schedule.json') if c.get('controlled_comparison') else None,
               shared_probe_manifest_hash=file_hash(manifest),shared_train_hash=file_hash(probe/'train.jsonl'),
               shared_validation_hash=file_hash(probe/'validation.jsonl'),
               methods={key:dict(loss=value['loss_type'],mode=value['training_mode'],teacher=value['teacher_update_policy'])
                        for key,value in configs.items()},
               budget_note='Same prompts, initialization, sampling counts and update caps; supervision and compute differ. Compare token and wall-time budgets too.'))
    return probe


def train(c,method,execute):
    root=Path(c['workdir'])
    current=json.loads((root/method/'training_config.json').read_text())
    if c.get('controlled_comparison'):
        entry=json.loads(Path(current['registry']).read_text())[current['student']]
        if Path(entry['path']).resolve()!=Path(c['model']['path']).resolve() or entry['revision']!=c['model']['revision']:
            raise ValueError('Controlled training registry differs from the frozen Base model')
        probe=root/'shared/probe'
        frozen=json.loads((probe/'complete.json').read_text())
        for name,expected_hash in frozen['files'].items():
            if file_hash(probe/name)!=expected_hash:
                raise ValueError(f'Controlled training input changed: {name}')
    if execute and c.get('monitor',{}).get('quality_gate',{}).get('enabled',False):
        from analysis.evaluate import enforce_quality
        from .common import read_jsonl
        probe=root/'shared/probe'
        marker=json.loads((probe/'complete.json').read_text())
        from analysis.common import source_hashes
        if marker['identity']['sources']!=source_hashes():
            raise ValueError('Prepared answer protocol or analysis sources changed; use a new run')
        baseline=probe/'baseline_live_rollouts.jsonl'
        if marker['files'].get(baseline.name)!=file_hash(baseline):
            raise ValueError('Baseline quality receipt missing or changed; prepare a new answer-v2 run')
        enforce_quality(read_jsonl(baseline),c['monitor'],root/'shared/quality/baseline')
    tracker=root/method/'train/checkpoints/latest_checkpointed_iteration.txt'
    if execute and tracker.is_file() and int(tracker.read_text().strip())>=current['steps']:
        from .pipeline import checkpoint_actor
        from .common import load_verl_utility
        step,actor=checkpoint_actor(current,'solution',current['steps'])
        # A previous process can exit between final checkpoint and job publication.
        # Reconstruct that job without performing update steps+1.
        if current.get('monitor') and (step in current['monitor']['optimizer_steps']
                                      or current['monitor'].get('checkpoint_retention')=='latest'):
            load_verl_utility('research_monitor').finish_checkpoint(current['monitor'],actor.parent,step)
        print(f'{method}: training already reached optimizer step {step}; reusing checkpoint')
        return str(root/method/'train'),None
    return plan(current,root/method/'train',execute)


def analyze(c,method,steps=None):
    from analysis.extract import extract
    from analysis.checkpoints import export_checkpoint, verify_initial_adapter
    from analysis.common import load_config
    from .retention import latest_only, completed_analysis, record_analysis, discard_export
    root=Path(c['workdir'])
    path=root/method/'analysis_config.json'
    analysis=load_config(path)
    current=json.loads((root/method/'training_config.json').read_text())
    jobs=sorted((root/method/'jobs').glob('step_*.json'))
    if not jobs:
        raise FileNotFoundError('No checkpoint jobs; run training first')
    selected=[p for p in jobs if steps is None or json.loads(p.read_text())['optimizer_step'] in steps]
    for job_path in selected:
        job=json.loads(job_path.read_text())
        if job['probe_manifest_hash']!=file_hash(Path(analysis['probe_root'])/'probe_manifest.jsonl'):
            raise ValueError('Queued job points to a different probe manifest')
        step=job['optimizer_step']
        try:
            if latest_only(current) and completed_analysis(job_path,path,analysis) is not None:
                if step:
                    discard_export(current,step)
                continue
            if step==0:
                initial=verify_initial_adapter(job['checkpoint_path'],current)
                for other_method in ('opsd','sft','grpo'):
                    other=root/other_method/'initial_adapter.json'
                    if other.is_file() and json.loads(other.read_text())!=initial:
                        raise ValueError('Initial LoRA tensors differ across methods; comparison is not paired')
                write_json(root/method/'initial_adapter.json',initial)
            # Verified zero-B LoRA gives exactly the archived Base function at step0.
            record_progress(c['workdir'],'exporting',method,step)
            checkpoint=c['model']['path'] if step==0 else str(export_checkpoint(current,step))
            revision=c['model']['revision'] if step==0 else f"{c['model']['revision']}:{method}:optimizer_step_{step}"
            record_progress(c['workdir'],'analyzing',method,step)
            result=extract(analysis,checkpoint,revision,step)
            if latest_only(current):
                record_analysis(job_path,path,result)
                if step:
                    discard_export(current,step)
            else:
                write_json(root/method/'job_status'/job_path.name,dict(status='complete',analysis_path=str(result)))
        except Exception as error:
            write_json(root/method/'job_status'/job_path.name,dict(status='failed',error=f'{type(error).__name__}: {error}'))
            raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    for stage in ('configure','prepare','train','analyze','report','run'):
        s=sub.add_parser(stage)
        s.add_argument('--workdir',default='research_runs/base_geometry_20_answer_v2_full_tokens_paired')
        if stage in ('configure','run'):
            s.add_argument('--config',default=str(ROOT/'configs/analysis_deepmath.json'))
            s.add_argument('--model-dir',required=True)
            s.add_argument('--model-revision',required=True)
            s.add_argument('--input',required=True)
            s.add_argument('--dataset-revision',required=True)
            s.add_argument('--reuse-probe',help='Controlled comparison: verify and adopt a server-side frozen probe without resampling')
            for name in ('steps','gpus','seed','train-n','dev-n','probe-n','train-probe-n'):
                s.add_argument('--'+name,type=int)
            s.add_argument('--monitor-steps')
        if stage in ('train','analyze','run'):
            s.add_argument('--method',choices=['opsd','sft','grpo','both','all'],default='opsd',
                           help='Default: opsd. Choose grpo or both explicitly to run the retained RL comparison.')
        if stage in ('train','run'):
            s.add_argument('--execute',action='store_true')
            s.add_argument('--foreground',action='store_true',
                           help='Run attached for debugging; --execute otherwise starts an SSH-independent background task')
        if stage=='analyze':
            s.add_argument('--steps',help='Comma-separated optimizer steps; step0 must be analyzed first')
    return p


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    args=parser().parse_args(arguments)
    method=getattr(args,'method','opsd')
    methods=('opsd','sft','grpo') if method=='all' else (('opsd','grpo') if method=='both' else (method,))
    c=configure(args) if args.stage in ('configure','run') else load(args.workdir)
    if ('sft' in methods) and not c.get('controlled_comparison'):
        raise ValueError('SFT in this pipeline requires configs/controlled_deepmath20.json')
    if args.stage in ('run','train') and args.execute and not args.foreground:
        from .job_control import start_session
        child_args=[args.stage,'--workdir',c['workdir'],'--method',method,'--execute']
        if args.stage=='run':
            for name in ('config','model_dir','model_revision','input','dataset_revision','steps','gpus','seed',
                         'train_n','dev_n','probe_n','train_probe_n','monitor_steps','reuse_probe'):
                value=getattr(args,name,None)
                if value is not None:
                    if name in ('config','model_dir','input'):
                        value=Path(value).resolve()
                    child_args.extend(['--'+name.replace('_','-'),str(value)])
        command=[sys.executable,'-u','-m','scripts.research.compare',*child_args,'--foreground']
        started=start_session(c['workdir'],command,methods)
        print(json.dumps(started,ensure_ascii=False,indent=2))
        if started['status']=='failed':
            raise SystemExit(1)
        return
    if args.stage=='configure':
        print('Configured:',Path(c['workdir'])/'config.json')
    elif args.stage=='prepare':
        print('Frozen probe:',prepare(c))
    elif args.stage=='report':
        from analysis.report import report
        print(report(c['workdir']))
        if c.get('controlled_comparison'):
            from .controlled_report import build
            print(build(c['workdir']))
    elif args.stage=='run' and not args.execute:
        print(json.dumps(dict(status='plan_only',model=c['model'],training=c['training'],
                   methods=list(methods),
                   stages=['prepare shared probes']+[f'train {method.upper()} -> analyze checkpoints' for method in methods]+['report'],
                   checkpoint_retention=c['training'].get('checkpoint_retention','all'),
                   analysis_schedule='after each save before rotation' if c['training'].get('checkpoint_retention')=='latest' else 'after training',
                   monitor_steps=c['monitor']['optimizer_steps']),indent=2))
    else:
        if args.stage=='run':
            from .preflight import check_environment
            record_progress(c['workdir'],'environment_check')
            status=check_environment(c['training']['gpus'])
            write_json(Path(c['workdir'])/'environment.json',status)
            if not status['ok']:
                raise RuntimeError('Environment check failed: '+'; '.join(status['errors']))
            record_progress(c['workdir'],'prepare')
            prepare(c)
        for method in methods:
            if args.stage in ('train','run'):
                record_progress(c['workdir'],'training',method)
                train(c,method,args.execute)
            if args.stage in ('analyze','run'):
                record_progress(c['workdir'],'analysis_verification',method)
                selected=None if args.stage=='run' or not args.steps else {int(v) for v in args.steps.split(',')}
                analyze(c,method,selected)
        if args.stage=='run':
            from analysis.report import report
            record_progress(c['workdir'],'report')
            print(report(c['workdir']))
            if c.get('controlled_comparison'):
                from .controlled_report import build
                print(build(c['workdir']))


if __name__=='__main__':
    main()
