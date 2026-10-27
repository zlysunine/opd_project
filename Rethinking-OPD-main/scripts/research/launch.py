"""Config-driven launch of existing verl OPD/GRPO and LLaMA-Factory SFT."""
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from .common import ROOT, digest, file_hash, provenance, write_json, write_jsonl
from .data import load_questions, export_verl


def hydra_value(value):
    """Quote values, but keep dictionary keys in Hydra's override grammar."""
    if isinstance(value,dict):
        if any(not isinstance(key,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',key) for key in value):
            raise ValueError('Hydra dictionary keys must be identifiers')
        return '{'+','.join(f'{key}:{hydra_value(item)}' for key,item in value.items())+'}'
    if isinstance(value,(list,tuple)):
        return '['+','.join(hydra_value(item) for item in value)+']'
    return json.dumps(value)


def resolve(path, stack=()):
    path = Path(path)
    if str(path.resolve()) in stack:
        raise ValueError('Config inheritance cycle')
    value = json.loads(path.read_text())
    base = resolve(ROOT/value['extends'], stack+(str(path.resolve()),)) if 'extends' in value else {}
    base.update({k:v for k,v in value.items() if k!='extends'})
    return base


def experiment_id(c):
    identity = '__'.join(str(c.get(k, 'none') or 'none') for k in
        ('group','size','initialization','method','reference_source','privilege_type','teacher_update_policy','seed'))
    if c['method']=='opsd' and c.get('loss_type')=='forward_kl':
        identity+=f'__{c.get("training_mode","full")}_r{c.get("lora_rank",0)}__forward_kl_full_vocab'
    return identity


def registry_entry(c, key, required=True):
    registry = json.loads((ROOT/c['registry']).read_text())
    value = registry[key] if key else {}
    if required and (not value.get('path') or not value.get('revision') or value.get('revision') in ('main','master')):
        raise ValueError(f'Model {key}: fill path and immutable revision in {c["registry"]}. ' + value.get('unavailable_reason',''))
    return value


def hydra_options(c, out, student, teacher=None):
    method = c['method']
    if method not in ('opd','opsd','rl'):
        raise ValueError(method)
    length = c['max_prompt_length']+c['max_response_length']
    options = {
        'algorithm.adv_estimator': 'grpo' if method=='rl' else 'token_reward_direct',
        'algorithm.use_kl_in_reward': False,
        'data.train_files': str(out/'train.parquet'), 'data.val_files': str(out/'validation.parquet'),
        'data.train_batch_size': c['batch'], 'data.shuffle': True, 'data.seed': c['seed'],
        'data.max_prompt_length':c['max_prompt_length'], 'data.max_response_length':c['max_response_length'],
        'data.filter_overlong_prompts':False, 'data.truncation':'error', 'data.return_raw_chat':True,
        '+data.apply_chat_template_kwargs.enable_thinking':c['enable_thinking'],
        'actor_rollout_ref.model.path':student,
        'actor_rollout_ref.model.use_remove_padding':True,
        'actor_rollout_ref.model.enable_gradient_checkpointing':c['activation_checkpointing'],
        'actor_rollout_ref.model.enable_activation_offload':True,
        'actor_rollout_ref.actor.optim.lr':c['learning_rate'],
        'actor_rollout_ref.actor.ppo_mini_batch_size':c['batch'],
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':c['microbatch'],
        'actor_rollout_ref.actor.use_dynamic_bsz':True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':length,
        'actor_rollout_ref.actor.use_kl_loss':False,
        'actor_rollout_ref.actor.loss_agg_mode':'token-mean',
        'actor_rollout_ref.actor.ppo_epochs':1,
        'actor_rollout_ref.actor.fsdp_config.model_dtype':'fp32',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload':c['optimizer_offload'],
        'actor_rollout_ref.actor.use_torch_compile':False,
        'actor_rollout_ref.actor.checkpoint.save_contents':['model','optimizer','extra'],
        'actor_rollout_ref.rollout.name':'vllm', 'actor_rollout_ref.rollout.temperature':c['sampling_temperature'],
        'actor_rollout_ref.rollout.top_p':1., 'actor_rollout_ref.rollout.n':c['rollouts'],
        'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.gpu_memory_utilization':0.25,
        'actor_rollout_ref.rollout.enforce_eager':True,
        'actor_rollout_ref.rollout.max_num_seqs':max(8,c['batch']*c['rollouts']),
        'actor_rollout_ref.rollout.max_model_len':length,
        'actor_rollout_ref.rollout.max_num_batched_tokens':length,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz':True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu':length,
        'actor_rollout_ref.rollout.calculate_log_probs':True,
        '+actor_rollout_ref.rollout.log_prob_top_k':0 if method=='rl' else c['top_k'],
        '+actor_rollout_ref.rollout.top_k_strategy':'only_stu',
        '+actor_rollout_ref.rollout.reward_weight_mode':'student_p',
        '+actor_rollout_ref.rollout.teacher_temperature':c.get('teacher_temperature',c['sampling_temperature']),
        '+actor_rollout_ref.rollout.seed':c['seed'],
        'reward_model.enable':method=='opd', 'reward_model.enable_resource_pool':False,
        'reward_model.model.path': teacher or student, 'reward_model.model.input_tokenizer':None,
        'reward_model.model.use_remove_padding':True,
        'reward_model.micro_batch_size_per_gpu':1, 'reward_model.use_dynamic_bsz':True,
        'reward_model.forward_max_token_len_per_gpu':length,
        'custom_reward_function.path':str(ROOT/'scripts/research/reward.py'),
        'custom_reward_function.name':'compute_score',
        'trainer.logger':['console'], 'trainer.project_name':'opsd_mechanisms',
        'trainer.experiment_name':experiment_id(c), 'trainer.n_gpus_per_node':c['gpus'], 'trainer.nnodes':1,
        'trainer.total_epochs':c['epochs'], 'trainer.save_freq':c['save_every'], 'trainer.test_freq':-1,
        'trainer.critic_warmup':0, 'trainer.val_before_train':False, 'trainer.is_plot':False, 'trainer.resume_mode':'auto',
        'trainer.default_local_dir':str(out/'checkpoints'),
        '+trainer.log_opd_dynamic_metrics':method!='rl' and c['top_k']>0,
        '+trainer.research_required_metrics':c.get('required_training_metrics',[]) if method=='opsd' else [],
        '+trainer.research_metrics_path':str(out/'metrics.jsonl'),
        '+trainer.research_budget':{k:c.get('budget_'+k) for k in ('optimized_tokens','rollout_tokens','wall_seconds')},
        '+ray_kwargs.ray_init.address':'local',
        '+ray_kwargs.ray_init._temp_dir':'/tmp/opsd-'+digest(str(out))[:10],
    }
    if c.get('steps') is not None:
        options['trainer.total_training_steps'] = c['steps']
    if method in ('opsd','rl') and c.get('training_mode')=='lora':
        options.update({'actor_rollout_ref.model.lora_rank':c['lora_rank'],
                        'actor_rollout_ref.model.lora_alpha':c['lora_alpha'],
                        'actor_rollout_ref.model.target_modules':c.get('lora_target_modules','all-linear'),
                        'actor_rollout_ref.model.lora_adapter_path':None})
    if method=='opsd':
        if c['teacher_update_policy'] != 'current_batch_stop_gradient':
            raise ValueError('Implemented OPSD update policy: current_batch_stop_gradient only')
        options['+opsd'] = {'enable':True, 'teacher_update_policy':c['teacher_update_policy'],
            'max_teacher_prompt_length':c['max_teacher_prompt_length'],
            'loss_type':c.get('loss_type','reverse_kl'),
            'forward_kl_chunk_size':c.get('forward_kl_chunk_size',16),
            'prompt_mode':c.get('prompt_mode','chat'),
            'chat_template_kwargs':{} if c.get('prompt_mode')=='plain_completion' else {'enable_thinking':c['enable_thinking']}}
        if c.get('loss_type')=='forward_kl':
            options['actor_rollout_ref.model.use_fused_kernels']=False
    if c.get('prompt_mode')=='plain_completion':
        del options['+data.apply_chat_template_kwargs.enable_thinking']
    if c.get('group')=='exp_base_geometry':
        options['actor_rollout_ref.model.lora_init_seed']=c['seed']
    if c.get('monitor'):
        monitor=copy.deepcopy(c['monitor'])
        if c.get('checkpoint_retention')=='latest':
            # Worker rotation deletes before saving. Rotate whole checkpoints only
            # after the driver has saved, logged and successfully analyzed them.
            options['trainer.max_actor_ckpt_to_keep']=None
            options['trainer.max_critic_ckpt_to_keep']=None
            monitor['checkpoint_retention']='latest'
            monitor['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
        options['+trainer.research_monitor']=monitor
    return options


def reference_source_info(c):
    if c.get('reference_source')=='deepmath_dataset':
        return {'source_type':'dataset_reference','dataset_id':c['dataset_id'],
                'revision':c.get('dataset_revision'),'reference_field':c.get('reference_field','r1_solution_1')}
    return registry_entry(c,c.get('reference_source'),False) if c.get('reference_source') else None


def validate_experiment(c):
    if c.get('checkpoint_retention','all') not in ('all','latest'):
        raise ValueError('checkpoint_retention must be all or latest')
    if c.get('group')=='exp_base_geometry':
        if (c['student']!='qwen3_1p7b_base' or c['initialization']!='base'
                or c.get('prompt_mode')!='plain_completion' or c['method'] not in ('opsd','rl')):
            raise ValueError('Base geometry comparison requires Qwen3-1.7B-Base, plain completion and OPSD/GRPO')
        if c['method']=='opsd' and (c.get('loss_type')!='forward_kl'
                or c['teacher_update_policy']!='current_batch_stop_gradient'):
            raise ValueError('Base monitoring preserves full-vocabulary forward KL and current-batch teacher')
        if c.get('external_teacher') or c.get('training_mode')!='lora' or c.get('lora_rank',0)<1:
            raise ValueError('Base comparison requires LoRA and no external teacher')
    if c['method']=='opsd':
        if c.get('loss_type','reverse_kl') not in ('reverse_kl','forward_kl'):
            raise ValueError('Supported OPSD losses: reverse_kl, forward_kl')
        if c.get('training_mode','full') not in ('full','lora'):
            raise ValueError('Supported OPSD training modes: full, lora')
        if c.get('training_mode')=='lora' and (c.get('lora_rank',0)<1 or c.get('lora_alpha',0)<=0):
            raise ValueError('LoRA rank and alpha must be positive')
        if c.get('loss_type')=='forward_kl' and c.get('forward_kl_chunk_size',16)<1:
            raise ValueError('Forward KL chunk size must be positive')
    if c['group']=='exp1_3p2':
        raise ValueError('Section 3.2 migration is cancelled; choose exp1_opsd_consistency or exp2')
    if c['group']!='exp1_opsd_consistency': return
    if c['student']!='qwen3_1p7b_nonthinking' or c['size']!='1.7B':
        raise ValueError('Experiment 1 uses Qwen3-1.7B for both student and privileged self-teacher')
    if c['method'] not in ('baseline','opsd') or c.get('external_teacher'):
        raise ValueError('Experiment 1 supports baseline and OPSD only; no external logits teacher')
    if c['method']=='opsd' and (c['teacher_update_policy']!='current_batch_stop_gradient' or c['top_k']<1):
        raise ValueError('Experiment 1 requires current self-teacher and top-k dynamic diagnostics')
    if c.get('dataset_id')!='zwhe99/DeepMath-103K' or (c.get('difficulty_min'),c.get('difficulty_max'))!=(5.,7.):
        raise ValueError('Experiment 1 requires DeepMath-103K, inclusive difficulty 5 through 7')


def validate_dataset(rows, c):
    import math
    if not rows: raise ValueError('Empty training dataset')
    if c['group'] in ('exp1_opsd_consistency','exp_base_geometry'):
        if not c.get('dataset_revision') or c['dataset_revision'] in ('main','master'):
            raise ValueError('Fill dataset_revision with the immutable DeepMath source revision')
        for row in rows:
            value=row.get('difficulty')
            if (row.get('dataset_id')!=c['dataset_id'] or row.get('dataset_revision')!=c['dataset_revision']
                    or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                    or not 5<=value<=7):
                raise ValueError('DeepMath identity/revision/difficulty mismatch; run prepare-deepmath for levels 5-7')
    if c['method']=='opsd':
        expected=reference_source_info(c) or {}
        for row in rows:
            if row['privilege_type']!=c['privilege_type']:
                raise ValueError('Configured privilege type differs from dataset')
            if row['privilege_type']=='empty': continue
            if expected.get('source_type')=='dataset_reference':
                if (row.get('dataset_id')!=expected['dataset_id'] or row.get('dataset_revision')!=expected['revision']
                        or row.get('reference_source_field')!=expected['reference_field']):
                    raise ValueError('DeepMath reference source/field/revision mismatch')
            else:
                if row['generator_model'] not in (expected.get('model_id'),expected.get('path')):
                    raise ValueError('Reference generator identity differs from experiment config')
                if not expected.get('revision') or row['generator_revision']!=expected['revision']:
                    raise ValueError('Reference generator revision differs from registered fixed revision')


def plan(c, output=None, execute=False):
    validate_experiment(c)
    out = Path(output or ROOT/c['output']/experiment_id(c)).resolve()
    student = registry_entry(c,c['student'],required=execute)
    teacher = registry_entry(c,c.get('external_teacher'),required=execute and c['method']=='opd')
    student_path = student.get('path') or f'<SET_LOCAL_PATH:{c["student"]}>'
    teacher_path = teacher.get('path')
    environment_report=None
    input_lengths=[]
    if execute:
        if c['method'] in ('opd','opsd','rl'):
            from .preflight import check_environment
            environment_report=check_environment(c['gpus'])
            if not environment_report['ok']:
                write_json(out/'preflight.json',environment_report)
                raise RuntimeError('Training environment check failed: '+ '; '.join(environment_report['errors']))
        from .models import Model
        from .common import opsd
        if not Path(student_path).is_dir():
            raise FileNotFoundError(student_path)
        # Validate tokenizer/architecture offline before Ray allocates resources.
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(student_path,local_files_only=True)
        template_path = c.get('chat_template_path') or student.get('chat_template_path')
        if template_path:
            tokenizer.chat_template = (ROOT/template_path).read_text()
        if not tokenizer.chat_template:
            raise ValueError('Missing verified student chat template')
        if teacher_path:
            tt = AutoTokenizer.from_pretrained(teacher_path,local_files_only=True)
            opsd.assert_tokenizer_compatible(tokenizer,tt,AutoConfig.from_pretrained(student_path).vocab_size,
                                             AutoConfig.from_pretrained(teacher_path).vocab_size)
        train = load_questions(ROOT/c['train_data'])
        val = load_questions(ROOT/c['validation_data'])
        validate_dataset(train,c)
        if c['group'] in ('exp1_opsd_consistency','exp_base_geometry'): validate_dataset(val,c)
        if c['method']!='baseline' and (any(r['split']!='train' for r in train) or any(r['split']!='validation' for r in val)):
            raise ValueError('Training requires train split; checkpoint selection requires a separate validation split')
        if {r['problem_id'] for r in train} & {r['problem_id'] for r in val}:
            raise ValueError('Training/validation problem leakage')
        if c['method'] in ('opd','opsd','rl'):
            from .preflight import validate_training_inputs
            architecture=AutoConfig.from_pretrained(student_path,local_files_only=True)
            input_lengths=validate_training_inputs(c,train,val,tokenizer,architecture.max_position_embeddings)
        files = [ROOT/c['train_data'],ROOT/c['validation_data']]
        if template_path: files.append(ROOT/template_path)
    else:
        files = []
    out.mkdir(parents=True,exist_ok=True)
    if environment_report is not None: write_json(out/'preflight.json',environment_report)
    if input_lengths: write_jsonl(out/'input_lengths.jsonl',input_lengths)
    resolved = {**c, 'experiment_id':experiment_id(c), 'student_model':student,
                'external_teacher_model':teacher if c['method']=='opd' else None,
                'privilege_generator_model':registry_entry(c,c.get('reference_source'),False) if c.get('reference_source') and c['reference_source']!='deepmath_dataset' and c['method'] in ('opsd','sft') else None,
                'privilege_source':reference_source_info(c) if c['method'] in ('opsd','sft') else None}
    resolved['supervision'] = {
        'opsd':('unprivileged student rollout; full-vocabulary KL(stopgrad(privileged current actor) || student)'
                if c.get('loss_type')=='forward_kl' else
                'unprivileged student rollout; offline reference; current student stop-gradient logits'),
        'opd':'unprivileged student rollout; frozen external teacher logits',
        'sft':'offline reference text as assistant targets, response-only labels',
        'rl':'unprivileged student rollout and verifiable outcome reward',
        'baseline':'no parameter update'}[c['method']]
    if execute:
        resolved['distinct_training_questions'] = len(train)
        resolved['validation_questions'] = len(val)
        resolved['data_hashes'] = {str(p):file_hash(p) for p in files}
        from .common import checkpoint_identity
        resolved['student_checkpoint_hash'] = checkpoint_identity(student_path)
        if teacher_path:
            resolved['external_teacher_checkpoint_hash'] = checkpoint_identity(teacher_path)
    from .common import opsd
    resolved['privilege_template_hash'] = digest(opsd.PRIVILEGE_TEMPLATE) if c['method']=='opsd' else None
    resolved['implementation_hash'] = digest(provenance(c)['source_hashes'])
    resolved_path = out/('resolved_config.json' if execute else 'plan_config.json')
    if execute and resolved_path.exists():
        previous = json.loads(resolved_path.read_text())
        if previous != resolved:
            raise ValueError('Run directory config changed; use a new output directory')
    write_json(resolved_path,resolved)
    write_json(out/('provenance.json' if execute else 'plan_provenance.json'),provenance(resolved,files))
    if c['method']=='sft':
        if execute:
            targets = []
            for r in train:
                if not r['privilege_text']:
                    raise ValueError('SFT requires nonempty targets')
                full = tokenizer.apply_chat_template(r['messages']+[{'role':'assistant','content':r['privilege_text']}], tokenize=True, add_generation_prompt=False, enable_thinking=c['enable_thinking'])
                if len(full)>c['max_prompt_length']+c['max_response_length']:
                    raise ValueError('SFT target exceeds configured cutoff; curate common-length data explicitly')
                targets.append({'messages':r['messages']+[{'role':'assistant','content':r['privilege_text']}]})
            write_json(out/'sft.json',targets)
        write_json(out/'dataset_info.json',{'research':{'file_name':'sft.json','formatting':'sharegpt',
                  'columns':{'messages':'messages'}, 'tags':{'role_tag':'role','content_tag':'content',
                  'user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}})
        if c['batch'] % (c['microbatch']*c['gpus']):
            raise ValueError('SFT global batch must divide microbatch * gpus exactly')
        cfg = {'model_name_or_path':student_path,'stage':'sft','do_train':True,'finetuning_type':'full',
               'dataset':'research','dataset_dir':str(out),'template':'qwen3','enable_thinking':c['enable_thinking'],
               'cutoff_len':c['max_prompt_length']+c['max_response_length'],
               'output_dir':str(out/'checkpoints'),'per_device_train_batch_size':c['microbatch'],
               'gradient_accumulation_steps':max(1,c['batch']//(c['microbatch']*c['gpus'])),
               'learning_rate':c['learning_rate'],'num_train_epochs':c['epochs'],
               'gradient_checkpointing':c['activation_checkpointing'],'bf16':True,
               'save_steps':c['save_every'],'logging_steps':1,'report_to':'none','overwrite_output_dir':False,
               'include_num_input_tokens_seen':True,'save_only_model':False}
        if c.get('steps') is not None: cfg['max_steps']=c['steps']
        write_json(out/'sft_config.json',cfg)
        if c.get('resume_sft_checkpoint'):
            cfg['resume_from_checkpoint'] = c['resume_sft_checkpoint']
            write_json(out/'sft_config.json',cfg)
        prefix = [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={c["gpus"]}','-m'] if c['gpus']>1 else [sys.executable,'-m']
        command = prefix+['scripts.research.sft_train',str(out/'sft_config.json'),str(out)]
    elif c['method'] in ('opd','opsd','rl'):
        opts = hydra_options(c,out,student_path,teacher_path)
        template_path = c.get('chat_template_path') or student.get('chat_template_path')
        if template_path:
            opts['actor_rollout_ref.model.custom_chat_template'] = (ROOT/template_path).read_text()
        if c['method']=='opd' and not teacher_path:
            opts['reward_model.model.path'] = '<SET_EXTERNAL_TEACHER_PATH>'
        command = [sys.executable,'-m','verl.trainer.main_ppo']+[f'{key}={hydra_value(value)}' for key,value in opts.items()]
        if execute:
            export_verl(train,out/'train.parquet')
            export_verl(val,out/'validation.parquet')
    elif c['method']=='baseline':
        command = [sys.executable,'-m','scripts.research.run','evaluate','--model',student_path,
                   '--revision',student.get('revision') or '<SET_REVISION>','--input',str(ROOT/c['validation_data']),
                   '--output',str(out/'evaluation.jsonl'),'--samples','16']
    else:
        raise ValueError(c['method'])
    write_json(out/('command.json' if execute else 'plan_command.json'),command)
    (out/('command.sh' if execute else 'plan_command.sh')).write_text(shlex.join(command)+'\n')
    if execute:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'verl'),str(ROOT/'LlamaFactory/src'),env.get('PYTHONPATH','')])
        if c['method'] in ('opd','opsd','rl'):
            from .preflight import compose_training_config
            write_json(out/'config_preflight.json',compose_training_config(command,env,out))
        print('Training log:',out/'train.log',flush=True)
        started = time.monotonic()
        with (out/'train.log').open('a') as log:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(out/'execution_summary.json',{'returncode':result.returncode,'process_wall_seconds':time.monotonic()-started,
                   'checkpoint_directory':str(out/'checkpoints'),'command':command})
        if result.returncode:
            raise RuntimeError(f'Training exited {result.returncode}; see {out}/train.log')
        if c.get('plot_after_training') and (out/'metrics.jsonl').exists():
            from .plots import plot_directory
            plot_directory(out,out/'plots')
    return str(out),command


def matrix(base, output, groups=('exp1_opsd_consistency','exp2'), seeds=None):
    if any(group not in ('exp1_opsd_consistency','exp2') for group in groups):
        raise ValueError('Supported groups: exp1_opsd_consistency, exp2; section 3.2 migration is removed')
    out=Path(output)
    records=[]
    for group in groups:
        scales=[('1.7B','1p7b')] if group=='exp1_opsd_consistency' else [('1.7B','1p7b'),('4B','4b')]
        for size,short in scales:
            conditions=[('baseline',None,'empty','none'),('opsd',None,'empty','current_batch_stop_gradient')]
            if group=='exp1_opsd_consistency':
                conditions += [('opsd','deepmath_dataset','solution','current_batch_stop_gradient')]
            else:
                conditions += [('sft','qwen3_4b_nonthinking','solution','none'),('rl',None,'empty','none'),
                               ('opsd','qwen3_4b_nonthinking','solution','current_batch_stop_gradient')]
            for method,source,privilege,policy in conditions:
                for seed in seeds or base.get('seeds',[base['seed']]):
                    c={**base,'group':group,'size':size,'initialization':'nonthinking',
                       'student':f'qwen3_{short}_nonthinking','method':method,'reference_source':source,
                       'external_teacher':None,'privilege_type':privilege,'teacher_update_policy':policy,'seed':seed}
                    if method=='rl': c['rollouts']=max(2,c['rollouts'])
                    if group=='exp1_opsd_consistency':
                        c['train_data']=f'research_data/deepmath_l5_7/{privilege}/train.jsonl'
                        c['validation_data']=f'research_data/deepmath_l5_7/{privilege}/validation.jsonl'
                    else:
                        c.update(training_mode='full',lora_rank=0,loss_type='reverse_kl')
                        source_name=source if privilege!='empty' else 'empty'
                        c['train_data']=f'research_data/references/{source_name}_train.jsonl'
                        c['validation_data']=f'research_data/references/{source_name}_validation.jsonl'
                    validate_experiment(c)
                    name=experiment_id(c)
                    path=out/f'{name}.json'
                    write_json(path,c)
                    command=[sys.executable,'-m','scripts.research.run','train','--config',str(path),'--execute']
                    records.append({'experiment_id':name,'config':str(path),'command':shlex.join(command),
                        'external_teacher':None,'privilege_generator':source if source not in (None,'deepmath_dataset') else None,
                        'privilege_source':source,'self_teacher':c['student'] if method=='opsd' else None,
                        'classification':'OPSD thinking-pattern consistency' if group=='exp1_opsd_consistency' else method})
    evaluations = []
    for record in records:
        c = json.loads(Path(record['config']).read_text())
        if c['group'] != 'exp2':
            continue
        for privileged in (False, True):
            # Checkpoint paths are deliberately explicit placeholders until a real saved/merged checkpoint is selected.
            condition = 'M0' if c['method']=='baseline' else 'M_'+c['method'].upper()
            evaluations.append({'experiment_id':record['experiment_id'], 'parameter_state':condition,
                'context_condition':'privileged' if privileged else 'unprivileged',
                'teacher_role':'OPSD_self_teacher' if c['method']=='opsd' and privileged else None,
                'reference_source':'qwen3_4b_nonthinking', 'dataset':'<SAME_FIXED_DIAGNOSTIC_REFERENCE_JSONL>',
                'checkpoint':'<INITIAL_OR_SAVED_HF_CHECKPOINT>', 'revision':'<IMMUTABLE_REVISION>',
                'command':shlex.join([sys.executable,'-m','scripts.research.run','evaluate',
                    '--model','<INITIAL_OR_SAVED_HF_CHECKPOINT>','--revision','<IMMUTABLE_REVISION>',
                    '--input','<SAME_FIXED_DIAGNOSTIC_REFERENCE_JSONL>','--output',
                    str(out/(record['experiment_id']+('_privileged' if privileged else '_plain')+'.eval.jsonl')),
                    '--samples','16']+(['--privileged'] if privileged else []))})
    write_jsonl(out/'evaluation_matrix.jsonl',evaluations)
    write_jsonl(out/'matrix.jsonl',records)
    (out/'commands.txt').write_text('\n'.join(r['command'] for r in records)+'\n')
    return records
