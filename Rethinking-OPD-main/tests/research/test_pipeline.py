"""CPU checks for the ordered launcher; GPU execution is explicitly mocked."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.research.common import ROOT, read_jsonl, write_json, write_jsonl
from scripts.research.models import ByteTokenizer, create_tiny
from scripts.research import pipeline
from scripts.research.launch import plan, hydra_options
from scripts.research.preflight import validate_training_inputs


class PipelineTests(unittest.TestCase):
    def setup_config(self, directory, *extra):
        root=Path(directory)
        model=root/'model'
        if not (model/'config.json').exists(): create_tiny(model,3)
        native=root/'native.jsonl'
        if not native.exists():
            write_jsonl(native,[{'question':f'fixture q{i}','final_answer':'1','difficulty':6.,
                'r1_solution_1':('r'*3000 if i==0 else 'Reference answer \\boxed{1}.')} for i in range(24)])
        args=pipeline.parser().parse_args(['configure','--workdir',str(root/'run'),
            '--model-dir',str(model),'--model-revision','local:fixture','--input',str(native),
            '--dataset-revision','local:source-fixture','--steps','2','--batch','4','--max-teacher-prompt-length','512',*extra])
        with contextlib.redirect_stdout(io.StringIO()): return pipeline.configure(args)

    def test_prepare_checks_lengths_and_keeps_conditions_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            with self.assertRaisesRegex(ValueError,'overlong'):
                pipeline.prepare(c,tokenizer=ByteTokenizer())
            with contextlib.redirect_stdout(io.StringIO()):
                report=pipeline.prepare(c,'drop',16,ByteTokenizer())
                reused=pipeline.prepare(c,'drop',16,ByteTokenizer())
            self.assertEqual(report['length_eligible_questions'],23)
            self.assertEqual(report['selected_questions'],16)
            self.assertEqual(report['cache_key'],reused['cache_key'])
            for split in ('train','validation','test'):
                s=read_jsonl(Path(c['pipeline_workdir'])/f'data/solution/{split}.jsonl')
                e=read_jsonl(Path(c['pipeline_workdir'])/f'data/empty/{split}.jsonl')
                self.assertEqual([r['problem_id'] for r in s],[r['problem_id'] for r in e])
                self.assertTrue(all(not r['privilege_text'] for r in e))
            Path(c['train_data']).write_text('modified\n')
            with self.assertRaisesRegex(ValueError,'modified'):
                pipeline.prepare(c,'drop',16,ByteTokenizer())

    def test_configure_refuses_model_revision_changes_in_existing_workdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.setup_config(tmp)
            self.setup_config(tmp) # exact retry is safe
            with self.assertRaisesRegex(ValueError,'model differs'):
                self.setup_config(tmp,'--model-revision','local:other')

    def test_training_seed_does_not_change_data_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            other={**c,'seed':43,'pipeline_workdir':str(Path(tmp)/'other')}
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.prepare(c,'drop',16,ByteTokenizer())
                pipeline.prepare(other,'drop',16,ByteTokenizer())
            for split in ('train','validation','test'):
                a=read_jsonl(Path(c['pipeline_workdir'])/f'data/solution/{split}.jsonl')
                b=read_jsonl(Path(other['pipeline_workdir'])/f'data/solution/{split}.jsonl')
                self.assertEqual([r['problem_id'] for r in a],[r['problem_id'] for r in b])

    def test_training_plan_uses_self_teacher_and_never_starts_process(self):
        with self.assertRaisesRegex(ValueError,'configured pipeline workspace'):
            pipeline.main(['train','--config',str(ROOT/'configs/research/pilot_1p7b.json'),'--condition','empty'])
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            with patch('scripts.research.launch.subprocess.run') as process:
                # provenance uses check_output -> subprocess.run; delegate Git calls only.
                original=subprocess.Popen
                def git_only(command,**kwargs):
                    if command[0]!='git': raise AssertionError('A plan must not start training')
                    proc=original(command,cwd=kwargs.get('cwd'),stdout=subprocess.PIPE)
                    stdout,_=proc.communicate()
                    return subprocess.CompletedProcess(command,proc.returncode,stdout=stdout.decode() if kwargs.get('text') else stdout)
                process.side_effect=git_only
                _,command=plan(c,Path(tmp)/'plan',False)
            options=[x for x in command if x.startswith(('+opsd=','reward_model.enable='))]
            self.assertIn('reward_model.enable=false',options)
            self.assertTrue(any('current_batch_stop_gradient' in x for x in options))
            self.assertFalse((Path(tmp)/'plan/execution_summary.json').exists())

    def test_preflight_rejects_insufficient_batches_and_context_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            with contextlib.redirect_stdout(io.StringIO()):pipeline.prepare(c,'drop',16,ByteTokenizer())
            train=read_jsonl(c['train_data']);val=read_jsonl(c['validation_data'])
            report=validate_training_inputs(c,train,val,ByteTokenizer(),2048)
            self.assertEqual(len(report),len(train)+len(val))
            with self.assertRaisesRegex(ValueError,'complete updates'):
                validate_training_inputs({**c,'steps':1000},train,val,ByteTokenizer(),2048)
            with self.assertRaisesRegex(ValueError,'exceeds model context'):
                validate_training_inputs(c,train,val,ByteTokenizer(),100)

    def test_rollout_seed_is_a_declared_dataclass_field(self):
        tree=ast.parse((ROOT/'verl/verl/workers/config/rollout.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RolloutConfig')
        fields={n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
        self.assertIn('seed',fields)
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            self.assertEqual(hydra_options(c,Path(tmp),'STUDENT')['+actor_rollout_ref.rollout.seed'],42)

    def test_shell_entrypoints_resolve_project_scripts_package(self):
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'verl')
        for script in ('run_controlled_comparison.sh','run_base_comparison.sh'):
            result=subprocess.run(['bash',str(ROOT/script),'--help'],cwd='/tmp',env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('usage:',result.stdout)
        for script in ('on_policy_distillation.sh','grpo.sh'):
            original=subprocess.check_output(['git','show',f'HEAD:Rethinking-OPD-main/{script}'],cwd=ROOT)
            self.assertEqual((ROOT/script).read_bytes(),original)
        from scripts.research.common import provenance
        sources=provenance({'test':'current_entrypoints'})['source_hashes']
        self.assertIn('run_controlled_comparison.sh',sources)
        self.assertNotIn('opsd_pipeline.sh',sources)
        self.assertNotIn('run_opsd_experiment.sh',sources)

    def workflow_args(self,c,*extra):
        info=json.loads(Path(c['registry']).read_text())[c['student']]
        return pipeline.parser().parse_args(['run','--workdir',c['pipeline_workdir'],
            '--model-dir',info['path'],'--model-revision',info['revision'],
            '--input',c['native_input'],'--dataset-revision',c['dataset_revision'],
            '--steps','2','--batch','4','--max-teacher-prompt-length','512',*extra])

    def test_one_click_plan_does_not_write_or_start_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'not_created'
            env=os.environ.copy()
            env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'verl'),env.get('PYTHONPATH','')])
            result=subprocess.run([sys.executable,'-m','scripts.research.pipeline','run','--workdir',str(work),
                '--model-dir',str(Path(tmp)/'absent_model'),'--model-revision','fixture:rev',
                '--input',str(Path(tmp)/'absent_data'),'--dataset-revision','fixture:data'],
                cwd='/tmp',env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('plan_only',result.stdout)
            self.assertFalse(work.exists())

    def test_one_click_dispatches_all_conditions_and_preserves_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            a=self.workflow_args(c,'--execute','--with-empty','--with-test','--overlength','drop','--max-problems','16')
            with patch('scripts.research.pipeline.main') as dispatch,contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_workflow(a)
            commands=[call.args[0] for call in dispatch.call_args_list]
            self.assertEqual([cmd[0] for cmd in commands],
                ['check','prepare']+['evaluate','train','merge','evaluate','plot','evaluate']*2)
            self.assertIn('empty',commands[8])
            self.assertTrue(all('--execute' in cmd for cmd in commands if cmd[0]=='train'))
            journal=Path(c['pipeline_workdir'])/'workflow.json'
            report=json.loads(journal.read_text())
            self.assertEqual(report['status'],'passed')
            self.assertTrue(all(s['status']=='passed' for s in report['attempts'][0]['stages']))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'already completed'):
                pipeline.run_workflow(a)

    def test_one_click_stops_at_failure_and_can_retry_same_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            a=self.workflow_args(c,'--execute')
            def failure(command):
                if command[0]=='train': raise RuntimeError('fixture training failure')
            with patch('scripts.research.pipeline.main',side_effect=failure) as dispatch, \
                 contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'fixture training failure'):
                pipeline.run_workflow(a)
            self.assertEqual([call.args[0][0] for call in dispatch.call_args_list],['check','prepare','evaluate','train'])
            journal=Path(c['pipeline_workdir'])/'workflow.json'
            self.assertEqual(json.loads(journal.read_text())['status'],'failed')
            with patch('scripts.research.pipeline.main'),contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_workflow(a)
            report=json.loads(journal.read_text())
            self.assertEqual(len(report['attempts']),2)
            self.assertEqual([r['status'] for r in report['attempts']],['failed','passed'])

    def test_actual_launch_sequence_with_mocked_gpu_transport(self):
        # This validates preflight/export/child-command/automatic plots, not GPU kernels.
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            with contextlib.redirect_stdout(io.StringIO()):pipeline.prepare(c,'drop',16,ByteTokenizer())
            c['batch']=1;c['steps']=1
            out=Path(tmp)/'actual_launch'
            original_run=subprocess.run
            def child(command,**kwargs):
                if len(command)>2 and command[2]=='verl.trainer.main_ppo':
                    keys=c['required_training_metrics']
                    write_jsonl(out/'metrics.jsonl',[{'step':1,'budget':{'optimized_tokens':8,'wall_seconds':.1},
                        'metrics':dict(zip(keys,[.5,-.01,2.,3.,1.]))}])
                    return subprocess.CompletedProcess(command,0)
                return original_run(command,**kwargs)
            def export(rows,path):Path(path).write_bytes(b'fixture-only')
            with patch('scripts.research.preflight.check_environment',return_value={'ok':True,'errors':[]}), \
                 patch('scripts.research.preflight.compose_training_config',return_value={'hydra':'fixture','rollout_schema':'fixture'}) as compose, \
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=ByteTokenizer()), \
                 patch('transformers.AutoConfig.from_pretrained',return_value=types.SimpleNamespace(max_position_embeddings=2048)), \
                 patch('scripts.research.launch.export_verl',side_effect=export), \
                 patch('scripts.research.launch.subprocess.run',side_effect=child), \
                 contextlib.redirect_stdout(io.StringIO()):
                plan(c,out,True)
            compose.assert_called_once()
            self.assertTrue((out/'plots/thinking_consistency.png').exists())
            self.assertEqual(json.loads((out/'execution_summary.json').read_text())['returncode'],0)
            self.assertTrue((out/'input_lengths.jsonl').exists())
            with patch('scripts.research.launch.subprocess.run',side_effect=child), \
                 patch('scripts.research.preflight.check_environment',return_value={'ok':False,'errors':['fixture missing dependency']}):
                with self.assertRaisesRegex(RuntimeError,'environment check failed'):plan(c,Path(tmp)/'blocked',True)

    def test_incomplete_checkpoint_is_rejected_before_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=self.setup_config(tmp)
            cp=Path(c['pipeline_workdir'])/'train/checkpoints'
            actor=cp/'global_step_2/actor'
            write_json(actor/'huggingface/config.json',{})
            (cp/'latest_checkpointed_iteration.txt').write_text('2')
            with self.assertRaisesRegex(FileNotFoundError,'rank 0'):pipeline.checkpoint_actor(c,'solution')
            (actor/'model_world_size_1_rank_0.pt').write_bytes(b'fixture')
            self.assertEqual(pipeline.checkpoint_actor(c,'solution')[0],2)
            with self.assertRaises(ValueError):pipeline.checkpoint_actor(c,'solution',3)


if __name__=='__main__': unittest.main()
