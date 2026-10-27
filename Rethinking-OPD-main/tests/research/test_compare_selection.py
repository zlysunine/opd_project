"""Default execution must monitor OPSD without accidentally launching the RL comparison."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.research import compare
from scripts.research.common import ROOT
from scripts.research.launch import hydra_options, resolve


RUN_ARGS = ['--model-dir','/fixture/base','--model-revision','fixture_commit',
            '--input','/fixture/deepmath.jsonl','--dataset-revision','fixture_dataset_commit']


class ComparisonSelectionTests(unittest.TestCase):
    def config(self):
        return dict(workdir='/fixture/run',model={},training={'gpus':1},monitor={'optimizer_steps':[0,1]})

    def test_all_execution_entrypoints_default_to_opsd_with_explicit_comparison_available(self):
        for stage in ('run','train','analyze'):
            args=[stage]+(RUN_ARGS if stage=='run' else [])
            self.assertEqual(compare.parser().parse_args(args).method,'opsd')
            for method in ('grpo','both'):
                self.assertEqual(compare.parser().parse_args(args+['--method',method]).method,method)
        # Shell supplies the OPSD default first; an explicit user option still wins.
        args=compare.parser().parse_args(['run','--method','opsd',*RUN_ARGS,'--method','grpo'])
        self.assertEqual(args.method,'grpo')

    def test_preview_describes_only_selected_methods(self):
        for option,expected in (([],['opsd']),(['--method','grpo'],['grpo']),(['--method','both'],['opsd','grpo'])):
            output=io.StringIO()
            with patch.object(compare,'configure',return_value=self.config()),contextlib.redirect_stdout(output):
                compare.main(['run',*RUN_ARGS,*option])
            plan=json.loads(output.getvalue())
            self.assertEqual(plan['methods'],expected)
            stages=[f'train {method.upper()} -> analyze checkpoints' for method in expected]
            self.assertEqual(plan['stages'],['prepare shared probes',*stages,'report'])

    def test_default_run_trains_and_analyzes_only_opsd_then_reports(self):
        for option,expected in (([],['opsd']),(['--method','both'],['opsd','grpo']),(['--method','grpo'],['grpo'])):
            c=self.config()
            with (patch.object(compare,'configure',return_value=c),
                  patch('scripts.research.preflight.check_environment',return_value={'ok':True}),
                  patch.object(compare,'write_json'),patch.object(compare,'prepare') as prepare,
                  patch.object(compare,'train') as train,patch.object(compare,'analyze') as analyze,
                  patch('analysis.report.report') as report,contextlib.redirect_stdout(io.StringIO())):
                compare.main(['run',*RUN_ARGS,'--execute','--foreground',*option])
                prepare.assert_called_once_with(c)
                self.assertEqual([call.args for call in train.call_args_list],[(c,method,True) for method in expected])
                self.assertEqual([call.args for call in analyze.call_args_list],[(c,method,None) for method in expected])
                report.assert_called_once_with(c['workdir'])

    def test_standalone_stages_also_do_not_start_grpo_by_default(self):
        c=self.config()
        with (patch.object(compare,'load',return_value=c),patch.object(compare,'train') as train,
              patch.object(compare,'analyze') as analyze):
            compare.main(['train','--execute','--foreground'])
            train.assert_called_once_with(c,'opsd',True)
            analyze.assert_not_called()
            train.reset_mock()
            compare.main(['analyze'])
            analyze.assert_called_once_with(c,'opsd',None)
            train.assert_not_called()


    def test_default_50_steps_include_final_monitoring_and_effective_training_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=root/'Qwen3-1.7B-Base'
            model.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'qwen3','_name_or_path':'Qwen/Qwen3-1.7B-Base'}))
            native=root/'native.jsonl'
            native.write_text('{}\n')
            output=io.StringIO()
            args=['run','--model-dir',str(model),'--model-revision','fixture_commit',
                  '--input',str(native),'--dataset-revision','fixture_dataset_commit','--workdir',str(root/'run')]
            with contextlib.redirect_stdout(output):
                compare.main(args)
            plan=json.loads(output.getvalue())
            self.assertEqual(plan['training']['steps'],50)
            self.assertEqual(plan['methods'],['opsd'])
            self.assertEqual(plan['monitor_steps'],[0,1,5,10,20,50])
            c=json.loads((root/'run/config.json').read_text())
            self.assertTrue(c['monitor']['compare_student_teacher'])
            self.assertEqual(c['monitor']['teacher_geometry_steps'],c['monitor']['optimizer_steps'])
            for key in ('optimizer_steps','teacher_geometry_steps','plot_full_rsm_steps'):
                self.assertIn(50,c['monitor'][key])
                self.assertLessEqual(max(c['monitor'][key]),50)
            self.assertEqual((c['training']['batch'],c['training']['rollouts'],c['training']['learning_rate']),(8,8,1e-6))
            self.assertFalse((root/'run/opsd/train').exists())
            profile=resolve(ROOT/c['training']['profile'])
            self.assertEqual(profile['steps'],50)
            options=hydra_options(profile,root/'plan','BASE')
            self.assertEqual(options['trainer.total_training_steps'],50)
            parsed=compare.parser().parse_args(['run',*RUN_ARGS])
            self.assertEqual(parsed.workdir,'research_runs/base_geometry_50_answer_v2_full_tokens_paired')


if __name__=='__main__':
    unittest.main()
