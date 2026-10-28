"""Preparation visibility and preserving expensive screening across quality failure."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.prepare_probe import freeze
from scripts.research.common import ROOT, write_json, write_jsonl
from scripts.research.job_control import status, render
from scripts.research.preparation_progress import PreparationProgress


class PreparationProgressTests(unittest.TestCase):
    def test_eta_is_based_on_completed_rows_and_no_eta_when_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'progress.json'
            with patch('scripts.research.preparation_progress.time.time',return_value=100.):
                progress = PreparationProgress(path,interval=0)
                progress.phase('eligibility_checks',total=100)
            initial = json.loads(path.read_text())
            self.assertIsNone(initial['remaining_seconds'])
            with patch('scripts.research.preparation_progress.time.time',return_value=110.):
                progress.update(25,accepted=20,excluded=5)
            value = json.loads(path.read_text())
            self.assertEqual(value['percent'],25.)
            self.assertEqual(value['rows_per_second'],2.5)
            self.assertEqual(value['remaining_seconds'],30.)
            with patch('scripts.research.preparation_progress.time.time',return_value=120.):
                progress.finish(ValueError('fixture error'))
            failed = json.loads(path.read_text())
            self.assertEqual(failed['processed'],25)
            self.assertEqual(failed['status'],'failed')
            self.assertIsNone(failed['remaining_seconds'])

    def test_subset_policy_is_part_of_selection_cache_identity(self):
        from scripts.research.preparation_cache import selection_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            write_json(model/'config.json',{'fixture':True})
            native=root/'data.json';write_json(native,[])
            c=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
            c['model'].update(path=str(model),revision='fixture')
            c['dataset'].update(input=str(native),revision='fixture')
            plain=selection_identity(c)
            self.assertNotIn('heldout_selection',plain['sampling'])
            c['monitor']['heldout_selection']={'policy':'first50','source_heldout_sha256':'fixture-a'}
            subset=selection_identity(c)
            self.assertNotEqual(plain,subset)
            c['monitor']['heldout_selection']['source_heldout_sha256']='fixture-b'
            self.assertNotEqual(subset,selection_identity(c))

    def test_status_reads_preserved_legacy_decisions_and_quality_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd = Path(tmp)
            write_jsonl(wd/'shared/preparation_audit/eligibility.jsonl',[
                dict(problem_id='a',exclusion_reason=None),
                dict(problem_id='b',exclusion_reason='overlong_prompt')])
            write_json(wd/'shared/quality/baseline/quality.json',dict(passed=False,splits=[
                dict(split='heldout_probe',n=8,failures=2,passed=False)]))
            report = status(wd)
            self.assertEqual(report['preparation']['processed'],2)
            self.assertEqual(report['preparation']['accepted'],1)
            self.assertEqual(report['phase'],'baseline_quality')
            output = render(report)
            self.assertIn('6/8（75.0%）',output)
            self.assertIn('筛选通过（去重前）：1',output)

    def test_status_reads_live_generation_instead_of_zero_stale_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd = Path(tmp)
            progress = PreparationProgress(wd/'shared/preparation_progress.json')
            progress.phase('baseline_generation',total=12)
            write_json(wd/'shared/generation_progress.json',dict(pool='live_rollout',completed=3,total=12,
                elapsed_seconds=30.,remaining_seconds=90.,updated_unix=progress.phase_started+30,status='running'))
            report = status(wd)
            self.assertEqual(report['preparation']['processed'],3)
            self.assertEqual(report['preparation']['remaining_seconds'],90.)

    def test_quality_failure_preserves_selection_and_retry_skips_screening(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd=Path(tmp);model_path=wd/'model';model_path.mkdir()
            write_json(model_path/'config.json',dict(fixture=True))
            native=wd/'native.jsonl'
            write_jsonl(native,[dict(problem_id=str(i),question=f'Question {i}',final_answer=str(i),
                difficulty=5.,topic='Math -> arithmetic',r1_solution_1=f'Final answer: \\boxed{{{i}}}') for i in range(4)])
            c=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
            c['model'].update(path=str(model_path),revision='fixture-model')
            c['dataset'].update(input=str(native),revision='fixture-data',train_n=2,dev_n=1)
            c['monitor'].update(heldout_probe_n=1,train_probe_n=1)
            c.update(workdir=str(wd),output_root=str(wd/'shared'))
            model=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:[1]),
                prompt=lambda *args,**kwargs:[1],
                model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768)))
            def failed_generation(model,rows,*args):
                return [{**row,'extraction_failed':True,'parse_status':'missing_final_answer'} for row in rows]
            with patch('analysis.prepare_probe.rollout_many',side_effect=failed_generation):
                with self.assertRaisesRegex(ValueError,'quality gate failed'):
                    freeze(c,model)
                checkpoint=wd/'shared/preparation_selection/selection_complete.json'
                self.assertTrue(checkpoint.is_file())
                first_workspace=next((wd/'shared').glob('.probe-*'))
                self.assertTrue((first_workspace/'eligible.jsonl').is_file())
                self.assertEqual(len((first_workspace/'eligibility.jsonl').read_text().splitlines()),4)
                receipt=json.loads(checkpoint.read_text())
                self.assertEqual(receipt['screening_counts']['accepted'],4)
                with patch('analysis.prepare_probe.load_questions',side_effect=AssertionError('must reuse screening')):
                    with self.assertRaisesRegex(ValueError,'quality gate failed'):
                        freeze(c,model)
            progress=json.loads((wd/'shared/preparation_progress.json').read_text())
            self.assertEqual(progress['status'],'failed')
            self.assertTrue(progress['reused_selection'])
            self.assertEqual(progress['accepted'],4)
            self.assertTrue((wd/'shared/quality/baseline/failed_responses.jsonl').is_file())
