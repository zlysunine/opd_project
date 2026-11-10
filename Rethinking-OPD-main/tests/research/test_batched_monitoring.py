"""Batch padding/RNG correctness, interrupted generation recovery and selection-cache guards."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from analysis.evaluate import rollout_many, length_grouped_batches
from scripts.research.common import ROOT, digest, file_hash, write_json, write_jsonl
from scripts.research.models import Model, create_tiny
from scripts.research.preparation_cache import FILES, restore, seal_snapshot


def question(i):
    return dict(problem_id=f'q{i}', question='x'*(i+1), ground_truth='2', benchmark='controlled_modular')


class FakeModel:
    identity = {'checkpoint':'fixture'}

    def __init__(self, fail_call=None):
        self.calls = 0
        self.fail_call = fail_call

    def prompt(self, row):
        return [2]*len(row['question'])

    def generate_batch(self, rows, seeds, **kwargs):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError('interrupted batch')
        return [([int(seed % 100)+3, 1], r'\boxed{2}', False) for seed in seeds]


class BatchedMonitoringTests(unittest.TestCase):
    def test_vllm_eos_normalization_and_continuous_results_keep_original_order(self):
        from analysis.vllm_generation import normalize_response
        self.assertEqual(normalize_response([7,1,8],1,'stop'),([7,1],False))
        self.assertEqual(normalize_response([7],1,'stop'),([7,1],False))
        self.assertEqual(normalize_response([7],1,'length'),([7],True))
        with self.assertRaises(ValueError):normalize_response([7],1,'abort')
        with tempfile.TemporaryDirectory() as tmp:
            rows=[question(i) for i in range(3)]
            config=dict(generation_engine='vllm',generation_batch_size=64,seed=42,max_probe_response_tokens=8)
            def generated(model, selected, seeds, maximum, directory, callback, fraction):
                for i in [2,0,1]:callback(i,([9,1],r'\boxed{2}',False))
            with patch('analysis.vllm_generation.generate',side_effect=generated) as run:
                out=rollout_many(FakeModel(),rows,config,'live_rollout',0,Path(tmp)/'cache',Path(tmp)/'progress.json')
                self.assertEqual([r['problem_id'] for r in out],[r['problem_id'] for r in rows])
                self.assertTrue(all(r['decoding']['continuous_batching'] for r in out))
                again=rollout_many(FakeModel(),rows,config,'live_rollout',0,Path(tmp)/'cache',Path(tmp)/'progress.json')
                self.assertEqual(out,again);self.assertEqual(run.call_count,1)

    def test_length_buckets_preserve_every_question_without_padding_long_with_short(self):
        rows=[question(i) for i in range(7)]
        hints={row['problem_id']:n for row,n in zip(rows,[16384,10,900,400,16000,800,12])}
        batches=length_grouped_batches(rows,8,hints)
        self.assertEqual(sorted(i for batch in batches for i in batch),list(range(7)))
        self.assertIn([4,0],batches)
        self.assertTrue(all(set(batch)<=set([0,4]) or not set(batch)&set([0,4]) for batch in batches))

    def test_bucketed_recovery_repeats_original_batch_shape_and_keeps_output_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[question(i) for i in range(4)]
            config=dict(generation_batch_size=2,generation_length_bucketing=True,seed=42,max_probe_response_tokens=8)
            hints={r['problem_id']:10 for r in rows}
            cache=Path(tmp)/'cache';progress=Path(tmp)/'progress.json'
            original=rollout_many(FakeModel(),rows,config,'fixed_base_rollout',0,cache,progress,length_hints=hints)
            (cache/'000001.json').unlink()
            model=FakeModel()
            recovered=rollout_many(model,rows,config,'fixed_base_rollout',0,cache,progress,length_hints=hints)
            self.assertEqual(original,recovered)
            self.assertEqual(model.calls,1)
            self.assertEqual([r['problem_id'] for r in recovered],[r['problem_id'] for r in rows])
            with self.assertRaisesRegex(ValueError,'identity changed'):
                rollout_many(model,rows,config,'fixed_base_rollout',0,cache,progress,
                             length_hints={r['problem_id']:20 for r in rows})

    def test_generation_backend_is_restored_before_geometry_even_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,7)
            model=Model(tmp,'fixture:7',prompt_mode='plain_completion')
            rows=[question(1),question(7)]
            model.generate_batch(rows,[1,2],max_tokens=3,greedy=True,attention_backend='sdpa')
            self.assertEqual(model.model.config._attn_implementation,'eager')
            with patch.object(model.model,'generate',side_effect=RuntimeError('generation failed')):
                with self.assertRaisesRegex(RuntimeError,'generation failed'):
                    model.generate_batch(rows,[1,2],max_tokens=3,greedy=True,attention_backend='sdpa')
            self.assertEqual(model.model.config._attn_implementation,'eager')

    def test_real_model_left_padding_matches_individual_greedy(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp, 7)
            model = Model(tmp, 'fixture:7', prompt_mode='plain_completion')
            rows = [question(1), question(7), question(19)]
            serial = [model.generate(row, max_tokens=5, greedy=True)[0] for row in rows]
            batch = model.generate_batch(rows, [17, 18, 19], max_tokens=5, greedy=True)
            self.assertEqual(serial, [value[0] for value in batch])

    def test_sampling_seeds_follow_questions_across_batch_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp, 8)
            model = Model(tmp, 'fixture:8', prompt_mode='plain_completion')
            rows = [question(1), question(7), question(19)]
            a = model.generate_batch(rows, [17, 18, 19], max_tokens=8)
            b = model.generate_batch(rows[::-1], [19, 18, 17], max_tokens=8)
            self.assertEqual([x[0] for x in a], [x[0] for x in b[::-1]])
            with self.assertRaises(ValueError):
                model.generate_batch(rows, [17], max_tokens=8)

    def test_interrupted_generation_resumes_only_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, progress = Path(tmp)/'cache', Path(tmp)/'progress.json'
            rows = [question(i) for i in range(5)]
            config = dict(generation_batch_size=2, seed=42, max_probe_response_tokens=8)
            model = FakeModel(fail_call=2)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                rollout_many(model, rows, config, 'fixed_base_rollout', 0, cache, progress)
            self.assertEqual(json.loads(progress.read_text())['completed'], 2)
            model = FakeModel()
            result = rollout_many(model, rows, config, 'fixed_base_rollout', 0, cache, progress)
            self.assertEqual(model.calls, 2)
            self.assertEqual([r['problem_id'] for r in result], [r['problem_id'] for r in rows])
            self.assertTrue(all(r['seed'] == int(digest([42, r['problem_id'], 'fixed_base_rollout'])[:8], 16)
                                for r in result))
            self.assertEqual(json.loads(progress.read_text())['completed'], 5)
            model = FakeModel()
            self.assertEqual(result, rollout_many(model, rows, config, 'fixed_base_rollout', 0, cache, progress))
            self.assertEqual(model.calls, 0)
            saved = json.loads((cache/'000000.json').read_text())
            saved['result']['response_ids'] = [99]
            write_json(cache/'000000.json', saved)
            with self.assertRaisesRegex(ValueError, 'item changed'):
                rollout_many(model, rows, config, 'fixed_base_rollout', 0, cache, progress)

    def test_changed_generation_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [question(0)]
            config = dict(generation_batch_size=2, seed=42, max_probe_response_tokens=8)
            rollout_many(FakeModel(), rows, config, 'live_rollout', 1, Path(tmp)/'cache', Path(tmp)/'progress.json')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                rollout_many(FakeModel(), rows, {**config, 'seed':43}, 'live_rollout', 1,
                             Path(tmp)/'cache', Path(tmp)/'progress.json')


class PreparationReuseTests(unittest.TestCase):
    def snapshot(self, tmp):
        backup = Path(tmp)/'backup'
        data = backup/'prepared'
        data.mkdir(parents=True)
        model = Path(tmp)/'model'
        create_tiny(model, 9)
        native = Path(tmp)/'native.jsonl'
        write_jsonl(native, [question(0)])
        c = dict(model=dict(path=str(model), revision='fixture:9'),
                 dataset=dict(input=str(native), revision='fixture:dataset', train_n=2, dev_n=1),
                 monitor=dict(seed=42, heldout_probe_n=1, train_probe_n=1),
                 training=dict(max_prompt_length=32, max_teacher_prompt_length=64, max_response_length=8))
        split_ids = dict(train=['q0', 'q1'], validation=['q2'], heldout_probe=['q3'], train_probe=['q0'])
        for name, values in split_ids.items():
            write_jsonl(data/f'{name}.jsonl', [dict(problem_id=i, original_id=i, split=name) for i in values])
        write_jsonl(data/'eligibility.jsonl', [dict(problem_id=f'q{i}', exclusion_reason=None,
                                                  reference_verified=True, student_tokens=10, teacher_tokens=20)
                                               for i in range(4)])
        write_jsonl(data/'eligible.jsonl', [])
        write_jsonl(data/'duplicates.jsonl', [])
        write_jsonl(data/'answer_encoding_quarantine.jsonl', [])
        write_json(data/'eligible.jsonl.manifest.json', {})
        for name in ('analysis/prepare_probe.py', 'analysis/evaluate.py', 'scripts/research/data.py',
                     'verl/verl/utils/opsd.py', 'scripts/val/eval/utils.py','scripts/research/answer_protocol.py'):
            dest = backup/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        write_json(backup/'source_config.json', c)
        write_json(backup/'snapshot.json', dict(source_workdir='old:fixture',
                   files={name:dict(sha256=file_hash(data/name)) for name in FILES}))
        seal_snapshot(backup)
        c['dataset']['preparation_cache'] = str(data)
        return c, data

    def test_reuse_preserves_ids_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, data = self.snapshot(tmp)
            output = Path(tmp)/'new'
            output.mkdir()
            splits, duplicates = restore(c, output)
            self.assertEqual([r['problem_id'] for r in splits['train']], ['q0', 'q1'])
            self.assertEqual(duplicates, [])
            for name in FILES:
                self.assertEqual(file_hash(data/name), file_hash(output/name))
            self.assertTrue((output/'preparation_reuse.json').is_file())

    def test_changed_rules_or_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, data = self.snapshot(tmp)
            output = Path(tmp)/'new'
            output.mkdir()
            c['training']['max_teacher_prompt_length'] = 65
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore(c, output)
            c['training']['max_teacher_prompt_length'] = 64
            (data/'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                restore(c, output)

    def test_generation_changes_reusable_but_selection_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, data = self.snapshot(tmp)
            backup = data.parent
            source = backup/'source/analysis/prepare_probe.py'
            text = source.read_text().replace("'fixed_base_rollout', 0,", "'different_pool', 0,")
            source.write_text(text)
            seal_snapshot(backup)
            source.write_text(text.replace("c['dataset']['dev_n']", "c['dataset']['dev_n']+1"))
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                seal_snapshot(backup)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
