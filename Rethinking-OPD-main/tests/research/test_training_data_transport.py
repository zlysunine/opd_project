"""Training metadata survives Parquet without empty-struct or union-key loss."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.training_data import export_verl_training
from scripts.research.answer_protocol import evaluate_answer


class TrainingDataTransportTests(unittest.TestCase):
    def test_empty_and_different_option_maps_round_trip_exactly(self):
        import pyarrow.parquet as pq
        from scripts.research.training_data import decode_extra_info
        rows=[]
        for i,mapping in enumerate([{}, {'A':'True','B':'False'}, {'A':'1','B':'2','C':'3'}]):
            rows.append(dict(problem_id=str(i),question='True or false?',ground_truth='A' if mapping else '2',
                benchmark='DeepMath-103K',messages=[dict(role='user',content='Question')],
                option_mapping=mapping,nested=dict(empty={},null=None,choices=[mapping])))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.parquet';export_verl_training(rows,path)
            saved=pq.read_table(path).to_pylist()
        for raw,item in zip(rows,saved):
            self.assertEqual(decode_extra_info(item['extra_info']),{**raw,'index':raw['problem_id']})
        mapped=decode_extra_info(saved[1]['extra_info'])
        self.assertTrue(evaluate_answer('Final answer: True','A',row=mapped)['correct'])
        self.assertEqual(decode_extra_info({}),{})
        self.assertIsNone(decode_extra_info(None))
        with self.assertRaises(ValueError):decode_extra_info('[1,2]')
