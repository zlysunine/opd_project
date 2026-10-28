"""Format, label semantics, shared grading and durable quality-gate regressions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.evaluate import answer_status, enforce_quality, rollout_many
from scripts.research import answer_protocol as protocol
from scripts.research.common import ROOT, opsd, read_jsonl, write_jsonl
from scripts.research.data import verify
from scripts.research.reward import compute_score


class AnswerProtocolTests(unittest.TestCase):
    def test_plain_base_prompt_has_the_same_contract_for_both_roles(self):
        student=opsd.plain_messages('Q?')
        self.assertIn('Final answer: \\boxed{YOUR_ANSWER}',student[0]['content'])
        teacher=opsd.privileged_messages(student,'REFERENCE','plain_completion')
        self.assertEqual(teacher[0]['content'],'Reference information:\nREFERENCE\n\n'+student[0]['content'])
        self.assertNotIn('REFERENCE',student[0]['content'])

    def test_valid_format_variants_and_nonempty_nested_math(self):
        cases={r'Final answer: \boxed{42}':'42',
               'Final answer: \\boxed \n {\\frac{1}{2}}':r'\frac{1}{2}',
               r'Final answer: $\boxed{\{1,2\}}$':r'\{1,2\}',
               'Final answer: 42':'42',r'\boxed{0}':'0',r'\fbox{2}':'2'}
        for text,expected in cases.items():
            with self.subTest(text=text):
                parsed=protocol.extract_final_answer(text)
                self.assertFalse(parsed['extraction_failed'])
                self.assertEqual(parsed['extracted_answer'],expected)
        self.assertTrue(protocol.extract_final_answer(r'Final answer: \boxed{42}')['format_compliant'])
        self.assertFalse(protocol.extract_final_answer('Final answer: 42')['format_compliant'])

    def test_missing_empty_malformed_and_placeholder_are_not_fabricated_answers(self):
        for text in ['42','The answer is 42.',r'\boxed{}',r'\boxed{ }',r'\boxed{42',
                     r'\boxed{42} later \boxed{',r'Final answer: \boxed{YOUR_ANSWER}',
                     'Final answer:','Final answer: unknown']:
            with self.subTest(text=text):
                self.assertTrue(protocol.extract_final_answer(text)['extraction_failed'])

    def test_conflicting_final_regions_and_malformed_last_region_do_not_fallback(self):
        for text in ['Final answer: 42\nFinal answer: 43',
                     'Final answer: \\boxed{42}\nFinal answer: \\boxed{',
                     r'\boxed{42}'+'\nFinal answer: ']:
            self.assertTrue(protocol.extract_final_answer(text)['extraction_failed'])
        self.assertEqual(protocol.extract_final_answer('Work: \\boxed{7}\nFinal answer: \\boxed{42}')['extracted_answer'],'42')

    def test_explicit_plain_final_is_graded_without_old_boxed_reparse(self):
        text='Reasoning.\nFinal answer: 42'
        self.assertTrue(verify(text,'42','DeepMath-103K'))
        self.assertEqual(compute_score('DeepMath-103K',text,'42'),1.)
        status=answer_status(text,'42','DeepMath-103K')
        self.assertTrue(status['correct'])
        self.assertFalse(status['format_compliant'])
        self.assertFalse(answer_status('Final answer: 41','42','DeepMath-103K')['correct'])

    def test_symbolic_equivalence_receives_candidates_not_response_text(self):
        grader=SimpleNamespace(grade_answer_mathd=lambda a,b:False,
                               grade_answer_sympy=lambda a,b:(a,b)==('x+x','2*x'))
        with patch.object(protocol,'_math_verifier',return_value=grader):
            self.assertTrue(verify('Final answer: x+x','2*x','DeepMath-103K'))

    def test_truncated_candidate_is_extractable_but_not_correct(self):
        status=answer_status('Final answer: \\boxed{42}','42','DeepMath-103K',truncated=True)
        self.assertFalse(status['extraction_failed'])
        self.assertFalse(status['correct'])

    def test_boolean_label_requires_authoritative_mapping(self):
        row=dict(question='Is this true or false?',ground_truth='A')
        self.assertEqual(protocol.answer_metadata(row)['answer_encoding_status'],'unmapped_boolean_label')
        self.assertFalse(answer_status(r'\boxed{True}','A','DeepMath-103K',row=row)['correct'])
        mapped={**row,'option_mapping':{'A':'True','B':'False'}}
        self.assertTrue(answer_status('Final answer: True','A','DeepMath-103K',row=mapped)['correct'])
        self.assertEqual(compute_score('DeepMath-103K','Final answer: True','A',extra_info=mapped),1.)
        self.assertFalse(answer_status('Final answer: False','A','DeepMath-103K',row=mapped)['correct'])

    def test_wrong_boolean_is_wrong_answer_not_extraction_failure(self):
        status=answer_status('Final answer: False','True','DeepMath-103K',row={'question':'True or false?'})
        self.assertFalse(status['extraction_failed'])
        self.assertFalse(status['correct'])
        self.assertTrue(answer_status('Final answer: 1','True','DeepMath-103K')['extraction_failed'])
        self.assertTrue(answer_status(r'Final answer: \boxed{\text{True}}','True','DeepMath-103K')['correct'])

    def test_choice_mapping_is_not_inferred_from_reference(self):
        self.assertEqual(protocol.answer_metadata({'question':'Pick a choice','ground_truth':'A',
                         'privilege_text':r'\boxed{A}'})['answer_encoding_status'],'ambiguous_letter_answer')
        row={'question':'Pick one\nA. 42\nB. 17','ground_truth':'A'}
        meta=protocol.answer_metadata(row)
        self.assertEqual(meta['answer_type'],'multiple_choice')
        self.assertTrue(protocol.grade_extracted_answer('42','A',meta))
        self.assertFalse(protocol.grade_extracted_answer('B','A',meta))
        content_target=protocol.answer_metadata({**row,'ground_truth':'42'})
        self.assertEqual(content_target['canonical_ground_truth'],'A')
        self.assertTrue(protocol.grade_extracted_answer('A','42',content_target))
        self.assertEqual(protocol.answer_metadata({'ground_truth':'A','answer_type':'math'})['answer_encoding_status'],'ok')

    def test_gate_persists_raw_failed_outputs_and_never_changes_denominator(self):
        rows=[dict(problem_id=str(i),split='heldout_probe',extraction_failed=i==0,
                   parse_status='missing_final_answer' if i==0 else 'extracted',response_text='raw') for i in range(100)]
        config={'quality_gate':dict(enabled=True,max_extraction_failure_rate=.01,splits=['heldout_probe'])}
        with tempfile.TemporaryDirectory() as tmp:
            enforce_quality(rows,config,tmp)
            self.assertTrue(json.loads((Path(tmp)/'quality.json').read_text())['passed'])
            rows[1]['extraction_failed']=True
            with self.assertRaisesRegex(ValueError,'quality gate failed'):
                enforce_quality(rows,config,tmp)
            summary=json.loads((Path(tmp)/'quality.json').read_text())
            self.assertEqual(summary['splits'][0]['n'],100)
            self.assertEqual(summary['splits'][0]['failures'],2)
            self.assertEqual(len(read_jsonl(Path(tmp)/'responses.jsonl')),100)
            self.assertEqual(len(read_jsonl(Path(tmp)/'failed_responses.jsonl')),2)

    def test_seventy_percent_gate_on_one_hundred_questions(self):
        rows=[dict(problem_id=str(i),split='heldout_probe',extraction_failed=i>=70,
                   parse_status='extracted' if i<70 else 'missing_final_answer',response_text='raw') for i in range(100)]
        config={'quality_gate':dict(enabled=True,max_extraction_failure_rate=.30,splits=['heldout_probe'])}
        with tempfile.TemporaryDirectory() as tmp:
            enforce_quality(rows,config,tmp)
            summary=json.loads((Path(tmp)/'quality.json').read_text())
            self.assertTrue(summary['passed'])
            self.assertEqual(summary['splits'][0]['successes'],70)
            self.assertEqual(summary['splits'][0]['extraction_success_rate'],.70)
            self.assertEqual(summary['splits'][0]['minimum_success_rate'],.70)
            rows[69]['extraction_failed']=True
            with self.assertRaisesRegex(ValueError,'quality gate failed'):
                enforce_quality(rows,config,tmp)
            self.assertEqual(len(read_jsonl(Path(tmp)/'responses.jsonl')),100)
            self.assertEqual(len(read_jsonl(Path(tmp)/'failed_responses.jsonl')),31)

    def test_empty_probe_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'quality gate failed'):
            enforce_quality([],{'quality_gate':{'enabled':True}},tmp)

    def test_behavior_and_fixed_generation_limits_and_serial_cache_recovery(self):
        calls=[]
        model=SimpleNamespace(identity={'fixture':True},prompt=lambda row:[1,2],
                              generate=lambda row,**kw:(calls.append(kw['max_tokens']) or ([3,1],r'Final answer: \boxed{2}',False)))
        row=dict(question='Q',ground_truth='2',benchmark='DeepMath-103K',problem_id='q',split='heldout_probe')
        config=dict(seed=42,max_probe_response_tokens=4,behavior_max_response_tokens=16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=rollout_many(model,[row],config,'live_rollout',0,root/'live',root/'progress')
            self.assertTrue(a[0]['correct'])
            self.assertEqual(a,rollout_many(model,[row],config,'live_rollout',0,root/'live',root/'progress'))
            rollout_many(model,[row],config,'fixed_base_rollout',0,root/'fixed',root/'progress')
            self.assertEqual(calls,[16,4])

    def test_freeze_quarantines_unmapped_boolean_and_gates_before_training(self):
        from analysis.prepare_probe import freeze
        from scripts.research.models import Model,create_tiny
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            create_tiny(root/'base',7)
            native=[dict(question=f'Question {i}: 1+1?',final_answer='2',difficulty=5.,topic='Math -> Algebra',
                         r1_solution_1=r'Final answer: \boxed{2}') for i in range(6)]
            native.append(dict(question='Determine true or false: every integer is even.',final_answer='A',
                               difficulty=5.,topic='Math -> Algebra',r1_solution_1=r'\boxed{A}'))
            write_jsonl(root/'data.jsonl',native)
            c=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
            c.update(output_root=str(root/'shared'))
            c['model'].update(path=str(root/'base'),revision='fixture',device='cpu',dtype='float32')
            c['dataset'].update(input=str(root/'data.jsonl'),revision='fixture',train_n=2,dev_n=1)
            c['training'].update(max_response_length=8)
            c['monitor'].update(heldout_probe_n=2,train_probe_n=1,max_probe_response_tokens=4,
                                behavior_max_response_tokens=16,anchors=['prompt_last'])
            model=Model(root/'base','fixture',prompt_mode='plain_completion')
            with patch.object(model,'generate',return_value=([3,1],r'Final answer: \boxed{2}',False)):
                target=freeze(c,model=model)
            bad=read_jsonl(target/'answer_encoding_quarantine.jsonl')
            self.assertEqual(len(bad),1)
            self.assertEqual(bad[0]['answer_encoding_status'],'unmapped_boolean_label')
            self.assertEqual(len(read_jsonl(target/'baseline_live_rollouts.jsonl')),3)
            self.assertTrue(json.loads((root/'shared/quality/baseline/quality.json').read_text())['passed'])
            c['output_root']=str(root/'rejected')
            with patch.object(model,'generate',return_value=([3,1],'2',False)):
                with self.assertRaisesRegex(ValueError,'quality gate failed'):
                    freeze(c,model=model)
            self.assertFalse((root/'rejected/probe/complete.json').exists())
            self.assertEqual(len(read_jsonl(root/'rejected/quality/baseline/failed_responses.jsonl')),3)


if __name__=='__main__':
    unittest.main()
