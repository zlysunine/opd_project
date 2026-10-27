"""Direct same-checkpoint geometry: independent formulae and real extraction."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import spearmanr

from analysis.metrics.paired import paired_geometry, summarize_student_teacher
from analysis.smoke import smoke
from scripts.research.common import read_jsonl, write_json, write_jsonl


class StudentTeacherGeometryTests(unittest.TestCase):
    def setUp(self):
        self.h=np.random.default_rng(91).normal(size=(8,5))
        self.ids=[f'q{i}' for i in range(8)]

    def test_identical_and_orthogonally_rotated_representations(self):
        rotation=np.linalg.qr(np.random.default_rng(13).normal(size=(5,5)))[0]
        for teacher in (self.h.copy(),self.h@rotation):
            result=paired_geometry(self.h,teacher,self.ids,self.ids,ks=[2,3])
            self.assertAlmostEqual(result['metrics']['rsa_rho_student_teacher'],1.)
            self.assertAlmostEqual(result['metrics']['rsa_rmse_student_teacher'],0.)
            self.assertAlmostEqual(result['metrics']['linear_cka_student_teacher'],1.)
            self.assertEqual(result['metrics']['neighbor_overlap_student_teacher_k2'],1.)
            self.assertEqual(result['n_pairs'],28)

    def test_different_contexts_match_independent_formulae_and_id_alignment(self):
        t=np.random.default_rng(123).normal(size=(8,5))
        order=[7,3,2,4,1,5,0,6]
        result=paired_geometry(self.h,t[order],self.ids,[self.ids[i] for i in order],ks=[2])
        h=self.h
        hnorm=h/np.linalg.norm(h,axis=1)[:,None]
        tnorm=t/np.linalg.norm(t,axis=1)[:,None]
        tri=np.triu_indices(8,1)
        a,b=(hnorm@hnorm.T)[tri],(tnorm@tnorm.T)[tri]
        metrics=result['metrics']
        self.assertAlmostEqual(metrics['rsa_rho_student_teacher'],spearmanr(a,b).statistic)
        self.assertAlmostEqual(metrics['rsa_rmse_student_teacher'],np.sqrt(np.mean((a-b)**2)))
        x,y=h-h.mean(0),t-t.mean(0)
        expected=np.linalg.norm(x.T@y)**2/(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y))
        self.assertAlmostEqual(metrics['linear_cka_student_teacher'],expected)
        sdist=np.linalg.norm(h[:,None]-h[None,:],axis=-1)
        tdist=np.linalg.norm(t[:,None]-t[None,:],axis=-1)
        np.fill_diagonal(sdist,np.inf)
        np.fill_diagonal(tdist,np.inf)
        si,ti=np.argsort(sdist,axis=1)[:,:2],np.argsort(tdist,axis=1)[:,:2]
        expected=np.mean([len(set(s)&set(t))/2 for s,t in zip(si,ti)])
        self.assertEqual(metrics['neighbor_overlap_student_teacher_k2'],expected)
        np.testing.assert_array_equal(result['neighborhoods'][2]['student_indices'],si)
        np.testing.assert_array_equal(result['neighborhoods'][2]['teacher_indices'],ti)

    def test_joint_valid_cohort_and_small_cohort_missing_values(self):
        h,t=self.h.copy(),self.h.copy()
        h[0]=0
        t[1]=np.nan
        result=paired_geometry(h,t,self.ids,self.ids,ks=[2,6])
        self.assertEqual(result['n_valid'],6)
        self.assertEqual(result['invalid_sample_ids'],['q0','q1'])
        self.assertEqual(result['valid_sample_ids'],self.ids[2:])
        self.assertEqual(result['n_pairs'],15)
        self.assertTrue(np.isnan(result['metrics']['neighbor_overlap_student_teacher_k6']))
        self.assertEqual(result['metrics']['neighbor_overlap_student_teacher_k2'],1.)
        np.testing.assert_array_less(result['neighborhoods'][2]['student_indices'],6)
        constant=paired_geometry(np.ones((3,5)),np.ones((3,5)),self.ids[:3],self.ids[:3],ks=[2])
        self.assertTrue(np.isnan(constant['metrics']['rsa_rho_student_teacher']))
        self.assertTrue(np.isnan(constant['metrics']['linear_cka_student_teacher']))
        self.assertEqual(constant['metrics']['rsa_rmse_student_teacher'],0.)
        empty=paired_geometry(np.empty((0,5)),np.empty((0,5)),[],[],ks=[2])
        self.assertEqual(empty['n_valid'],0)
        self.assertTrue(all(np.isnan(v) for v in empty['metrics'].values()))

    def test_pair_sampling_is_seeded_and_cka_can_be_disabled(self):
        t=np.random.default_rng(321).normal(size=(8,5))
        first=paired_geometry(self.h,t,self.ids,self.ids,ks=[],max_pairs=9,seed=18,cka=False)
        second=paired_geometry(self.h,t,self.ids,self.ids,ks=[],max_pairs=9,seed=18,cka=False)
        self.assertEqual(first['n_pairs'],9)
        self.assertEqual(first['metrics'],second['metrics'])
        self.assertNotIn('linear_cka_student_teacher',first['metrics'])

    def test_mismatched_or_duplicate_ids_are_rejected(self):
        for other in (['wrong',*self.ids[1:]],['q0',*self.ids[:-1]]):
            with self.assertRaises(ValueError):
                paired_geometry(self.h,self.h,self.ids,other)
        with self.assertRaises(ValueError):
            paired_geometry(self.h,self.h[:,:4],self.ids,self.ids)

    def test_summary_has_separate_role_and_does_not_mix_splits(self):
        arrays,meta={},{}
        for split in ('heldout_probe','train_probe'):
            for role in ('student','privileged_teacher'):
                key=split+'__'+role
                arrays[key]=self.h if split=='heldout_probe' else self.h[:4]
                meta[key]=dict(role=role,pool='fixed_base_rollout',split=split,layer=1,anchor='prompt_last',
                               sample_ids=self.ids if split=='heldout_probe' else self.ids[:4])
        rows=[]
        manifest,neighbors=summarize_student_teacher(arrays,meta,dict(knn_k=[2],rsa_max_pairs=100,seed=7),
                                                    lambda *a,**kw:rows.append((a,kw)))
        self.assertEqual({row['n_valid'] for row in manifest},{4,8})
        self.assertEqual({a[3] for a,kw in rows},{'fixed_base_rollout'})
        self.assertTrue(all(kw['role']=='student_vs_privileged_teacher' for a,kw in rows))
        self.assertTrue(all('_to_initial' not in a[0] for a,kw in rows))
        self.assertEqual(len([k for k in neighbors if k.endswith('__sample_ids')]),2)
        undefined=[]
        manifest,_=summarize_student_teacher({}, {},dict(knn_k=[2],rsa_max_pairs=100,seed=7),
                                             lambda *a,**kw:undefined.append((a,kw)),
                                             [('fixed_base_rollout','heldout_probe',1,'response_128')])
        self.assertEqual(manifest[0]['n_valid'],0)
        self.assertTrue(all(a[2]==0 and np.isnan(a[1]) for a,kw in undefined))
        del meta['heldout_probe__privileged_teacher']
        with self.assertRaisesRegex(ValueError,'Missing same-checkpoint teacher'):
            summarize_student_teacher(arrays,meta,dict(knn_k=[2],rsa_max_pairs=100,seed=7),lambda *a,**kw:None)

    def test_real_checkpoint_extraction_and_cache_keep_both_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fixture'
            smoke(root,compare_teacher=True)
            for step in (0,1):
                directory=root/'analysis/steps'/f'step_{step:06d}'
                rows=read_jsonl(directory/'metrics.jsonl')
                direct=[r for r in rows if r['metric']=='rsa_rho_student_teacher']
                self.assertEqual(len(direct),8)  # Two splits x two anchors x two blocks.
                self.assertTrue(all(r['role']=='student_vs_privileged_teacher' for r in direct))
                self.assertTrue(all(r['value'] is not None for r in direct))
                self.assertTrue(any(r['value']<.999 for r in direct))
                self.assertTrue(any(r['metric']=='rsa_rho_to_initial' and r['role']=='student' for r in rows))
                self.assertTrue(any(r['metric']=='rsa_rho_to_initial' and r['role']=='privileged_teacher' for r in rows))
                meta=json.loads((directory/'anchors.json').read_text())
                with np.load(directory/'anchors.npz',allow_pickle=False) as data:
                    for pair in json.loads((directory/'student_teacher_comparisons.json').read_text()):
                        s,t=pair['student_key'],pair['teacher_key']
                        self.assertEqual(meta[s]['sample_ids'],meta[t]['sample_ids'])
                        self.assertTrue(all(a<b for a,b in zip(meta[s]['position_indices'],meta[t]['position_indices'])))
                        hs,ht=data[s].astype(np.float64),data[t].astype(np.float64)
                        hn=hs/np.linalg.norm(hs,axis=1)[:,None]
                        tn=ht/np.linalg.norm(ht,axis=1)[:,None]
                        tri=np.triu_indices(len(hn),1)
                        expected=spearmanr((hn@hn.T)[tri],(tn@tn.T)[tri]).statistic
                        row=next(r for r in direct if (r['split'],r['layer'],r['anchor'])==
                                 (pair['split'],pair['layer'],pair['anchor']))
                        self.assertAlmostEqual(row['value'],expected,places=6)
                completed=json.loads((directory/'complete.json').read_text())
                self.assertIn('student_teacher_comparisons.json',completed['files'])
                self.assertIn('student_teacher_neighbors.npz',completed['files'])

    def test_report_exports_direct_role_and_layer_by_step_figures(self):
        from analysis.report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in (0,1):
                directory=root/'opsd/analysis/steps'/f'step_{step:06d}'
                write_json(directory/'complete.json',dict(identity=dict(probe_manifest_hash='same_frozen_questions')))
                write_json(directory/'run_manifest.json',dict(identity=dict(step=step),hooks=dict(layers=2),
                           config=dict(run_id='paired_plot_fixture',table_format='jsonl',
                                       monitor=dict(plot_full_rsm_steps=[],plot_relative_depths=[0.]))))
                write_jsonl(directory/'curvature.jsonl',[])
                write_jsonl(directory/'metrics.jsonl',[
                    dict(optimizer_step=step,run_id='paired_plot_fixture',pool='fixed_base_rollout',
                         split='heldout_probe',role='student_vs_privileged_teacher',anchor='prompt_last',
                         layer=layer,metric='rsa_rho_student_teacher',value=.5+.1*step+.01*layer,
                         n_valid=8,status='ok') for layer in (1,2)])
            output=report(root)
            self.assertIn('student_vs_privileged_teacher',(output/'metrics.csv').read_text())
            manifest=json.loads((output/'report_manifest.json').read_text())
            direct=[p for p in manifest['plots'] if 'rsa_rho_student_teacher' in p]
            self.assertEqual(len(direct),2)
            self.assertTrue(any('__heatmap' in p for p in direct))
            self.assertTrue(all((output/p).stat().st_size>1000 for p in direct))


if __name__=='__main__':
    unittest.main()
