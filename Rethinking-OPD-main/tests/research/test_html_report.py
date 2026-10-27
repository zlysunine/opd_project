"""Portable reports preserve measurement scope, roles, token weights and missing data."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from analysis.html_report import build, enrich, read_run, write_html, Scripts
from scripts.research.build_metric_report import metric_entry, scalar_point, sha
from scripts.research.common import write_json, write_jsonl


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external=[]

    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value and not value.startswith(('data:','#')):
                self.external.append((tag,key,value))


class HtmlReportTests(unittest.TestCase):
    def fixture(self,root):
        for step in (0,2):
            directory=root/'opsd/analysis/steps'/f'step_{step:06d}'
            def row(name,value,role='student',**extra):
                return dict(optimizer_step=step,metric=name,value=value,role=role,pool='fixed_base_rollout',
                            split='heldout_probe',anchor=None,layer=None,n_valid=4,status='ok',**extra)
            measurements=[row('student_entropy_nats',2.5,distribution_scope='all_valid_tokens',aggregation='token_mean'),
                          row('teacher_entropy_nats',2.2,distribution_scope='all_valid_tokens',aggregation='token_mean'),
                          row('rsa_rho_student_teacher',.3+.1*step,'student_vs_privileged_teacher'),
                          row('rsa_rho_to_initial',1.-step*.01),
                          row('rsa_rho_to_initial',1.-step*.02,'privileged_teacher'),
                          row('my_added_metric',step+7)]
            for measurement in measurements:
                if measurement['metric'].startswith('rsa_'):
                    measurement.update(layer=1,anchor='response_128')
            write_jsonl(directory/'metrics.jsonl',measurements)
            write_jsonl(directory/'distributions.jsonl',[
                dict(optimizer_step=step,pool='fixed_base_rollout',split='heldout_probe',role='student',
                     response_position='all_valid_tokens',distribution_scope='all_valid_tokens',
                     aggregation='token_mean',n_valid_tokens=n,student_entropy_nats=h,
                     metric_sums={'student_entropy_nats':h*n},metric_token_counts={'student_entropy_nats':n})
                for h,n in [(1.,1),(3.,3)]])
            write_json(directory/'run_manifest.json',dict(identity=dict(step=step,probe_manifest_hash='same_probe'),
                       config=dict(method='opsd',run_id='fixture',model=dict(name_or_path='TINY_HTML_FIXTURE'),
                                   training=dict(status='NOT_RUN',steps=2))))
            write_json(directory/'complete.json',dict(identity=dict(probe_manifest_hash='same_probe'),
                       files={'metrics.jsonl':sha(directory/'metrics.jsonl')}))
        write_jsonl(root/'opsd/train/metrics.jsonl',[
            dict(step=s,metrics={'actor/forward_kl':.1/s}) for s in (1,2,3)])

    def test_raw_token_mean_stays_separate_from_question_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            payload,_,_=read_run(root)
            payload=enrich(payload)
            main=next(e for e in payload['metrics'] if e['name']=='student_entropy_nats' and e['kind']=='analysis')
            sample=next(e for e in payload['metrics'] if e['name']=='student_entropy_nats' and e['kind']=='distribution')
            self.assertEqual(main['variants'][0]['points'][0][2],2.5)
            self.assertEqual(main['variants'][0]['points'][0][3],4)
            self.assertEqual(sample['variants'][0]['points'][0][2],2.)
            self.assertEqual(main['variants'][0]['dims']['aggregation'],'token_mean')
            self.assertEqual(sample['variants'][0]['dims']['aggregation'],'question_distribution_summary')
            self.assertEqual(payload['meta']['selected_step'],2)
            self.assertEqual(payload['meta']['training_steps'],[1,2])  # Uncheckpointed step3 is excluded.
            self.assertEqual(payload['meta']['mode'],'cpu_fixture')
            self.assertTrue(any(e['name']=='my_added_metric' for e in payload['metrics']))

    def test_comparison_roles_and_scope_are_not_mislabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            payload=enrich(read_run(root)[0])
            coverage={c['metric']:c for c in payload['meta']['coverage']}
            self.assertTrue(coverage['rsa_rho_student_teacher']['available'])
            self.assertTrue(coverage['student_entropy_nats']['available'])
            self.assertFalse(coverage['overlap_token_advantage_nats']['available'])
            initial=next(e for e in payload['metrics'] if e['name']=='rsa_rho_to_initial')
            self.assertEqual({v['dims']['comparison'] for v in initial['variants']},
                             {'student_vs_initial_student','teacher_vs_initial_teacher'})
            plan={row['metric']:row for row in payload['meta']['metric_plan']}
            self.assertEqual(plan['top-16 overlap ratio']['student_initial']['fields'],['token_top16_overlap_to_initial'])
            self.assertEqual(plan['top-16 overlap ratio']['student_teacher']['fields'],['token_top16_overlap'])
            self.assertEqual(plan['RSA（Spearman）']['student_initial']['roles'],['student'])
            self.assertEqual(plan['RSA（Spearman）']['student_teacher']['fields'],['rsa_rho_student_teacher'])
            self.assertEqual(plan['轨迹曲率']['student_teacher']['status'],'separate_curves')

    def test_historical_selected_positions_cannot_claim_new_metrics(self):
        entry=metric_entry('student_entropy_nats','analysis',[
            dict(dims=dict(role='student',pool='fixed_base_rollout'),points=[scalar_point(20,None,1.,512,512,'ok')])])
        payload=enrich(dict(meta=dict(model='Qwen',selected_step=20),metrics=[entry],gallery=[]))
        self.assertEqual(payload['metrics'][0]['variants'][0]['dims']['distribution_scope'],'legacy_selected_positions')
        self.assertFalse(any(c['available'] for c in payload['meta']['coverage']))
        self.assertEqual(payload['meta']['counts']['analysis'],1)

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            path=root/'opsd/analysis/steps/step_000002/metrics.jsonl'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Completed analysis file changed'):
                read_run(root)

    def test_all_images_and_scripts_are_embedded_and_json_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'plot.png').write_bytes(b'png-fixture')
            entry=metric_entry('my_metric','analysis',[dict(dims={},points=[scalar_point(0,None,1.,1,1,'ok')])])
            entry['png']='plot.png'
            payload=dict(meta=dict(model='</script><img src="bad">',selected_step=0,monitor_steps=[0],training_steps=[]),
                         metrics=[entry],gallery=[dict(name='RSM',path='plot.png')])
            target=root/'portable.html'
            with patch('analysis.html_report.plotly_bundle',return_value='/* bundled Plotly test fixture */'):
                result=write_html(payload,target,root,{'source':'fixture'})
            document=Document()
            text=target.read_text()
            document.feed(text)
            self.assertEqual(document.external,[])
            self.assertNotIn('</script><img src="bad">',text)
            self.assertIn('\\u003c/script\\u003e',text)
            self.assertTrue(result['gallery'][0]['path'].startswith('data:image/png;base64,'))
            self.assertTrue(result['metrics'][0]['png'].startswith('data:image/png;base64,'))
            self.assertIn('exportAll',text)

    def test_archived_export_can_be_rendered_without_server_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            entry=metric_entry('rsa_rho_to_initial','analysis',[
                dict(dims={'role':'student'},points=[scalar_point(20,1,.99,8,8,'ok')])])
            write_json(root/'data/report_data.json',dict(meta=dict(model='Archived model',selected_step=20,
                       monitor_steps=[0,20],training_steps=[]),metrics=[entry],gallery=[]))
            with patch('analysis.html_report.plotly_bundle',return_value='/* fixture */'):
                target,payload=build(root,root/'complete.html')
            self.assertTrue(target.is_file())
            self.assertEqual(payload['meta']['mode'],'archived_export')
            self.assertFalse(next(c for c in payload['meta']['coverage'] if c['metric']=='rsa_rho_student_teacher')['available'])

    @unittest.skipUnless(shutil.which('node'),'Node is needed to exercise application controls')
    def test_javascript_contexts_and_heatmap_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            with patch('analysis.html_report.plotly_bundle',return_value='/* fixture */'):
                target,payload=build(root,root/'report.html')
            parser=Scripts()
            parser.feed(target.read_text())
            app=root/'app.js'
            app.write_text(parser.scripts[-1])
            data=root/'payload.json'
            data.write_text(json.dumps(payload))
            result=subprocess.run(['node',str(Path(__file__).with_name('html_report_runtime.js')),str(data),str(app)],
                                  text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,msg=result.stdout+result.stderr)


if __name__=='__main__':
    unittest.main()
