"""Exercise real detached processes and progress inspection without model training."""
import contextlib
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.research import compare, job_control


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class JobControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.started = []
        self.addCleanup(self.stop_fixtures)

    def stop_fixtures(self):
        for workdir in self.started:
            directory = job_control._current(workdir)
            state = job_control._read(directory/'state.json',{}) if directory else {}
            if job_control._alive(state.get('supervisor')) and state.get('status') not in job_control.TERMINAL:
                job_control.stop_session(workdir)
                self.wait_for(lambda:job_control.status(workdir)['status'] in job_control.TERMINAL)

    def wait_for(self, predicate, timeout=8):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        self.fail('Timed out waiting for fixture process')

    def start(self, workdir, code):
        self.started.append(workdir)
        return job_control.start_session(workdir,[sys.executable,'-u','-c',code])

    def test_survives_launcher_session_disconnect_and_explicit_sighup(self):
        workdir = self.root/'disconnect'
        self.started.append(workdir)
        counter = self.root/'counter'
        finish = self.root/'finish'
        task = ("import pathlib,time; "
                "from scripts.research.job_control import record_progress; "
                f"record_progress({str(workdir)!r},'training','opsd'); "
                f"counter=pathlib.Path({str(counter)!r}); finish=pathlib.Path({str(finish)!r}); "
                "print('fixture started',flush=True)\n"
                "for i in range(150):\n"
                " tmp=counter.with_suffix('.tmp'); tmp.write_text(str(i)); tmp.replace(counter)\n"
                " if finish.exists(): break\n"
                " time.sleep(0.1)\n"
                "print('fixture finished',flush=True)\n")
        launcher_code = ("import json,time; from scripts.research.job_control import start_session; "
                         f"print(json.dumps(start_session({str(workdir)!r},{[sys.executable,'-u','-c',task]!r})),flush=True); "
                         "time.sleep(60)")
        launcher = subprocess.Popen([sys.executable,'-u','-c',launcher_code],cwd=job_control.ROOT,
                                    start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertTrue(select.select([launcher.stdout],[],[],8)[0])
            json.loads(launcher.stdout.readline())
            self.wait_for(counter.exists)
            report = job_control.status(workdir)
            self.assertEqual(report['phase'],'training')
            supervisor = report['supervisor']['pid']
            child = report['child']['pid']
            self.assertNotEqual(os.getsid(supervisor),os.getsid(launcher.pid))
            self.assertEqual(os.getsid(supervisor),supervisor)
            self.assertEqual(os.getsid(child),child)
            before = int(counter.read_text())
            os.killpg(launcher.pid,signal.SIGHUP)
            launcher.wait(timeout=3)
            os.kill(supervisor,signal.SIGHUP)
            os.kill(child,signal.SIGHUP)
            self.wait_for(lambda:int(counter.read_text())>before)
            self.assertEqual(job_control.status(workdir)['status'],'running')
            finish.write_text('done')
            self.wait_for(lambda:job_control.status(workdir)['status']=='completed')
            report = job_control.status(workdir)
            self.assertEqual(report['returncode'],0)
            self.assertIn('fixture finished',Path(report['console_log']).read_text())
        finally:
            if launcher.poll() is None:
                os.killpg(launcher.pid,signal.SIGTERM)
                launcher.wait(timeout=3)
            launcher.stdout.close()
            launcher.stderr.close()

    def test_duplicate_launch_is_blocked_and_stop_only_targets_managed_task(self):
        workdir = self.root/'duplicate'
        self.start(workdir,"import time; print('fixture alive',flush=True); time.sleep(30)")
        self.wait_for(lambda:job_control.status(workdir)['status']=='running')
        with self.assertRaisesRegex(RuntimeError,'active task'):
            self.start(workdir,"raise SystemExit('should never run')")
        response = job_control.stop_session(workdir)
        self.assertEqual(response['status'],'stop_requested')
        self.wait_for(lambda:job_control.status(workdir)['status']=='stopped')

    def test_exit_code_and_logs_report_failure_instead_of_completion(self):
        workdir = self.root/'failure'
        self.start(workdir,"print('fixture failure',flush=True); raise SystemExit(7)")
        self.wait_for(lambda:job_control.status(workdir)['status']=='failed')
        report = job_control.status(workdir)
        self.assertEqual(report['returncode'],7)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            job_control.show_logs(workdir,lines=10,follow=True)
        self.assertIn('fixture failure',output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(job_control.main(['status','--workdir',str(workdir),'--watch']),1)

    def test_restart_preserves_attempt_logs_and_releases_lock(self):
        workdir = self.root/'history'
        self.start(workdir,"print('first attempt'); raise SystemExit(7)")
        self.wait_for(lambda:job_control.status(workdir)['status']=='failed')
        first = job_control._current(workdir)
        self.wait_for(lambda:not job_control._alive(job_control._read(first/'state.json')['supervisor']))
        self.start(workdir,"print('second attempt')")
        self.wait_for(lambda:job_control.status(workdir)['status']=='completed')
        second = job_control._current(workdir)
        self.assertNotEqual(first,second)
        self.assertIn('first attempt',(first/'console.log').read_text())
        self.assertIn('second attempt',(second/'console.log').read_text())

    def test_progress_uses_metrics_checkpoint_and_analysis_not_process_liveness_alone(self):
        workdir = self.root/'progress'
        write(workdir/'config.json',dict(training=dict(steps=200),monitor=dict(optimizer_steps=[0,1,200])))
        metrics = workdir/'opsd/train/metrics.jsonl'
        write(metrics,dict(step=3,budget={'wall_seconds':12},metrics={'actor/forward_kl':0.25}))
        with metrics.open('a') as stream:
            stream.write('{"step":4,"metrics":')
        tracker = workdir/'opsd/train/checkpoints/latest_checkpointed_iteration.txt'
        tracker.parent.mkdir(parents=True)
        tracker.write_text('2')
        for step,status in ((0,'complete'),(1,'pending')):
            name = f'step_{step:06d}.json'
            write(workdir/'opsd/jobs'/name,dict(optimizer_step=step))
            write(workdir/'opsd/job_status'/name,dict(status=status))
        report = job_control.status(workdir)
        item = report['methods']['opsd']
        self.assertEqual(report['status'],'not_started')
        self.assertEqual(item['completed_steps'],3)
        self.assertEqual(item['checkpoint_step'],2)
        self.assertEqual(item['percent'],1.5)
        self.assertEqual(item['analyzed_steps'],[0])
        self.assertEqual(item['latest_metrics']['actor/forward_kl'],0.25)

    def test_restart_does_not_present_previous_attempt_metrics_as_current_progress(self):
        workdir = self.root/'restart'
        write(workdir/'config.json',dict(training=dict(steps=200)))
        metrics = workdir/'opsd/train/metrics.jsonl'
        write(metrics,dict(step=20,metrics={}))
        old = time.time()-60
        os.utime(metrics,(old,old))
        tracker = workdir/'opsd/train/checkpoints/latest_checkpointed_iteration.txt'
        tracker.parent.mkdir(parents=True)
        tracker.write_text('10')
        control = workdir/'.session'
        write(control/'current.json',dict(attempt='fixture'))
        directory = control/'attempts/fixture'
        write(directory/'request.json',dict(methods=['opsd']))
        write(directory/'state.json',dict(status='running',supervisor=job_control._identity(os.getpid()),
                                         started_unix=time.time()))
        write(directory/'progress.json',dict(phase='analyzing',method='opsd',optimizer_step=10))
        report = job_control.status(workdir)
        self.assertEqual(report['methods']['opsd']['completed_steps'],10)
        self.assertTrue(report['methods']['opsd']['metrics_from_previous_attempt'])
        self.assertEqual(report['analysis_step'],10)

    def test_pid_reuse_is_not_reported_running_or_signaled(self):
        workdir = self.root/'reused'
        write(workdir/'.session/current.json',dict(attempt='fixture'))
        fake = dict(job_control._identity(os.getpid()),start_ticks='not-the-same-process')
        write(workdir/'.session/attempts/fixture/state.json',dict(status='running',supervisor=fake))
        self.assertEqual(job_control.status(workdir)['status'],'lost')
        with patch('os.kill') as kill:
            with self.assertRaisesRegex(ValueError,'No verified live supervisor'):
                job_control.stop_session(workdir)
            kill.assert_not_called()

    def test_default_execute_detaches_and_child_explicitly_selects_foreground(self):
        workdir = self.root/'dispatch'
        config = dict(workdir=str(workdir))
        with (patch.object(compare,'load',return_value=config),patch.object(compare,'train') as train,
              patch.object(job_control,'start_session',return_value={'status':'starting'}) as start,
              contextlib.redirect_stdout(io.StringIO())):
            compare.main(['train','--workdir',str(workdir),'--execute'])
        train.assert_not_called()
        command = start.call_args.args[1]
        self.assertEqual(command[-1],'--foreground')
        self.assertIn('--execute',command)
        self.assertEqual(start.call_args.args[2],('opsd',))

    def test_background_run_makes_relative_input_paths_absolute_before_detaching(self):
        config = dict(workdir=str(self.root/'run'))
        with (contextlib.chdir(self.root),patch.object(compare,'configure',return_value=config),
              patch.object(job_control,'start_session',return_value={'status':'starting'}) as start,
              contextlib.redirect_stdout(io.StringIO())):
            compare.main(['run','--workdir','run','--config','config.json','--model-dir','Base',
                          '--model-revision','fixture-model','--input','native.parquet',
                          '--dataset-revision','fixture-data','--execute'])
        command = start.call_args.args[1]
        for flag,name in (('--workdir','run'),('--config','config.json'),('--model-dir','Base'),('--input','native.parquet')):
            self.assertEqual(command[command.index(flag)+1],str(self.root/name))

    def test_status_shell_entry_works_outside_repo_and_does_not_create_run(self):
        workdir = self.root/'not_started'
        result = subprocess.run(['bash',str(job_control.ROOT/'run_base_comparison.sh'),'status',
                                 '--workdir',str(workdir),'--json'],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'not_started')
        self.assertFalse(workdir.exists())


if __name__=='__main__':
    unittest.main()
