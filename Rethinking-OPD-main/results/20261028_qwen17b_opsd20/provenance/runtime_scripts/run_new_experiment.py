"""Run the documented pilot, verify it, then launch the authorized 20-step run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from scripts.research.common import digest, write_json
from analysis.prepare_probe import normalized_question
from scripts.research.job_control import record_progress

ROOT = Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
LOG = ROOT / 'research_runs/server_handoff_20261028'
PILOT = ROOT / 'research_runs/base_opsd_20261028_acceptance_2step_q70_n50'
COORD = LOG / 'pipeline_q70_n50'
PREPARATION_ATTEMPT = '20261027T233150Z_a33bbd8f'
FORMAL_TEMPLATE = LOG / 'formal_template_20step.json'
stop_monitor = threading.Event()
started = time.time()
phase_started = started
phases = []


def phase(name, **details):
    global phase_started
    now = time.time()
    phases.append(dict(phase=name, started_unix=now, **details))
    phase_started = now
    write_json(LOG/'pipeline_status.json', dict(status='running', phases=phases, updated_unix=now, **details))
    record_progress(COORD, name, 'opsd')
    print(name, json.dumps(details, ensure_ascii=False), flush=True)


def extraction_summary(workdir, steps):
    from collections import Counter
    targets = [('baseline', workdir/'shared/quality/baseline')]
    targets += [(f'step_{step:06d}', workdir/'opsd/analysis/quality'/f'step_{step:06d}') for step in steps]
    records, missing, passed = [], [], True
    for label, directory in targets:
        quality = directory/'quality.json'
        responses = directory/'responses.jsonl'
        if not quality.is_file() or not responses.is_file():
            missing.append(label)
            continue
        q = json.loads(quality.read_text())
        rows = [json.loads(line) for line in responses.open() if line.strip()]
        passed = passed and q['passed']
        for split in sorted({row['split'] for row in rows}):
            selected = [row for row in rows if row['split'] == split]
            n = len(selected)
            failures = sum(bool(row['extraction_failed']) for row in selected)
            records.append(dict(stage=label, split=split, n=n, extraction_successes=n-failures,
                extraction_failures=failures, extraction_success_rate=(n-failures)/n,
                extraction_failure_rate=failures/n,
                truncation_rate=sum(bool(row['truncated']) for row in selected)/n,
                accuracy=sum(bool(row['correct']) for row in selected)/n,
                parse_status_counts=dict(Counter(row.get('parse_status','legacy') for row in selected)),
                quality_gate_applies=split in q['settings']['splits'],
                quality_gate_passed=q['passed'] if split in q['settings']['splits'] else None,
                source=str(responses)))
    return dict(workdir=str(workdir), required_heldout_success_rate=1-json.loads((workdir/'config.json').read_text())['monitor']['quality_gate']['max_extraction_failure_rate'],
                expected_stages=[label for label,_ in targets], pending_stages=missing,
                completed=not missing, all_gates_passed=not missing and passed,
                rates=records, updated_unix=time.time())


def resources():
    with (LOG/'resource_samples.jsonl').open('a', buffering=1) as out:
        while not stop_monitor.is_set():
            row = dict(unix=time.time(), phase=phases[-1]['phase'] if phases else 'starting',
                       disk_free_bytes=shutil.disk_usage(ROOT).free)
            try:
                row['cgroup_memory_bytes'] = int(Path('/sys/fs/cgroup/memory.current').read_text())
                result = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                row['gpu'] = result.stdout.strip()
                row['gpu_query_returncode'] = result.returncode
                write_json(LOG/'acceptance_extraction_success_rates.json', extraction_summary(PILOT, [0,1,2]))
            except Exception as exc:
                row['sampling_error'] = str(exc)
            out.write(json.dumps(row)+'\n')
            stop_monitor.wait(15)


def wait_attempt(workdir, attempt=None):
    last_print = 0
    while True:
        pointer = json.loads((workdir/'.session/current.json').read_text())
        current = attempt or pointer['attempt']
        state = json.loads((workdir/'.session/attempts'/current/'state.json').read_text())
        if time.time()-last_print >= 60:
            print('SESSION', workdir.name, state['status'], 'heartbeat', state.get('heartbeat_utc'), flush=True)
            last_print = time.time()
        if state['status'] in ('completed','failed','stopped','lost'):
            if state['status'] != 'completed' or state.get('returncode') != 0:
                raise RuntimeError(f"{workdir.name}: {state['status']}, returncode={state.get('returncode')}; "
                                   f"inspect {workdir/'.session/attempts'/current/'console.log'}")
            time.sleep(1)
            return state
        pid = (state.get('supervisor') or {}).get('pid')
        if not pid or not Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Session supervisor disappeared: {workdir}')
        time.sleep(15)


def arguments(config, workdir):
    c = json.loads((PILOT/'config.json').read_text())
    return ['bash',str(ROOT/'run_base_comparison.sh'),'--config',str(config),
            '--model-dir',c['model']['path'],'--model-revision',c['model']['revision'],
            '--input',c['dataset']['input'],'--dataset-revision',c['dataset']['revision'],
            '--workdir',str(workdir),'--method','opsd']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_completed(workdir, final_step, steps):
    checkpoints = workdir/'opsd/train/checkpoints'
    assert int((checkpoints/'latest_checkpointed_iteration.txt').read_text()) == final_step
    checkpoint = checkpoints/f'global_step_{final_step}'
    for relative in ['data.pt','research_checkpoint.json','actor/model_world_size_1_rank_0.pt',
                     'actor/optim_world_size_1_rank_0.pt','actor/extra_state_world_size_1_rank_0.pt',
                     'actor/lora_adapter/adapter_model.safetensors']:
        assert (checkpoint/relative).is_file(), relative
    assert json.loads((workdir/'shared/quality/baseline/quality.json').read_text())['passed']
    for step in steps:
        directory = workdir/'opsd/analysis/steps'/f'step_{step:06d}'
        receipt = json.loads((directory/'complete.json').read_text())
        for name, expected in receipt['files'].items():
            assert sha(directory/name) == expected, f'{directory/name}: hash mismatch'
        assert (directory/'student_teacher_comparisons.json').is_file()
        assert (directory/'student_teacher_neighbors.npz').is_file()
        assert json.loads((workdir/'opsd/analysis/quality'/f'step_{step:06d}'/'quality.json').read_text())['passed']
    html = workdir/'report/OPSD_report.html'
    assert html.is_file() and html.stat().st_size > 10000
    assert not (workdir/'grpo/train').exists()
    quality = extraction_summary(workdir, steps)
    assert quality['all_gates_passed'], 'Extraction success-rate checks did not all pass'
    from pilot_validation import validate_training_metrics
    training_metrics = validate_training_metrics(workdir, final_step)
    write_json(workdir/'opsd/train/metric_validation.json',training_metrics)
    return dict(workdir=str(workdir), checkpoint=str(checkpoint), report=str(html),
                verified_steps=steps, quality_gates_passed=True, extraction_success_rates=quality,
                training_metrics_validation=training_metrics)


def main():
    phase('waiting_baseline_50', workdir=str(PILOT))
    from adopt_baseline_prefix import wait_and_adopt
    wait_and_adopt()
    phase('acceptance_training_and_analysis', workdir=str(PILOT))
    pointer=PILOT/'.session/current.json'
    previous=None
    if pointer.is_file():
        attempt=json.loads(pointer.read_text())['attempt']
        previous=json.loads((PILOT/'.session/attempts'/attempt/'state.json').read_text())
    if previous is None:
        subprocess.run(arguments(LOG/'acceptance_config_q70_n50.json', PILOT)+['--execute'], cwd=ROOT, check=True)
    elif previous['status'] in ('failed','stopped','lost'):
        raise RuntimeError('Pilot has failed; preserve evidence and diagnose before retrying')
    wait_attempt(PILOT)
    pilot_receipt = verify_completed(PILOT, 2, [0,1,2])
    write_json(LOG/'acceptance_verification.json', pilot_receipt)

    phase('formal_inputs_reuse', workdir=str(PILOT))
    from reuse_full_pilot_inputs import prepare_formal_from_pilot
    reuse=prepare_formal_from_pilot()
    formal=Path(reuse['formal'])
    config_path=Path(reuse['config'])
    train_n=reuse['counts']['train']
    write_json(LOG/'formal_selection_reuse.json',reuse)
    phase('formal_plan', workdir=str(formal), train_n=train_n)
    with (LOG/'formal-actual-plan.json').open('w') as out:
        subprocess.run(arguments(config_path, formal), cwd=ROOT, stdout=out, check=True)
    phase('formal_training_and_analysis', workdir=str(formal), train_n=train_n)
    subprocess.run(arguments(config_path, formal)+['--execute'], cwd=ROOT, check=True)
    wait_attempt(formal)
    final = verify_completed(formal, 20, [0,1,5,10,20])
    final.update(status='completed', phases=phases, elapsed_seconds=time.time()-started)
    write_json(LOG/'pipeline_status.json', final)
    write_json(LOG/'formal_verification.json', final)
    record_progress(COORD, 'completed', 'opsd', 20)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    threading.Thread(target=resources, daemon=True).start()
    try:
        main()
    except BaseException as exc:
        write_json(LOG/'pipeline_status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                    phases=phases, elapsed_seconds=time.time()-started))
        traceback.print_exc()
        raise
    finally:
        stop_monitor.set()
