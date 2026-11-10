"""Audited execution-only overrides for resuming an existing, frozen OPSD experiment.

Analysis identities and scientific configuration stay immutable. Effective Hydra
configuration and an explicit execution profile record the actual microbatch and
memory settings. No launch/config hash guards are disabled or rewritten.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .common import ROOT, checkpoint_identity, file_hash, provenance, read_jsonl, write_json
from .job_control import record_progress


ALLOWED = {
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 'microbatch',
    'actor_rollout_ref.model.enable_activation_offload': 'boolean',
    'actor_rollout_ref.actor.fsdp_config.optimizer_offload': 'boolean',
    'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 'token_budget',
    'actor_rollout_ref.rollout.max_num_batched_tokens': 'token_budget',
    '+opsd.diagnostic_token_budget': 'token_budget',
}


def build_command(original, overrides, checkpoint, stop_at_step=None):
    if len(original) < 3 or original[1:3] != ['-m', 'verl.trainer.main_ppo']:
        raise ValueError('Expected the recorded verl training command')
    for key, value in overrides.items():
        kind = ALLOWED.get(key)
        if kind is None:
            raise ValueError('Not an execution-only override: '+key)
        if kind == 'boolean' and not isinstance(value, bool):
            raise ValueError('Expected boolean: '+key)
        if kind != 'boolean' and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError('Expected integer: '+key)
        if kind == 'microbatch' and value not in (1, 2, 4, 8):
            raise ValueError('Untested microbatch size')
        if kind == 'token_budget' and not 2560 <= value <= 16384:
            raise ValueError('Token budget outside the validated bounds')
    changes = {**overrides, 'trainer.resume_mode':'resume_path', 'trainer.resume_from_path':str(checkpoint),
               'trainer.del_local_ckpt_after_load':False}
    if stop_at_step is not None:
        planned = next((json.loads(item.split('=', 1)[1]) for item in original[3:]
                        if item.startswith('trainer.total_training_steps=')), None)
        saved = int(str(checkpoint).rsplit('global_step_', 1)[1])
        if (isinstance(stop_at_step, bool) or not isinstance(stop_at_step, int)
                or planned is None or not saved < stop_at_step <= planned):
            raise ValueError('Stop step must be after the saved step and within the original budget')
        changes['trainer.total_training_steps'] = stop_at_step
    command, pending = list(original[:3]), dict(changes)
    for item in original[3:]:
        key = item.split('=', 1)[0]
        if key in pending:
            command.append(key+'='+json.dumps(pending.pop(key), ensure_ascii=False))
        else:
            command.append(item)
    command.extend(key+'='+json.dumps(value, ensure_ascii=False) for key,value in pending.items())
    return command


def verify_files(files):
    for name, expected in files.items():
        if not Path(name).is_file() or file_hash(name) != expected:
            raise ValueError('Resume input changed: '+name)


def run(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    work = Path(profile['workdir'])
    attempt = profile_path.parent
    if profile['schema'] != 1 or profile['step'] < 1:
        raise ValueError('Invalid resume profile')
    verify_files(profile['immutable_files'])
    verify_files(profile['checkpoint_files'])
    current = json.loads((work/'config.json').read_text())
    from analysis.common import source_hashes
    frozen = json.loads((work/'shared/probe/complete.json').read_text())
    if frozen['identity']['sources'] != source_hashes():
        raise ValueError('Analysis sources changed; old analyses cannot be silently reused')
    if checkpoint_identity(current['model']['path']) != frozen['identity']['checkpoint_hash']:
        raise ValueError('Base model changed')
    if provenance(current)['source_hashes'] != profile['execution_sources']:
        raise ValueError('Execution sources changed after the profile was sealed')
    tracker = work/'opsd/train/checkpoints/latest_checkpointed_iteration.txt'
    if int(tracker.read_text().strip()) != profile['step']:
        raise ValueError('Latest checkpoint moved; create a fresh resume profile')
    metrics = read_jsonl(work/'opsd/train/metrics.jsonl')
    if not metrics or max(row['step'] for row in metrics) != profile['step']:
        raise ValueError('Metrics/checkpoint boundary mismatch; do not discard completed updates')
    checkpoint = work/'opsd/train/checkpoints'/f"global_step_{profile['step']}"
    if not (checkpoint/'data.pt').is_file():
        raise ValueError('Missing saved dataloader state')
    original = json.loads((attempt/'original_command.json').read_text())
    target_steps = profile.get('stop_at_step', current['training']['steps'])
    command = build_command(original, profile['overrides'], checkpoint, profile.get('stop_at_step'))
    from .job_control import training_target
    if training_target(work, current['training']['steps']) != target_steps:
        raise ValueError('Progress display target and execution stop step disagree')
    write_json(attempt/'effective_command.json', command)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    from .preflight import compose_training_config
    record_progress(work, 'resume_preflight', 'opsd', profile['step'])
    write_json(attempt/'preflight.json', compose_training_config(command, env, attempt))
    print('VERIFIED_RESUME_BOUNDARY', profile['step'], 'OVERRIDES', profile['overrides'], flush=True)
    record_progress(work, 'training', 'opsd')
    started = time.monotonic()
    with (work/'opsd/train/train.log').open('a') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    write_json(attempt/'execution_summary.json', dict(returncode=result.returncode,
               wall_seconds=time.monotonic()-started, checkpoint=str(checkpoint), profile_hash=file_hash(profile_path)))
    if result.returncode:
        raise RuntimeError(f'Resumed training exited {result.returncode}; see opsd/train/train.log')
    latest = read_jsonl(work/'opsd/train/metrics.jsonl')[-1]
    if latest['step'] != target_steps or not math.isfinite(latest['metrics']['actor/forward_kl']):
        raise RuntimeError('Resumed process exited without completing the training budget')
    from .compare import analyze
    record_progress(work, 'analysis_verification', 'opsd')
    analyze(current, 'opsd')
    from analysis.report import report
    record_progress(work, 'report')
    print(report(work), flush=True)
    write_json(work/'training_completed.json', dict(completed_steps=latest['step'], target_steps=target_steps,
               profile=str(profile_path), profile_hash=file_hash(profile_path), completed_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    run(parser.parse_args().profile)


if __name__ == '__main__':
    main()
