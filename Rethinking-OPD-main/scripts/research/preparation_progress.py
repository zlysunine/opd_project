"""Observable preparation stages; only measured counts produce rates and ETAs."""
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class PreparationProgress:
    def __init__(self, path, interval=5.):
        self.path = Path(path)
        self.interval = interval
        self.started = time.time()
        self.phase_started = self.started
        self.last_write = 0.
        self.value = dict(schema=1, status='running', phase='identity_check', processed=0,
                          total=None, accepted=0, excluded=0, exclusion_counts={},
                          started_unix=self.started, phase_started_unix=self.started,
                          pid=os.getpid(), history=[])
        self.update(0, force=True)

    def phase(self, name, total=None, **details):
        now = time.time()
        previous = self.value
        history = previous['history'] + [dict(phase=previous['phase'], processed=previous['processed'],
                    total=previous['total'], accepted=previous.get('accepted'), excluded=previous.get('excluded'),
                    elapsed_seconds=max(0.,now-self.phase_started))]
        self.phase_started = now
        self.value = dict(previous, phase=name, processed=0, total=total,
                          current_problem_id=None, phase_started_unix=now, history=history)
        self.value.update(details)
        self.update(0, total, force=True)

    def update(self, processed, total=None, force=False, **details):
        self.value.update(processed=processed, **details)
        if total is not None:
            self.value['total'] = total
        now = time.time()
        if not force and now-self.last_write < self.interval and processed != self.value['total']:
            return
        elapsed = max(0.,now-self.phase_started)
        count, total = self.value['processed'], self.value['total']
        rate = count/elapsed if count > 0 and elapsed > 0 else None
        running = self.value['status'] == 'running'
        self.value.update(updated_unix=now, elapsed_seconds=max(0.,now-self.started),
                          phase_elapsed_seconds=elapsed,
                          percent=100*count/total if total else None,
                          rows_per_second=rate,
                          remaining_seconds=(total-count)/rate if running and total is not None and rate else None)
        write_json(self.path, self.value)
        self.last_write = now

    def finish(self, error=None):
        self.value['status'] = 'failed' if error is not None else 'completed'
        if error is not None:
            self.value['error'] = f'{type(error).__name__}: {error}'
        self.value['finished_unix'] = time.time()
        self.update(self.value['processed'], force=True)


@contextmanager
def preparation_workspace(parent):
    """Keep completed selection files when generation/quality later fails."""
    path = Path(tempfile.mkdtemp(prefix='.probe-', dir=parent))
    yield path  # Successful freeze renames it; failed freeze leaves it intact.


@lru_cache(maxsize=16)
def _audit_counts(path, mtime_ns, size):
    counts = Counter()
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                counts[row['exclusion_reason'] or 'accepted'] += 1
    return dict(counts)


def read_preparation(workdir):
    """Also show evidence from older runs without inventing intermediate counts."""
    shared = Path(workdir)/'shared'
    path = shared/'preparation_progress.json'
    if path.is_file():
        result = json.loads(path.read_text())
        generation = shared/'generation_progress.json'
        if result['phase'] in ('baseline_generation','fixed_pool_generation') and generation.is_file():
            generated = json.loads(generation.read_text())
            pool = 'live_rollout' if result['phase']=='baseline_generation' else 'fixed_base_rollout'
            if generated['pool']==pool and generated['updated_unix']>=result['phase_started_unix']:
                count,total = generated['completed'],generated['total']
                elapsed = generated['elapsed_seconds']
                result.update(processed=count,total=total,percent=100*count/total if total else None,
                              rows_per_second=count/elapsed if elapsed and count else None,
                              remaining_seconds=generated.get('remaining_seconds'),
                              updated_unix=generated['updated_unix'])
        quality = shared/'quality/baseline/quality.json'
        if result['phase']=='baseline_quality' and quality.is_file() and quality.stat().st_mtime>=result['phase_started_unix']:
            result['baseline_quality'] = json.loads(quality.read_text())
        return result
    audit = shared/'preparation_audit/eligibility.jsonl'
    if not audit.is_file():
        return None
    stat = audit.stat()
    counts = _audit_counts(str(audit), stat.st_mtime_ns, stat.st_size)
    total, accepted = sum(counts.values()), counts.get('accepted',0)
    result = dict(schema=1, source='legacy_artifacts', phase='selection', status='completed',
                  processed=total, total=total, percent=100. if total else None,
                  accepted=accepted, excluded=total-accepted,
                  exclusion_counts={k:v for k,v in counts.items() if k!='accepted'},
                  screening_completed=True, updated_unix=stat.st_mtime,
                  elapsed_seconds=None, rows_per_second=None, remaining_seconds=None)
    quality = shared/'quality/baseline/quality.json'
    generation = shared/'generation_progress.json'
    if quality.is_file():
        q = json.loads(quality.read_text())
        result.update(phase='baseline_quality', status='completed' if q['passed'] else 'failed',
                      baseline_quality=q, updated_unix=quality.stat().st_mtime)
        if not q['passed']:
            result['error'] = 'Baseline extraction quality gate failed; screening decisions are preserved.'
    elif generation.is_file():
        result.update(phase='baseline_generation',status='running')
    return result
