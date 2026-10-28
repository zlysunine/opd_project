"""User-authorized final shutdown, gated on verified publication and finished jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent
LOG=ROOT/'research_runs/server_handoff_20261028'
OUTPUT=ROOT/'results/20261028_qwen17b_opsd20'


def read(path):
    return json.loads(Path(path).read_text())


def record(state, **extra):
    value=dict(state=state,utc=datetime.now(timezone.utc).isoformat(),**extra)
    path=LOG/'shutdown_receipt.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--delay',type=int,default=10)
    args=parser.parse_args()
    def git(*command):
        return subprocess.check_output(['git',*command],cwd=REPO,text=True,timeout=45).strip()
    if git('rev-parse','HEAD')!=args.commit or git('status','--porcelain'):
        raise ValueError('Commit changed or working tree is dirty')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=args.commit:raise ValueError('Remote main does not contain the published commit')
    receipt=read(LOG/'formal_verification.json')
    if receipt.get('status')!='completed' or not receipt['quality_gates_passed']:
        raise ValueError('Final experiment is not fully verified')
    if not read(LOG/'final_code_validation.json')['passed']:
        raise ValueError('Regression validation did not pass')
    for workdir in [Path(receipt['workdir']),LOG/'pipeline_q70_n50']:
        attempt=read(workdir/'.session/current.json')['attempt']
        state=read(workdir/'.session/attempts'/attempt/'state.json')
        if state['status']!='completed' or state['returncode']!=0:
            raise ValueError(f'Job still running or failed: {workdir}')
    manifest=read(OUTPUT/'artifact_manifest.json')
    for name,expected in manifest['files'].items():
        path=OUTPUT/name
        if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError(f'Published file changed: {name}')
    record('scheduled',commit=args.commit,remote_main=remote,delay_seconds=args.delay,
           command=['/bin/bash','/usr/bin/shutdown'],models_uploaded=False)
    time.sleep(args.delay)
    subprocess.run(['sync'],check=True)
    record('command_invoked',commit=args.commit,remote_main=remote,
           command=['/bin/bash','/usr/bin/shutdown'],models_uploaded=False,
           note='A local command invocation cannot independently confirm the external control-panel state.')
    subprocess.run(['sync'],check=True)
    result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],check=False)
    record('command_returned',commit=args.commit,returncode=result.returncode,
           note='Control-panel power state is not independently observable from this process.')
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        record('blocked',error=str(exc))
        raise
