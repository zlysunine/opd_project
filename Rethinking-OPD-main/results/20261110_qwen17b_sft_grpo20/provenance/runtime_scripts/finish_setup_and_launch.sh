#!/usr/bin/env bash
set -euo pipefail
exec 9>/root/autodl-tmp/sft_grpo_setup/bootstrap.lock
flock -n 9
while kill -0 18113 2>/dev/null; do sleep 5; done
source /root/autodl-tmp/activate_sft_grpo.sh
python -m pip install --no-deps --no-build-isolation -e /root/autodl-tmp/opd_project/Rethinking-OPD-main/verl
python -m pip freeze > /root/autodl-tmp/sft_grpo_setup/final-pip-freeze.txt
python -u - <<'PY'
from scripts.research.preflight import check_environment
from pathlib import Path
import json
report=check_environment(1)
Path('/root/autodl-tmp/sft_grpo_setup/preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
assert report['ok'],report['errors']
PY
OMP_NUM_THREADS=2 python -m unittest discover -s tests/research -p test_controlled_comparison.py -v > /root/autodl-tmp/sft_grpo_setup/test-controlled.log 2>&1
OMP_NUM_THREADS=2 python -m unittest discover -s tests/research -p test_checkpoint_retention.py -v > /root/autodl-tmp/sft_grpo_setup/test-retention.log 2>&1
OMP_NUM_THREADS=2 python -m unittest discover -s tests/research -p test_forward_kl.py -v > /root/autodl-tmp/sft_grpo_setup/test-forward-kl.log 2>&1
bash -n run_controlled_comparison.sh
git diff --check
python -u - <<'PY'
from pathlib import Path
import json,sys
from scripts.research.job_control import start_session
setup=Path('/root/autodl-tmp/sft_grpo_setup')
inputs=json.loads((setup/'verified_inputs.json').read_text())
assert inputs['dataset_sha256']=='e4981f9b3f3b2840b71539a1970858f1448ea798f26b1e976031cbb1fb7780b5'
root=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/sft_grpo_campaign_20261110')
result=start_session(root,[sys.executable,'-u',str(setup/'run_campaign.py')],('sft','grpo'))
(setup/'launch_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
assert result['status']!='failed'
PY
