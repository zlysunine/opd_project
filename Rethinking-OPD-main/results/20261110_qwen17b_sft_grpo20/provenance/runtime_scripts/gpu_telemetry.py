"""Record physical NVML occupancy; PyTorch virtual-pool peaks are not physical VRAM."""
from pathlib import Path
import json,subprocess,time
setup=Path('/root/autodl-tmp/sft_grpo_setup');state_path=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/sft_grpo_campaign_20261110/campaign_state.json')
start=time.time()
while time.time()-start<86400:
 try:
  state=json.loads(state_path.read_text())
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
  used,total,util,power=map(float,raw.split(','))
  with (setup/'gpu_physical_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),stage=state.get('stage'),used_mib=used,total_mib=total,utilization_percent=util,power_watts=power))+'\n')
  if state.get('stage')=='completed':break
 except Exception as e:
  with (setup/'gpu_telemetry_errors.log').open('a') as f:f.write(type(e).__name__+': '+str(e)+'\n')
 time.sleep(5)
