"""Read-only progress viewer; Ctrl+C never stops training."""
import argparse,json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs')
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def show():
 state=read(ROOT/'sft_grpo_campaign_20261110/campaign_state.json')
 print(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间'))
 labels={'formal_preparation':'核验并准备冻结输入','formal_sft_training':'SFT训练','formal_sft_baseline_verification':'SFT Base基线验收（SFT20已结束）','formal_sft_analysis':'SFT全部监测点分析（训练已结束）','formal_grpo_training':'GRPO训练及监测','formal_grpo_analysis':'GRPO分析核验','formal_verification':'权重/题目顺序/精度审计','final_report':'生成完整报告','completed':'全部完成','failed':'错误待处理'}
 print('流程阶段:',labels.get(state.get('stage'),state.get('stage','环境安装／输入核验')))
 if state.get('stage')=='completed':print('报告验证与打包:', '已完成' if state.get('report_browser_verified') else '进行中')
 if state.get('error'):print('错误:',state['error'])
 paths=[state['formal']] if state.get('formal') and (Path(state['formal'])/'config.json').is_file() else [state.get('acceptance',str(ROOT/'sft_grpo_acceptance_fast_20261110'))]
 for run_path in paths:
  run=Path(run_path);name=run.name
  if not run.exists():continue
  print(name)
  for method in ('sft','grpo'):
   path=run/method/'train/metrics.jsonl';latest={}
   if path.is_file():
    for line in path.read_text().splitlines():
     try:latest=json.loads(line)
     except json.JSONDecodeError:continue
   config=read(run/'config.json');total=config.get('training',{}).get('steps','?')
   complete=sorted(int(p.name.split('_')[1]) for p in (run/method/'analysis/steps').glob('step_*') if (p/'complete.json').is_file())
   print(f"  {method}: 已记录更新 {latest.get('step',0)}/{total}; 分析完成 {complete}; loss={latest.get('metrics',{}).get('actor/loss','—')}")
   generation=read(run/method/'analysis/generation_progress.json')
   if generation:
    print('    评测生成:',generation.get('optimizer_step'),str(generation.get('completed'))+'/'+str(generation.get('total')),generation.get('status'))
    if generation.get('status')=='completed' and generation.get('optimizer_step') not in complete:print('    回答生成已完成，正在提取隐状态、分布和几何指标；尚未到完整分析完成状态。')
 import subprocess
 gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
 if gpu.returncode==0:print('GPU 显存已用/总量，利用率:',gpu.stdout.strip())
 if state.get('report'):print('完整报告:',state['report'])
 return state.get('stage')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 try:
  while True:
   stage=show()
   if not a.watch or stage=='failed':break
   if stage=='completed' and read(ROOT/'sft_grpo_campaign_20261110/campaign_state.json').get('report_browser_verified'):break
   time.sleep(5);print('\n')
 except KeyboardInterrupt:pass
