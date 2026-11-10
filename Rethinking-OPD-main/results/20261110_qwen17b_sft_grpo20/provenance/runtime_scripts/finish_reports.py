"""Wait for verified training, validate the actual offline reports, then package them."""
from pathlib import Path
import fcntl,hashlib,html,json,os,shutil,subprocess,sys,time
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
SETUP=Path('/root/autodl-tmp/sft_grpo_setup')
STATE=ROOT/'research_runs/sft_grpo_campaign_20261110/campaign_state.json'
RESULT=ROOT/'results/20261110_qwen17b_sft_grpo20'

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def process():
 while True:
  try:state=json.loads(STATE.read_text())
  except (FileNotFoundError,json.JSONDecodeError):state={}
  if state.get('stage')=='completed':break
  time.sleep(20)
 from analysis.html_report import read_run,write_html
 formal=ROOT/'research_runs/sft_grpo20_throughput_20261110'
 for method,name in [('sft','SFT_report.html'),('grpo','GRPO_report.html'),(None,'SFT_GRPO_report.html')]:
  payload,provenance,asset_root=read_run(formal/method if method else formal)
  note='GRPO的PyTorch峰值内存计数受vLLM睡眠/虚拟内存池重映射影响，可能超过物理显存；保留原始计数，不将其解释为实体显存占用。物理占用见gpu_physical_telemetry.jsonl（5秒采样，恢复后开始记录）。'
  payload['meta'].setdefault('notices',[]).append(note)
  for entry in payload['metrics']:
   if any(key in entry.get('name','') for key in ('max_memory_allocated','max_memory_reserved','peak_memory_max_rank')):entry.setdefault('notes',[]).append(note)
  write_html(payload,RESULT/'report'/name,asset_root,provenance)
 env=os.environ.copy();env['PYTHONPATH']='/root/autodl-tmp/report_tools:'+env.get('PYTHONPATH','')
 validations={}
 for name in ['SFT_GRPO_report.html','SFT_report.html','GRPO_report.html']:
  page=RESULT/'report'/name
  subprocess.run([sys.executable,str(SETUP/'check_report_browser.py'),str(page)],check=True,env=env)
  validations[name]=json.loads(page.with_suffix('.browser_validation.json').read_text())
 historical=RESULT/'history';historical.mkdir(exist_ok=True)
 shutil.copy2(ROOT/'results/20261028_qwen17b_opsd20/OPSD_report.html',historical/'OPSD_report.html')
 verification=json.loads((RESULT/'verification.json').read_text())
 summary=json.loads((RESULT/'report/controlled_summary.json').read_text())
 table=[]
 for method,r in verification.items():
  assert r['steps']==20 and r['metrics_rows']==20
  budget=summary['methods'][method]['budget']
  table.append(f"<tr><td>{method.upper()}</td><td>{r['steps']}</td><td>{r['baseline_correct']}/{r['heldout_n']}</td><td>{r['final_correct']}/{r['heldout_n']}</td><td>{int(budget['optimized_tokens']):,}</td><td>{int(budget['rollout_tokens']):,}</td></tr>")
 doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFT / GRPO 20步最终结果</title><style>body{max-width:1050px;margin:32px auto;padding:0 18px;font:16px/1.8 system-ui;color:#173249;background:#f6f8fb}a{color:#145e91}.links{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.links a{display:block;background:white;padding:18px;border:1px solid #d8e3eb;border-radius:8px;text-decoration:none}table{border-collapse:collapse;background:white;width:100%;margin:24px 0}th,td{padding:10px;border:1px solid #d8e3eb;text-align:left}</style><h1>SFT / GRPO · 20步最终结果</h1><p>两组均已完成独立20步训练、监测点分析及权重合并精度审计。训练已经结束。</p><div class="links"><a href="report/SFT_GRPO_report.html">完整全指标交互报告</a><a href="report/SFT_report.html">逐项查看 SFT</a><a href="report/GRPO_report.html">逐项查看 GRPO</a><a href="report/CONTROLLED_COMPARISON.html">两组配对统计与预算</a><a href="history/OPSD_report.html">历史 OPSD 报告（本轮未重训）</a></div><h2>实际测量结果</h2><table><tr><th>方法</th><th>更新数</th><th>初始正确题</th><th>最终正确题</th><th>优化tokens</th><th>rollout tokens</th></tr>'''+''.join(table)+'''</table><p>同一冻结heldout50题，监测0/1/5/10/20；查看配对区间和逐题变化时请打开配对统计报告。SFT监督序列是重复reference，其rollout tokens为0。</p><p>历史OPSD缺少逐步训练题目ID收据，仅作为独立历史参照，不称为训练暴露严格配对的三组实验。本轮评测使用vLLM连续批处理与CUDA Graph，最多64条并发，Base基线已按同一路径重新测量。几何和分布提取保持eager，历史OPSD的生成数值路径不同。</p><p>原始回答、anchor、RSM与曲率数组保存在服务器正式run；所有监测点LoRA均归档，只保留最新完整恢复checkpoint。GRPO的PyTorch内存峰值计数受vLLM睡眠与虚拟内存池重映射影响，可能超过物理显存，不能当作实际物理占用；实际占用以gpu_physical_telemetry.jsonl的nvidia-smi采样为准（采样从修复后恢复阶段开始）。SHA256见RESULT_MANIFEST.json。</p></html>'''
 (RESULT/'index.html').write_text(doc)
 (ROOT/'SFT_GRPO_FINAL_RESULTS_zh.md').write_text('# SFT / GRPO 20步最终结果\n\n统一入口：[最终报告](results/20261110_qwen17b_sft_grpo20/index.html)。\n\n训练及全部监测、精度审计和报告浏览器检查已完成。历史OPSD只作独立参照，本轮未重训。\n原始产物：`research_runs/sft_grpo20_throughput_20261110/`。\n完整压缩包：`/root/autodl-tmp/SFT_GRPO_20step_results.zip`。\n')
 for method in ('sft','grpo'):
  shutil.copy2(ROOT/'research_runs/sft_grpo20_throughput_20261110'/method/'initial_adapter.json',RESULT/method/'initial_adapter.json')
  shutil.copy2(ROOT/'research_runs/sft_grpo20_throughput_20261110'/method/'training_config.json',RESULT/method/'training_config.json')
  adoption=ROOT/'research_runs/sft_grpo20_throughput_20261110'/method/'baseline_generation_adoption.json'
  if adoption.exists():shutil.copy2(adoption,RESULT/method/adoption.name)
 for name in ['verified_inputs.json','preflight.json','final-pip-freeze.txt','server_source_changes.json','server_source_patch.diff','server_source_snapshot.json','evaluation_optimization_approval.json','fast_generation_gpu_audit.json','cuda_allocator_profile.json','throughput_training_evidence.json','sft_batch_benchmark.json','sft_chunk_benchmark.json','checkpoint_cuda_handoff_fix.json','checkpoint_cuda_handoff.patch','test-cache-handoff-retention.log','gpu_physical_telemetry.jsonl']:
  shutil.copy2(SETUP/name,RESULT/name)
 source=json.loads((SETUP/'server_source_snapshot.json').read_text())
 for relative,expected in source.items():
  original=ROOT/relative
  assert hashlib.sha256(original.read_bytes()).hexdigest()==expected,relative
  target=RESULT/'provenance/sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
 for name in ['vllm54_pilot/summary.json','test-throughput-controlled.log','test-throughput-retention.log','test-vllm-generation.log','grpo-throughput-preflight.log','run_campaign.py']:
  original=SETUP/name;target=RESULT/'provenance'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
 files={str(p.relative_to(RESULT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULT.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json'}
 write(RESULT/'RESULT_MANIFEST.json',dict(methods=['sft','grpo'],steps=20,browser_validation=validations,files=files))
 archive=Path('/root/autodl-tmp/SFT_GRPO_20step_results.zip')
 with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(RESULT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(RESULT.parent))
 with ZipFile(archive) as z:assert z.testzip() is None
 state.update(report=str(RESULT/'index.html'),download_archive=str(archive),report_browser_verified=True,updated_unix=time.time())
 write(STATE,state);write(SETUP/'report_delivery.json',state)
 print('ALL_REPORTS_VERIFIED_AND_PACKAGED',json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':
 with (SETUP/'report_postprocessing.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:process()
  except Exception as exc:
   write(SETUP/'report_delivery_error.json',dict(error=f'{type(exc).__name__}: {exc}',updated_unix=time.time()))
   raise
