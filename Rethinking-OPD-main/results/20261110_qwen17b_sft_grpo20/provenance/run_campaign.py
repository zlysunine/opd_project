"""Server orchestration: two-method GPU acceptance, independent 20-step runs, reports."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/opd_project/Rethinking-OPD-main')
SETUP=Path('/root/autodl-tmp/sft_grpo_setup')
CAMPAIGN=ROOT/'research_runs/sft_grpo_campaign_20261110'
ACCEPT=ROOT/'research_runs/sft_grpo_acceptance_fast_20261110'
FORMAL=ROOT/'research_runs/sft_grpo20_throughput_20261110'
MODEL=Path('/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base')
NATIVE=Path('/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet')
PROBE=ROOT/'research-runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe'
CONFIG=ROOT/'configs/controlled_sft_grpo20_throughput_server.json'
METHODS=('sft','grpo')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stage(name,**extra):
    value=dict(stage=name,updated_unix=time.time(),acceptance=str(ACCEPT),formal=str(FORMAL),**extra)
    write(CAMPAIGN/'campaign_state.json',value)
    print('CAMPAIGN_STAGE',json.dumps(value,ensure_ascii=False),flush=True)


def run(command):
    print('RUN',json.dumps([str(x) for x in command],ensure_ascii=False),flush=True)
    subprocess.run([str(x) for x in command],cwd=ROOT,check=True)


def execute_comparison(workdir,steps):
    command=[sys.executable,'-u','-m','scripts.research.compare','run','--config',CONFIG,
      '--model-dir',MODEL,'--model-revision','ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
      '--input',NATIVE,'--dataset-revision','5cf055d1fe3d7a2eb19719ac020211469736ae44',
      '--reuse-probe',PROBE,'--workdir',workdir,'--method','sft_grpo','--steps',str(steps),
      '--monitor-steps','0,1,2' if steps==2 else '0,1,5,10,20','--execute','--foreground']
    run(command)


def verify(workdir,steps):
    from scripts.research.common import file_hash
    from scripts.research.retention import completed_analysis,_saved_checkpoint
    from analysis.common import load_config
    fingerprints=[];initials=[];result={}
    schedule=rows(workdir/'shared/probe/train_schedule.jsonl')
    assert len(schedule)==8*steps
    for method in METHODS:
        directory=workdir/method
        logs=rows(directory/'train/metrics.jsonl')
        assert [r['step'] for r in logs]==list(range(1,steps+1)),(method,'steps')
        for r in logs:
            for name in ('actor/loss','actor/grad_norm'):
                assert math.isfinite(r['metrics'][name]),(method,r['step'],name)
        fingerprint=json.loads((directory/'initial_adapter.json').read_text())
        assert fingerprint['zero_b_verified'];fingerprints.append(fingerprint)
        for step in range(1,steps+1):
            record=json.loads((directory/'train/exposures'/f'step_{step:06d}.json').read_text())
            assert record['problem_ids']==[r['problem_id'] for r in schedule[(step-1)*8:step*8]]
        config_path=directory/'analysis_config.json';analysis=load_config(config_path)
        for step in analysis['monitor']['optimizer_steps']:
            job=directory/'jobs'/f'step_{step:06d}.json'
            assert completed_analysis(job,config_path,analysis) is not None,(method,step)
            assert (directory/'adapter_archive'/f'global_step_{step}'/'archive_complete.json').is_file()
        checkpoint=directory/'train/checkpoints'/f'global_step_{steps}'
        _saved_checkpoint(checkpoint)
        held=[r for r in rows(directory/'analysis/steps/step_000000/live_rollouts.jsonl') if r['split']=='heldout_probe']
        initials.append({r['problem_id']:r['correct'] for r in held})
        final=[r for r in rows(directory/'analysis/steps'/f'step_{steps:06d}'/'live_rollouts.jsonl') if r['split']=='heldout_probe']
        result[method]=dict(steps=steps,metrics_rows=len(logs),baseline_correct=sum(initials[-1].values()),
                 final_correct=sum(r['correct'] for r in final),heldout_n=len(final),checkpoint=str(checkpoint),
                 schedule_sha256=file_hash(workdir/'shared/probe/question_schedule.json'))
    assert fingerprints[0]==fingerprints[1],'Initial LoRA mismatch'
    assert initials[0]==initials[1],'Base outcomes mismatch'
    assert not (workdir/'opsd/train/metrics.jsonl').exists(),'OPSD must not run'
    write(workdir/'verification.json',result)
    print('VERIFICATION',json.dumps(result),flush=True)
    return result


def roundtrips(workdir,steps):
    from analysis.checkpoints import export_checkpoint
    from scripts.research.retention import discard_export
    for method in METHODS:
        directory=workdir/method
        output=directory/'roundtrip';output.mkdir(exist_ok=True)
        command=[sys.executable,'-u','-m','scripts.research.audit_adapter_roundtrip',
                 '--base',MODEL,'--input',workdir/'shared/probe/fixed_pool.jsonl','--samples','4']
        run(command+['--adapter',directory/'adapter_archive/global_step_0','--require-zero','--output',output/'zero_update.json'])
        training=json.loads((directory/'training_config.json').read_text())
        merged=export_checkpoint(training,steps)
        run(command+['--adapter',directory/'adapter_archive'/f'global_step_{steps}',
            '--merged',merged,'--output',output/f'trained_step{steps}.json'])
        discard_export(training,steps)


def final_report():
    from analysis.html_report import build
    for method in METHODS:
        build(FORMAL/method,FORMAL/'report'/f'{method.upper()}_report.html')
    shutil.copy2(FORMAL/'report/OPSD_report.html',FORMAL/'report/SFT_GRPO_report.html')
    result=ROOT/'results/20261110_qwen17b_sft_grpo20'
    result.mkdir(parents=True,exist_ok=True)
    shutil.copytree(FORMAL/'report',result/'report',dirs_exist_ok=True)
    for filename in ('config.json','design_manifest.json','environment.json','verification.json'):
        shutil.copy2(FORMAL/filename,result/filename)
    for method in METHODS:
        shutil.copytree(FORMAL/method/'roundtrip',result/method/'roundtrip',dirs_exist_ok=True)
        shutil.copytree(FORMAL/method/'train/exposures',result/method/'exposures',dirs_exist_ok=True)
        shutil.copy2(FORMAL/method/'train/metrics.jsonl',result/method/'training_metrics.jsonl')
    note='''# SFT / GRPO 20-step 实验结果\n\n打开 `report/SFT_GRPO_report.html` 查看全部指标；`SFT_report.html` 与 `GRPO_report.html` 为单组报告。\n`report/CONTROLLED_COMPARISON.html` 展示两组同题配对统计与预算。\n\n本轮仅执行 SFT/GRPO，各从 Qwen3-1.7B-Base 独立训练20步，SFT通过先前2步CUDA与权重更新验收后正式训练；GRPO在本次正式运行中逐步验收。\n使用历史冻结题集，监测0/1/5/10/20；逐步训练题目顺序在两组间一致。\n保留所有监测点的分析结果和 LoRA adapter，仅保留最新完整恢复 checkpoint。\n自主评测使用vLLM连续批处理与CUDA Graph，最多64条并发，上限16384tokens；SFT微批64，分块输出层CE，先训练20步再评测全部监测点；GRPO按32768 tokens动态微批，rollout显存比例0.70。\nstep0基线用同一新生成路径重新测量。几何、分布及adapter精度审计保持HF eager。\n\n历史 OPSD 的完整结果位于仓库 `results/20261028_qwen17b_opsd20/`；本轮未重训OPSD。\n历史运行缺少逐步训练题目ID收据，不能声称与这两组构成训练暴露严格配对的三组实验。\n原始回答、RSM/anchor/曲率等数组仍保存在服务器正式run中，不随轻量报告省略科学测量。\n'''
    (result/'README.md').write_text(note)
    files={str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest() for p in result.rglob('*') if p.is_file()}
    write(result/'RESULT_MANIFEST.json',dict(selected_steps=20,methods=list(METHODS),files=files,raw_run=str(FORMAL)))
    return result


def main():
    CAMPAIGN.mkdir(parents=True,exist_ok=True)
    with (CAMPAIGN/'campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # SFT already passed two real CUDA updates and adapter-change checks.
            # Full diagnostics are deliberately not a prerequisite to SFT20.
            stage('formal_preparation')
            command=[sys.executable,'-u','-m','scripts.research.compare','configure','--config',CONFIG,
              '--model-dir',MODEL,'--model-revision','ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
              '--input',NATIVE,'--dataset-revision','5cf055d1fe3d7a2eb19719ac020211469736ae44',
              '--reuse-probe',PROBE,'--workdir',FORMAL,'--steps','20',
              '--monitor-steps','0,1,5,10,20']
            run(command)
            run([sys.executable,'-u','-m','scripts.research.compare','prepare','--workdir',FORMAL])
            from scripts.research.preflight import check_environment
            preflight=check_environment(1);write(FORMAL/'environment.json',preflight)
            assert preflight['ok'],preflight
            stage('formal_sft_training')
            run([sys.executable,'-u','-m','scripts.research.compare','train','--workdir',FORMAL,
                 '--method','sft','--execute','--foreground'])
            assert [r['step'] for r in rows(FORMAL/'sft/train/metrics.jsonl')]==list(range(1,21))
            stage('formal_sft_baseline_verification')
            run([sys.executable,'-u','-m','scripts.research.compare','analyze','--workdir',FORMAL,'--method','sft','--steps','0'])
            stage('formal_grpo_training')
            run([sys.executable,'-u','-m','scripts.research.compare','train','--workdir',FORMAL,
                 '--method','grpo','--execute','--foreground'])
            stage('formal_grpo_analysis')
            run([sys.executable,'-u','-m','scripts.research.compare','analyze','--workdir',FORMAL,'--method','grpo'])
            stage('formal_sft_analysis')
            run([sys.executable,'-u','-m','scripts.research.compare','analyze','--workdir',FORMAL,'--method','sft'])
            from scripts.research.retention import process_checkpoint
            process_checkpoint(FORMAL/'sft/analysis_config.json',FORMAL/'sft/train/checkpoints/global_step_20',20)
            stage('formal_verification')
            verify(FORMAL,20);roundtrips(FORMAL,20)
            stage('final_report')
            run([sys.executable,'-u','-m','scripts.research.compare','report','--workdir',FORMAL])
            report=final_report()
            stage('completed',report=str(report/'report/SFT_GRPO_report.html'))
        except Exception as exc:
            stage('failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':main()
