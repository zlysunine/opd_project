"""Detached training sessions and read-only progress/log inspection (Linux)."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'completed', 'failed', 'stopped', 'lost'}


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identity(pid):
    """PID + boot + kernel start time distinguish a task from a reused PID."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def _alive(identity):
    return bool(identity and _identity(identity['pid']) == identity)


def _current(workdir):
    directory = Path(workdir).resolve()/'.session'
    pointer = _read(directory/'current.json')
    if pointer is None:
        return None
    attempt = pointer['attempt']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        raise ValueError('Invalid session attempt identifier')
    return directory/'attempts'/attempt


def record_progress(workdir, phase, method=None, step=None):
    """Only managed children write per-attempt stage state; foreground stays usable."""
    directory = os.environ.get('OPSD_SESSION_DIR')
    if not directory:
        return
    directory = Path(directory)
    request = _read(directory/'request.json')
    if request is None or Path(request['workdir']).resolve() != Path(workdir).resolve():
        return
    _json(directory/'progress.json', dict(phase=phase, method=method, optimizer_step=step,
                                         updated_utc=_now(), writer_pid=os.getpid()))


def start_session(workdir, command, methods=('opsd',)):
    """Transfer an exclusive lock to a terminal-independent supervisor."""
    workdir = Path(workdir).resolve()
    control = workdir/'.session'
    control.mkdir(parents=True, exist_ok=True)
    with (control/'lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This workdir already has an active task; use status/logs instead') from None
        previous = _current(workdir)
        state = _read(previous/'state.json', {}) if previous else {}
        if _alive(state.get('supervisor')) or _alive(state.get('child')):
            raise RuntimeError('A previous process for this workdir is still alive; refusing duplicate training')
        attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        directory = control/'attempts'/attempt
        directory.mkdir(parents=True)
        request = dict(workdir=str(workdir), command=list(command), methods=list(methods),
                       created_utc=_now(), created_unix=time.time(), python=sys.executable)
        _json(directory/'request.json', request)
        _json(directory/'state.json', dict(status='starting', supervisor=None, child=None,
                                          started_utc=_now(), started_unix=time.time(), returncode=None))
        _json(control/'current.json', dict(attempt=attempt))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'verl'),env.get('PYTHONPATH','')])
        runner = [sys.executable,'-u','-m','scripts.research.job_control','_worker',
                  '--directory',str(directory),'--lock-fd',str(lock.fileno())]
        try:
            with (directory/'console.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(runner, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lock.fileno(),))
        except Exception as error:
            _json(directory/'state.json', dict(status='failed', supervisor=None, child=None,
                  finished_utc=_now(), returncode=None, error=f'{type(error).__name__}: {error}'))
            raise
        # The inherited descriptor keeps the flock held after this launcher exits.
    # Reap the supervisor if the caller stays alive (e.g. an API client). This
    # daemon thread never keeps an SSH launcher alive or controls the task.
    threading.Thread(target=process.wait,daemon=True).start()
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        state = _read(directory/'state.json', {})
        if state.get('supervisor') or process.poll() is not None:
            break
        time.sleep(0.05)
    if process.poll() is not None and state.get('status') not in TERMINAL:
        state.update(status='failed', returncode=process.returncode, finished_utc=_now(),
                     error='Background supervisor exited during startup; inspect console.log')
        _json(directory/'state.json',state)
    return dict(status=state.get('status','starting'), workdir=str(workdir),
                supervisor_pid=process.pid, console_log=str(directory/'console.log'))


def _worker(directory, lock_fd):
    directory = Path(directory).resolve()
    os.fstat(lock_fd)
    os.set_inheritable(lock_fd, False)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    request = _read(directory/'request.json')
    state = _read(directory/'state.json')
    state.update(supervisor=_identity(os.getpid()), heartbeat_utc=_now())
    _json(directory/'state.json',state)
    child = None

    def signal_child(signum):
        try:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signum)
        except ProcessLookupError:
            # It can exit between poll() and killpg(). It is still ours to reap.
            pass

    try:
        env = os.environ.copy()
        env['OPSD_SESSION_DIR'] = str(directory)
        env['PYTHONUNBUFFERED'] = '1'
        child = subprocess.Popen(request['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        state.update(status='running', child=_identity(child.pid), heartbeat_utc=_now())
        _json(directory/'state.json',state)
        last_heartbeat = time.monotonic()
        stopping_since = None
        while child.poll() is None:
            if requested and stopping_since is None:
                stopping_since = time.monotonic()
                state['status'] = 'stopping'
                _json(directory/'state.json',state)
                signal_child(signal.SIGTERM)
            elif stopping_since is not None and time.monotonic()-stopping_since > 30:
                signal_child(signal.SIGKILL)
            if time.monotonic()-last_heartbeat >= 2:
                state['heartbeat_utc'] = _now()
                _json(directory/'state.json',state)
                last_heartbeat = time.monotonic()
            time.sleep(0.2)
        result = child.wait()
        state.update(status='stopped' if requested else ('completed' if result==0 else 'failed'),
                     returncode=result, finished_utc=_now(), heartbeat_utc=_now())
        _json(directory/'state.json',state)
        return result if result>=0 else 128-result
    except BaseException as error:
        if child is not None and child.poll() is None:
            signal_child(signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_child(signal.SIGKILL)
                child.wait()
        state.update(status='failed', finished_utc=_now(), heartbeat_utc=_now(),
                     returncode=child.returncode if child else None,
                     error=f'{type(error).__name__}: {error}')
        _json(directory/'state.json',state)
        raise
    finally:
        os.close(lock_fd)


def _last_metric(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0,os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0,size-262144))
            data = stream.read()
        # Concurrent writers may not have finished the final JSON line yet.
        lines = data.split(b'\n')[:-1]
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except (ValueError,UnicodeDecodeError):
                continue
            if isinstance(row,dict) and isinstance(row.get('step'),int):
                return row
    except FileNotFoundError:
        pass
    return None


def training_target(workdir, original):
    """A requested shorter budget does not rewrite frozen analysis identities."""
    limit = _read(Path(workdir)/'run_limits.json', {})
    target = limit.get('stop_at_step', original)
    if target is not None and (isinstance(target, bool) or not isinstance(target, int)
                               or target < 1 or (original is not None and target > original)):
        raise ValueError('Invalid requested training stop step')
    return target


def status(workdir):
    workdir = Path(workdir).resolve()
    directory = _current(workdir)
    state = _read(directory/'state.json', {}) if directory else {}
    request = _read(directory/'request.json', {}) if directory else {}
    progress = _read(directory/'progress.json', {}) if directory else {}
    lifecycle = state.get('status','not_started')
    if lifecycle not in TERMINAL and lifecycle != 'not_started' and not _alive(state.get('supervisor')):
        if _alive(state.get('child')):
            lifecycle = 'orphaned'
        elif lifecycle != 'starting' or time.time()-state.get('started_unix',0)>10:
            lifecycle = 'lost'
    config = _read(workdir/'config.json', {})
    methods = request.get('methods',['opsd'])
    reports = {}
    for method in methods:
        base = workdir/method
        training = _read(base/'training_config.json',config.get('training',{}))
        total = training_target(workdir, training.get('steps'))
        metrics_path = base/'train/metrics.jsonl'
        metric = _last_metric(metrics_path)
        previous_attempt = bool(metric and directory and metrics_path.stat().st_mtime < state.get('started_unix',0))
        tracker = base/'train/checkpoints/latest_checkpointed_iteration.txt'
        try:
            checkpoint_step = int(tracker.read_text().strip())
        except FileNotFoundError:
            checkpoint_step = None
        completed = checkpoint_step or 0
        if metric and not previous_attempt:
            completed = metric['step']
        jobs = {p.stem:_read(p) for p in (base/'jobs').glob('step_*.json')}
        results = {p.stem:_read(p) for p in (base/'job_status').glob('step_*.json')}
        analyzed = sorted(jobs[key]['optimizer_step'] for key,value in results.items()
                          if key in jobs and value.get('status')=='complete')
        failed = {jobs[key]['optimizer_step']:value.get('error') for key,value in results.items()
                  if key in jobs and value.get('status')=='failed'}
        expected = training.get('monitor',{}).get('optimizer_steps',config.get('monitor',{}).get('optimizer_steps',[]))
        if total is not None:
            expected = [step for step in expected if step <= total]
        reports[method] = dict(completed_steps=completed, total_steps=total,
                percent=100*completed/total if total else None, checkpoint_step=checkpoint_step,
                metrics_from_previous_attempt=previous_attempt,
                latest_metric_step=metric['step'] if metric else None,
                latest_metrics=metric.get('metrics',{}) if metric else {},
                budget=metric.get('budget',{}) if metric else {},
                analyzed_steps=analyzed, expected_analysis_steps=expected, failed_analysis=failed,
                metrics_updated_unix=metrics_path.stat().st_mtime if metric else None,
                train_log=str(base/'train/train.log'))
    from .preparation_progress import read_preparation
    preparation = read_preparation(workdir)
    generation = _read(workdir/'shared/generation_progress.json')
    current_phase = progress.get('phase','starting' if directory else 'not_started')
    if preparation and current_phase in ('starting','not_started','prepare'):
        current_phase = preparation['phase']
    return dict(status=lifecycle, workdir=str(workdir), phase=current_phase,
                preparation=preparation, generation=generation,
                active_method=progress.get('method'), analysis_step=progress.get('optimizer_step'),
                supervisor=state.get('supervisor'), child=state.get('child'),
                started_utc=state.get('started_utc'), heartbeat_utc=state.get('heartbeat_utc'),
                finished_utc=state.get('finished_utc'), returncode=state.get('returncode'),
                error=state.get('error'), methods=reports,
                console_log=str(directory/'console.log') if directory else None)


def render(report):
    phases = {'not_started':'尚未启动','starting':'正在启动','environment_check':'检查环境',
              'prepare':'准备数据和固定输入','training':'训练','analysis_verification':'校验分析结果',
              'exporting':'导出分析权重','analyzing':'分析 checkpoint','checkpoint_cleanup':'清理旧权重','report':'生成报告',
              'identity_check':'校验输入身份','model_loading':'加载模型','selection_cache':'校验已保存筛选结果',
              'native_read':'读取原生数据','native_adaptation':'难度筛选与参考文本分词',
              'eligibility_checks':'逐题核验参考答案和长度','selection':'逐题筛选',
              'split_freeze':'去重并冻结划分','selection_checkpoint':'保存筛选检查点',
              'baseline_generation':'生成基线回答','baseline_quality':'检查基线抽取成功率',
              'fixed_pool_generation':'生成固定输入池','publishing_probe':'保存固定输入池','complete':'数据准备完成'}
    states = {'not_started':'未启动','starting':'启动中','running':'运行中','stopping':'停止中',
              'completed':'已完成','failed':'失败','stopped':'已停止','lost':'进程已退出（无完成记录）',
              'orphaned':'监控进程退出，训练子进程仍存活'}
    lines = [f"任务状态：{states.get(report['status'],report['status'])}",
             f"当前阶段：{phases.get(report['phase'],report['phase'])}"]
    if report['started_utc']:
        started = datetime.fromisoformat(report['started_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        lines.append('启动时间：'+started.strftime('%Y-%m-%d %H:%M:%S %z'))
    if report['supervisor']:
        lines.append(f"后台 PID：{report['supervisor']['pid']}")
    if report['heartbeat_utc']:
        heartbeat = datetime.fromisoformat(report['heartbeat_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
        lines.append('后台心跳：'+heartbeat.strftime('%Y-%m-%d %H:%M:%S %z'))
    prep = report.get('preparation')
    if prep:
        processed, total = prep.get('processed'), prep.get('total')
        if total is not None:
            lines.append(f"  数据处理：{processed}/{total}；本阶段记录状态：{prep['status']}")
        if prep.get('accepted') is not None:
            lines.append(f"  筛选通过（去重前）：{prep['accepted']}；排除：{prep.get('excluded','未知')}")
        if prep.get('exclusion_counts'):
            lines.append('  排除原因：'+str(prep['exclusion_counts']))
        if prep.get('elapsed_seconds') is not None:
            lines.append(f"  准备累计耗时：{prep['elapsed_seconds']/60:.1f} 分钟")
        if prep.get('rows_per_second') is not None:
            lines.append(f"  本阶段处理速度：{prep['rows_per_second']:.2f} 题/秒")
        if prep.get('status')=='running' and prep.get('remaining_seconds') is not None:
            lines.append(f"  本阶段预计剩余：{prep['remaining_seconds']/60:.1f} 分钟（基于实测平均速度）")
        if prep.get('current_problem_id'):
            lines.append('  最近处理题目：'+str(prep['current_problem_id']))
        if prep.get('updated_unix'):
            updated = datetime.fromtimestamp(prep['updated_unix'],ZoneInfo('Asia/Shanghai'))
            lines.append('  数据进度更新时间：'+updated.strftime('%Y-%m-%d %H:%M:%S %z'))
        for item in prep.get('baseline_quality',{}).get('splits',[]):
            n, failures = item['n'], item['failures']
            rate = f"{(n-failures)/n:.1%}" if n else '未知'
            lines.append(f"  基线 {item['split']} 抽取成功：{n-failures}/{n}（{rate}）；"
                         f"门槛：{'通过' if item['passed'] else '失败'}")
        if prep.get('error'):
            lines.append('  数据准备错误：'+prep['error'])
    gen = report.get('generation')
    if gen:
        lines.append(f"  回答生成：{gen['pool']} {gen['completed']}/{gen['total']}；状态：{gen['status']}")
    for method, item in report['methods'].items():
        percent = f" ({item['percent']:.1f}%)" if item['percent'] is not None else ''
        lines.append(f"{method.upper()} 训练：{item['completed_steps']}/{item['total_steps'] or '?'}{percent}；"
                     f"最新 checkpoint：{item['checkpoint_step'] if item['checkpoint_step'] is not None else '无'}")
        if item['metrics_from_previous_attempt']:
            lines.append('  本次尚无新指标；训练步数暂按已保存 checkpoint 显示。')
        if item['metrics_updated_unix'] is not None:
            updated = datetime.fromtimestamp(item['metrics_updated_unix'],ZoneInfo('Asia/Shanghai'))
            lines.append('  指标更新时间：'+updated.strftime('%Y-%m-%d %H:%M:%S %z'))
        lines.append(f"  分析完成：{len(item['analyzed_steps'])}/{len(item['expected_analysis_steps'])}；"
                     f"已完成 step：{item['analyzed_steps']}")
        if report['active_method']==method and report['phase'] in ('exporting','analyzing'):
            lines.append(f"  当前分析 step：{report['analysis_step']}")
        values = item['latest_metrics']
        keys = ('actor/forward_kl','opd/overlap_ratio','opd/student_entropy','opd/teacher_entropy')
        present = [f'{key}={values[key]}' for key in keys if key in values]
        if present:
            lines.append('  最近记录指标：'+', '.join(present))
        if item['failed_analysis']:
            lines.append('  分析失败：'+str(item['failed_analysis']))
        lines.append('  训练日志：'+item['train_log'])
    if report['returncode'] is not None:
        lines.append('退出码：'+str(report['returncode']))
    if report['error']:
        lines.append('错误：'+report['error'])
    if report['console_log']:
        lines.append('完整控制台日志：'+report['console_log'])
    return '\n'.join(lines)


def show_logs(workdir, stream='console', method='opsd', lines=50, follow=False):
    report = status(workdir)
    filename = report['console_log'] if stream=='console' else str(Path(workdir)/method/'train/train.log')
    if filename is None:
        raise ValueError('No managed task has been started in this workdir')
    path = Path(filename)
    handle = None
    try:
        while True:
            if handle is None and path.is_file():
                handle = path.open(errors='replace')
                handle.seek(0,os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0,size-262144))
                text = handle.read()
                print(''.join(text.splitlines(keepends=True)[-lines:]),end='',flush=True)
            elif handle is not None:
                print(handle.read(),end='',flush=True)
            if not follow:
                if handle is None:
                    print(f'日志尚未创建：{path}')
                return
            if status(workdir)['status'] in TERMINAL:
                if handle is not None:
                    print(handle.read(),end='',flush=True)
                return
            time.sleep(1)
    finally:
        if handle is not None:
            handle.close()


def stop_session(workdir):
    directory = _current(workdir)
    state = _read(directory/'state.json',{}) if directory else {}
    if not _alive(state.get('supervisor')):
        raise ValueError('No verified live supervisor; no signal was sent')
    pid = state['supervisor']['pid']
    if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
        descriptor = os.pidfd_open(pid)
        try:
            if not _alive(state['supervisor']):
                raise ValueError('Supervisor identity changed; no signal was sent')
            signal.pidfd_send_signal(descriptor,signal.SIGTERM)
        finally:
            os.close(descriptor)
    else:
        os.kill(pid,signal.SIGTERM)
    return dict(status='stop_requested', supervisor_pid=state['supervisor']['pid'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    for name in ('status','logs','stop'):
        command = sub.add_parser(name)
        command.add_argument('--workdir',default='research_runs/base_geometry_50')
        if name=='status':
            command.add_argument('--json',action='store_true')
            command.add_argument('--watch',action='store_true')
            command.add_argument('--interval',type=float,default=5)
        elif name=='logs':
            command.add_argument('--stream',choices=['console','train'],default='console')
            command.add_argument('--method',choices=['opsd','sft','grpo'],default='opsd')
            command.add_argument('--lines',type=int,default=50)
            command.add_argument('--follow',action='store_true')
    worker = sub.add_parser('_worker')
    worker.add_argument('--directory',required=True)
    worker.add_argument('--lock-fd',required=True,type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action=='_worker':
            return _worker(args.directory,args.lock_fd)
        if args.action=='status':
            if args.interval<=0:
                raise ValueError('--interval must be positive')
            while True:
                report = status(args.workdir)
                if args.watch and sys.stdout.isatty():
                    print('\033[2J\033[H',end='')
                print(json.dumps(report,ensure_ascii=False,indent=2) if args.json else render(report),flush=True)
                if not args.watch or report['status'] in TERMINAL or report['status']=='not_started':
                    return int(args.watch and report['status']!='completed')
                time.sleep(args.interval)
        elif args.action=='logs':
            if args.lines<1:
                raise ValueError('--lines must be positive')
            show_logs(args.workdir,args.stream,args.method,args.lines,args.follow)
        else:
            print(json.dumps(stop_session(args.workdir)))
        return 0
    except KeyboardInterrupt:
        # This is only the viewing client, never the detached training process.
        return 0


if __name__=='__main__':
    raise SystemExit(main())
