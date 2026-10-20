"""Small Section 3.1 experiment; no model downloads or cloud billing API calls.

The supervisor uses only the standard library. GPU dependencies are imported in
workers so --dry-run and the comparison report also work on a CPU machine.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
GROUPS = ("nonthinking", "grpo")
DYNAMIC_METRICS = {
    "overlap_ratio": ("opd/overlap_ratio", "Overlap ratio"),
    "overlap_token_advantage": ("opd/overlap_token_advantage", "Overlap-token advantage (nats)"),
    "student_entropy": ("opd/student_entropy", "Student entropy (nats)"),
    "teacher_entropy": ("opd/teacher_entropy", "Teacher entropy (nats)"),
    "entropy_gap": ("opd/entropy_gap", "Mean absolute entropy gap (nats)"),
    "overlap_nonempty_fraction": ("opd/overlap_nonempty_fraction", "Nonempty-overlap fraction"),
    "valid_response_tokens": ("opd/valid_response_tokens", "Valid response tokens"),
}
DYNAMIC_START = "<!-- opd-dynamic-metrics:start -->"
DYNAMIC_END = "<!-- opd-dynamic-metrics:end -->"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Print plan/commands; no imports, GPU work or writes")
    p.add_argument("--prepare-only", action="store_true", help="CPU: validate local models and prepare data before rental")
    p.add_argument("--summarize-only", type=Path, help="CPU: regenerate dynamic-metric reports from an existing run's logs")
    p.add_argument("--output", type=Path)
    p.add_argument("--student", type=Path, default=ROOT / "model/Qwen3-1.7B-Base")
    p.add_argument("--teacher-nonthinking", type=Path, default=ROOT / "model/Qwen3-4B")
    p.add_argument("--teacher-grpo", type=Path, default=ROOT / "model/Qwen3-4B-Base-GRPO")
    p.add_argument("--gpus", type=int, default=int(os.environ.get("NGPUS", "2")))
    p.add_argument("--billable-gpus", type=int, default=None, help="All rented GPUs, even if CUDA_VISIBLE_DEVICES hides some")
    p.add_argument("--price", type=float, default=float(os.environ.get("GPU_PRICE_RMB", "5.59")), help="Actual RMB per GPU-hour")
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--already-spent", type=float, default=0.0, help="Rental fees incurred before starting this script")
    p.add_argument("--reserve", type=float, default=5.0, help="Unused allowance for cleanup, storage and shutting down instance")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--response-length", type=int, default=3072)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=64)
    p.add_argument("--eval-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worker", choices=("prepare", "train", "eval"), help=argparse.SUPPRESS)
    p.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--group", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    return p


def make_config(a):
    paid = a.billable_gpus if a.billable_gpus is not None else a.gpus
    positive = (a.price, a.budget, a.gpus, paid, a.steps, a.batch, a.rollouts,
                a.response_length, a.train_size, a.eval_size, a.eval_samples, a.save_every)
    if not all(math.isfinite(x) and x > 0 for x in positive):
        raise ValueError("Prices, counts and lengths must be finite and positive")
    if not all(math.isfinite(x) and x >= 0 for x in (a.reserve, a.already_spent)):
        raise ValueError("reserve/already-spent must be finite and nonnegative")
    if a.budget > 30:
        raise ValueError("This pilot deliberately refuses a budget above RMB 30")
    usable = a.budget - a.reserve - a.already_spent
    if usable <= 0 or paid < a.gpus:
        raise ValueError("No usable budget, or billable GPUs < used GPUs")
    if a.batch % a.gpus or (a.batch * a.rollouts) % a.gpus:
        raise ValueError("batch and expanded batch must be divisible by the GPU count")
    if a.train_size < a.batch * a.steps:
        raise ValueError("train-size must cover steps*batch without repeating questions")
    if a.steps % a.save_every:
        raise ValueError("steps must be divisible by save-every for matched checkpoint selection")
    if a.eval_size > 83:
        raise ValueError("This pilot uses the 83-question AMC23 set; eval-size must be <=83")
    out = (a.output or ROOT / "pilot_runs" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    return {
        "output": str(out), "student": str(a.student.resolve()),
        "teachers": {"nonthinking": str(a.teacher_nonthinking.resolve()), "grpo": str(a.teacher_grpo.resolve())},
        "gpus": a.gpus, "billable_gpus": paid, "price_rmb_per_gpu_hour": a.price,
        "budget_rmb": a.budget, "already_spent_rmb": a.already_spent, "reserve_rmb": a.reserve,
        "wall_limit_seconds": usable / (paid * a.price) * 3600,
        "steps": a.steps, "save_every": a.save_every, "batch": a.batch,
        "rollouts": a.rollouts, "response_length": a.response_length,
        "train_size": a.train_size, "eval_size": a.eval_size, "eval_samples": a.eval_samples, "seed": a.seed,
    }


def train_args(c, group):
    out = Path(c["output"])
    length = 1024 + c["response_length"]
    # Keep FP32 actor/Adam states, BF16 FSDP computation as in the original worker.
    # Limit packed forwards instead of changing the optimizer-state dtype.
    options = {
        "algorithm.adv_estimator": "token_reward_direct", "algorithm.use_kl_in_reward": False,
        "data.train_files": str(out / "train.parquet"), "data.val_files": str(out / "dev.parquet"),
        "data.train_batch_size": c["batch"], "data.shuffle": False, "data.seed": c["seed"],
        "data.max_prompt_length": 1024, "data.max_response_length": c["response_length"],
        "data.filter_overlong_prompts": True, "data.truncation": "error", "data.return_raw_chat": True,
        "+data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.model.path": c["student"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_activation_offload": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.ppo_mini_batch_size": c["batch"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": length,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model"],
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0, "actor_rollout_ref.rollout.n": c["rollouts"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.30,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.max_num_seqs": max(8, c["batch"] * c["rollouts"]),
        "actor_rollout_ref.rollout.max_model_len": length,
        "actor_rollout_ref.rollout.max_num_batched_tokens": length,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": length,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "+actor_rollout_ref.rollout.log_prob_top_k": 16,
        "+actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "+actor_rollout_ref.rollout.teacher_temperature": 1.0,
        "+actor_rollout_ref.rollout.seed": c["seed"],
        "reward_model.enable": True, "reward_model.enable_resource_pool": False,
        "reward_model.model.path": c["teachers"][group], "reward_model.model.input_tokenizer": None,
        "reward_model.model.use_remove_padding": True,
        "reward_model.model.fsdp_config.param_offload": False,
        "+reward_model.model.dtype": "fp32",
        "reward_model.micro_batch_size_per_gpu": 1,
        "reward_model.use_dynamic_bsz": True, "reward_model.forward_max_token_len_per_gpu": length,
        "custom_reward_function.path": str(ROOT / "verl/verl/utils/reward_score/ttrl_math/__init__.py"),
        "custom_reward_function.name": "reward_func",
        "trainer.logger": ["console"], "trainer.project_name": "thinking_consistency_pilot",
        "trainer.experiment_name": group, "trainer.n_gpus_per_node": c["gpus"], "trainer.nnodes": 1,
        "trainer.total_epochs": 1, "trainer.total_training_steps": c["steps"],
        "trainer.save_freq": c["save_every"], "trainer.test_freq": -1,
        "trainer.val_before_train": False, "trainer.is_plot": False, "trainer.resume_mode": "disable",
        "+trainer.log_opd_dynamic_metrics": True,
        "trainer.default_local_dir": str(out / group),
        "+ray_kwargs.ray_init.address": "local",
        # Ray UNIX socket paths must remain short even if the output path is long.
        "+ray_kwargs.ray_init._temp_dir": "/tmp/opd-" + hashlib.sha256(str(out).encode()).hexdigest()[:8] + "-" + group[0],
    }
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in options.items()]


def prepare(c):
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    out = Path(c["output"])
    tok = AutoTokenizer.from_pretrained(c["student"], local_files_only=True)
    if not tok.chat_template:
        raise ValueError("Student has no chat template. Use the author's matching tokenizer; do not invent a new template.")
    student_vocab = tok.get_vocab()
    metadata = []
    for model in [c["student"], *c["teachers"].values()]:
        path = Path(model)
        if not list(path.glob("*.safetensors")):
            raise ValueError(f"Missing local safetensors weights: {path}. Download BEFORE renting GPUs.")
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        other = AutoTokenizer.from_pretrained(path, local_files_only=True)
        vocab = other.get_vocab()
        if any(vocab.get(token) != index for token, index in student_vocab.items()):
            raise ValueError(f"Tokenizer token IDs do not match student: {model}")
        if cfg.vocab_size <= max(student_vocab.values()):
            raise ValueError(f"Teacher/student output vocabulary too small: {model}")
        metadata.append({"path": model, "model_type": cfg.model_type, "hidden_size": cfg.hidden_size})
    write_json(out / "model_metadata.json", metadata)
    source = ROOT / "datasets/dapo-math-17k.parquet"
    table = pq.read_table(source)
    candidates = list(range(len(table)))
    random.Random(c["seed"]).shuffle(candidates)
    selected = []
    for i in candidates:
        prompt = table["prompt"][i].as_py()
        ids = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) <= 1024:
            selected.append(i)
        if len(selected) == c["train_size"]:
            break
    if len(selected) != c["train_size"]:
        raise ValueError("Not enough training prompts within the prompt cap")
    pq.write_table(table.take(selected), out / "train.parquet")
    dev = pq.read_table(ROOT / "datasets/test_data/AMC23/test.parquet")
    dev_ids = sorted(random.Random(c["seed"] + 1).sample(range(len(dev)), c["eval_size"]))
    dev = dev.take(dev_ids)
    for row in dev.to_pylist():
        ids = tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) > 1024:
            raise ValueError("An evaluation prompt exceeds 1024 tokens; use another fixed dev subset")
    pq.write_table(dev, out / "dev.parquet")
    write_json(out / "data_manifest.json", {"source": str(source), "train_indices": selected,
               "dev_source": "AMC23", "dev_indices": dev_ids, "seed": c["seed"],
               "chat_template": tok.chat_template, "enable_thinking": False})
    print(f"Prepared {len(selected)} training prompts, {len(dev)} AMC23 questions", flush=True)


def preflight_gpu(c):
    import torch
    import verl

    for package in ("vllm", "ray", "hydra", "omegaconf", "psutil", "flash_attn", "math_verify", "pylatexenc"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"Missing dependency {package}; install the project's environment BEFORE renting GPUs")
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
        for group in GROUPS:
            compose(config_name="ppo_trainer", overrides=train_args(c, group))
    if not Path(verl.__file__).resolve().is_relative_to(ROOT / "verl"):
        raise RuntimeError("Imported verl is not the local customized version")
    if torch.cuda.device_count() != c["gpus"]:
        raise RuntimeError("Visible CUDA GPU count must match --gpus; set CUDA_VISIBLE_DEVICES explicitly")
    if any(torch.cuda.get_device_properties(i).total_memory < 70 * 1024**3 for i in range(c["gpus"])):
        raise RuntimeError("Pilot targets >=80GB-class GPUs. Smaller cards need a separately validated configuration")


def train(c, group):
    import runpy
    import ray
    from transformers import set_seed

    set_seed(c["seed"])
    # Allow SIGTERM from the budget supervisor to shut down this private Ray cluster.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(124))
    sys.argv = ["verl.trainer.main_ppo", *train_args(c, group)]
    try:
        runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
    finally:
        ray.shutdown()


def evaluate(c, model, label):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from scripts.val.eval.utils import grade_answer_verl

    out = Path(c["output"])
    rows = pq.read_table(out / "dev.parquet").to_pylist()
    tok = AutoTokenizer.from_pretrained(c["student"], local_files_only=True)
    ids = [tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False) for row in rows]
    llm = LLM(model=model, tokenizer=c["student"], tensor_parallel_size=c["gpus"],
              dtype="bfloat16", max_model_len=1024 + c["response_length"],
              max_num_batched_tokens=1024 + c["response_length"], max_num_seqs=32,
              gpu_memory_utilization=0.65, enforce_eager=True, seed=c["seed"])
    sampling = SamplingParams(n=c["eval_samples"], temperature=0.7, top_p=0.95,
                              max_tokens=c["response_length"], seed=c["seed"],
                              stop_token_ids=[tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
                                              if t in tok.get_vocab()])
    per_question, lengths, clipped = [], [], []
    grading_timeouts = 0

    def timeout_handler(*_):
        raise TimeoutError("answer checker timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    # Small batches produce partial JSONL evidence even if the total budget expires.
    with (out / f"eval_{label}.jsonl").open("w") as f:
        for start in range(0, len(rows), 8):
            outputs = llm.generate([{"prompt_token_ids": x} for x in ids[start:start + 8]], sampling, use_tqdm=False)
            for offset, result in enumerate(outputs):
                i = start + offset
                scores = []
                for completion in result.outputs:
                    signal.setitimer(signal.ITIMER_REAL, 2.0)
                    try:
                        correct = bool(grade_answer_verl(completion.text, str(rows[i]["reward_model"]["ground_truth"])))
                    except TimeoutError:
                        grading_timeouts += 1
                        correct = False
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    scores.append(int(correct))
                    lengths.append(len(completion.token_ids))
                    clipped.append(completion.finish_reason == "length")
                    f.write(json.dumps({"question": i, "correct": correct, "finish_reason": completion.finish_reason,
                                        "tokens": len(completion.token_ids), "response": completion.text}, ensure_ascii=False) + "\n")
                    f.flush()
                per_question.append(sum(scores) / len(scores))
    write_json(out / f"eval_{label}.json", {"label": label, "questions": len(per_question),
               "samples_per_question": c["eval_samples"], "avg_accuracy": sum(per_question) / len(per_question),
               "per_question": per_question, "mean_tokens": sum(lengths) / len(lengths),
               "truncation_rate": sum(clipped) / len(clipped), "grading_timeouts": grading_timeouts})


class BudgetRunner:
    def __init__(self, c):
        self.c = c
        self.start = time.monotonic()
        self.deadline = self.start + c["wall_limit_seconds"]
        self.events = []

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def run(self, name, command, seconds):
        """Kill only this subprocess tree, including Ray/vLLM detached children."""
        import psutil

        allowance = min(seconds, self.remaining() - 15)
        if allowance < 1:
            raise TimeoutError("No budget remains for " + name)
        begun = time.monotonic()
        print(f"[{name}] limit={allowance / 60:.1f} min; total remaining={self.remaining() / 60:.1f} min", flush=True)
        env = os.environ.copy()
        env.pop("RAY_ADDRESS", None)
        owned = {}
        with (Path(self.c["output"]) / (name + ".log")).open("w") as log:
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            parent = psutil.Process(p.pid)
            limit = begun + allowance
            timed_out = False
            try:
                while p.poll() is None:
                    try:
                        for proc in [parent, *parent.children(recursive=True)]:
                            owned[proc.pid] = proc
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic() >= limit:
                        timed_out = True
                        break
                    time.sleep(min(0.5, max(0.01, limit - time.monotonic())))
            finally:
                # Even normal Ray driver exit can leave worker processes behind.
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=min(5, self.remaining()))
                    except subprocess.TimeoutExpired:
                        pass
                descendants = [proc for pid, proc in owned.items() if pid != p.pid]
                for proc in descendants:
                    try:
                        proc.terminate()
                    except psutil.Error:
                        pass
                _, alive = psutil.wait_procs(descendants, timeout=min(2, self.remaining()))
                for proc in alive:
                    try:
                        proc.kill()
                    except psutil.Error:
                        pass
                if p.poll() is None:
                    p.kill()
                p.wait()
                # POSIX children that are still in the original process group
                # are also ours, including ones created between polling ticks.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        code = 124 if timed_out else p.returncode
        event = {"name": name, "returncode": code, "seconds": time.monotonic() - begun}
        self.events.append(event)
        write_json(Path(self.c["output"]) / "stages.json", self.events)
        print(f"[{name}] exit={code}; see {name}.log", flush=True)
        return code


def checkpoints(directory, world_size):
    complete = {}
    tracker = Path(directory) / "latest_checkpointed_iteration.txt"
    try:
        last_saved = int(tracker.read_text().strip())
    except (OSError, ValueError):
        return complete
    for actor in Path(directory).glob("global_step_*/actor"):
        step = int(actor.parent.name.removeprefix("global_step_"))
        if step <= last_saved and (actor / "huggingface/config.json").is_file() and all(
            (actor / f"model_world_size_{world_size}_rank_{rank}.pt").is_file() for rank in range(world_size)
        ):
            complete[step] = actor
    return complete


def dynamic_points(path):
    """Parse paper diagnostics; missing historical metrics remain None."""
    points = {}
    for line in Path(path).read_text(errors="replace").splitlines() if Path(path).exists() else []:
        if "training/global_step:" not in line:
            continue
        step = re.search(r"training/global_step:\s*(\d+)", line)
        if not step:
            continue
        row = {"step": int(step[1])}
        for name, (key, _) in DYNAMIC_METRICS.items():
            match = re.search(re.escape(key) + r":\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", line)
            value = float(match[1]) if match else None
            row[name] = value if value is not None and math.isfinite(value) else None
        # Legacy overlap is the same quantity. Legacy adv_intersection uses a
        # different normalization, so it must NOT be substituted for paper Adv.
        if row["overlap_ratio"] is None:
            match = re.search(r"val-topk/overlap_ratio:\s*([-+\d.eE]+)", line)
            if match:
                value = float(match[1])
                if math.isfinite(value):
                    row["overlap_ratio"] = value
        if any(row[name] is not None for name in DYNAMIC_METRICS):
            points[row["step"]] = row
    return [points[step] for step in sorted(points)]


def overlap_points(path):
    return [{"step": row["step"], "overlap": row["overlap_ratio"]} for row in dynamic_points(path)
            if row["overlap_ratio"] is not None]


def export_dynamic_metrics(out, matched_step):
    out = Path(out)
    curves = {group: dynamic_points(out / f"train_{group}.log") for group in GROUPS}
    snapshots = {}
    for group, rows in curves.items():
        write_json(out / f"dynamic_metrics_{group}.json", rows)
        snapshots[group] = {
            "step1": next((row for row in rows if row["step"] == 1), None),
            "matched": next((row for row in rows if row["step"] == matched_step), None),
            "last_logged": rows[-1] if rows else None,
        }
    write_json(out / "dynamic_metrics.json", {
        "matched_step": matched_step, "log_base": "e", "entropy_support": "full vocabulary",
        "aggregation": "valid response position mean; advantage excludes empty intersections",
        "measurement_time": "before the update at training/global_step",
        "curves": curves, "snapshots": snapshots,
    })
    with (out / "dynamic_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["teacher_group", "step", *DYNAMIC_METRICS])
        writer.writeheader()
        for group, rows in curves.items():
            writer.writerows({"teacher_group": group, **row} for row in rows)
    lines = [DYNAMIC_START, "论文 §2.3 动态指标（均在更新前、各组自己的学生轨迹上统计）：", "",
             f"| 指标 | Non-thinking step 1 | Non-thinking step {matched_step or '—'} | GRPO step 1 | GRPO step {matched_step or '—'} |",
             "|---|---:|---:|---:|---:|"]
    for name, (_, label) in DYNAMIC_METRICS.items():
        if name == "valid_response_tokens":
            continue
        values = []
        for group in GROUPS:
            for phase in ("step1", "matched"):
                row = snapshots[group][phase]
                value = row.get(name) if row else None
                values.append("缺失" if value is None else f"{value:.6g}")
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines += ["", "Overlap-token advantage 在交集内分别归一化学生/教师概率，并先按每个位置的交集大小取平均。",
              "Entropy 是全词表熵；gap 是逐位置绝对差的平均 E[|Hq−Hp|]，单位为 nats。",
              "空交集不参与 advantage 的平均，Nonempty-overlap fraction 用于报告其覆盖率；缺失值不补零。",
              "完整曲线与原始数值见 dynamic_metrics.svg、dynamic_metrics.csv 和 dynamic_metrics.json。"]
    if any(row[name] is None for rows in curves.values() for row in rows
           for name in list(DYNAMIC_METRICS)[:5]):
        lines.append("部分指标缺失：旧日志不能反推出交集归一化 advantage 或逐位置熵差；需使用新版采集重新训练。")
    lines.append(DYNAMIC_END)
    draw_dynamic_metrics(out, curves, matched_step)
    return snapshots, "\n".join(lines)


def summarize_existing(directory):
    """Rebuild diagnostics on a CPU, without changing billing or verdicts."""
    out = Path(directory).resolve()
    if not out.is_dir():
        raise ValueError("Run directory does not exist: " + str(out))
    path = out / "summary.json"
    result = read_json(path) if path.exists() else {}
    selected = out / "selected_checkpoint.json"
    matched = result.get("matched_step")
    if matched is None and selected.exists():
        matched = read_json(selected)["step"]
    snapshots, section = export_dynamic_metrics(out, matched)
    draw_overlap(out, matched)
    if path.exists():
        result["dynamic_metrics"] = snapshots
        write_json(path, result)
    markdown = out / "summary.md"
    text = markdown.read_text() if markdown.exists() else "# 小批量实验动态指标\n"
    if DYNAMIC_START in text:
        text = re.sub(re.escape(DYNAMIC_START) + r".*?" + re.escape(DYNAMIC_END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    markdown.write_text(text)
    print(f"Dynamic metrics refreshed: {out / 'dynamic_metrics.svg'}", flush=True)


def report(c, runner, matched_step, error=None):
    out = Path(c["output"])
    elapsed = time.monotonic() - runner.start
    result = {"matched_step": matched_step, "error": error, "elapsed_seconds": elapsed,
              "estimated_compute_rmb": elapsed / 3600 * c["billable_gpus"] * c["price_rmb_per_gpu_hour"],
              "already_spent_rmb": c["already_spent_rmb"], "evaluations": {}, "overlap": {}, "stages": runner.events}
    for label in ("initial", *GROUPS):
        path = out / f"eval_{label}.json"
        if path.exists():
            result["evaluations"][label] = read_json(path)
    lines = ["# §3.1 小预算验证", "", f"共同 checkpoint：{matched_step or '无'} step。",
             f"脚本计时内估算 GPU 费用：¥{result['estimated_compute_rmb']:.2f}；启动前已花 ¥{c['already_spent_rmb']:.2f}。",
             "这不是云平台账单；实例停止计算后仍需手动关机。", "",
             f"| 模型 | AMC23 avg@{c['eval_samples']} | 截断比例 | 平均回答 token |", "|---|---:|---:|---:|"]
    for label, value in result["evaluations"].items():
        lines.append(f"| {label} | {value['avg_accuracy']:.2%} | {value['truncation_rate']:.1%} | {value['mean_tokens']:.0f} |")
    for group in GROUPS:
        points = overlap_points(out / f"train_{group}.log")
        write_json(out / f"overlap_{group}.json", points)
        initial = next((p["overlap"] for p in points if p["step"] == 1), None)
        matched = next((p["overlap"] for p in points if p["step"] == matched_step), None)
        result["overlap"][group] = {"step1": initial, "matched": matched}
    snapshots, section = export_dynamic_metrics(out, matched_step)
    result["dynamic_metrics"] = snapshots
    lines += ["", section]
    evals = result["evaluations"]
    if matched_step and all(label in evals for label in ("initial", *GROUPS)):
        a, b = evals["nonthinking"]["per_question"], evals["grpo"]["per_question"]
        differences = [y - x for x, y in zip(a, b)]
        rng = random.Random(c["seed"])
        samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
        gap = sum(differences) / len(differences)
        result["grpo_minus_nonthinking"] = {"gap": gap, "paired_bootstrap_95": [samples[50], samples[1949]]}
        lines += ["", f"GRPO 教师组 − Non-thinking 教师组：{gap * 100:+.2f} 个百分点；",
                  f"按题配对 bootstrap 95% 区间 [{samples[50] * 100:+.2f}, {samples[1949] * 100:+.2f}] 个百分点。",
                  "该区间只反映这批题的抽样波动，不包含训练 seed 不确定性。"]
        clipped = max(v["truncation_rate"] for v in evals.values())
        timeouts = sum(v["grading_timeouts"] for v in evals.values())
        oi, og = result["overlap"]["nonthinking"]["step1"], result["overlap"]["grpo"]["step1"]
        if error or clipped > 0.2 or timeouts:
            verdict = "结果不足：存在未完成阶段、较多截断或评分超时，应检查日志和长度设置。"
        elif gap > 0 and samples[50] > 0 and oi is not None and og is not None and og > oi:
            verdict = "出现与 §3.1 一致的早期信号；尚需增加训练量和 seed 复核。"
        else:
            verdict = "未得到清晰的联合证据；短训练结果不足以支持或否定 §3.1。"
    else:
        verdict = "结果不足：缺少共同 checkpoint 或完整评估。日志中的 overlap 仍可用于诊断。"
    result["verdict"] = verdict
    lines += ["", verdict, "", "step 1 overlap 来自首次更新前的学生轨迹；后续 overlap 来自各自演化后的轨迹。",
              "这是缩减实验：单 seed、短训练、短回答、AMC23 子集，不等同于原论文复现。"]
    if error:
        lines += ["", "运行状态：" + error]
    write_json(out / "summary.json", result)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    draw_overlap(out, matched_step)
    print("\n".join(lines), flush=True)


def draw_overlap(out, matched_step):
    """Portable chart without plotting dependencies or GPU work."""
    curves = {g: overlap_points(out / f"train_{g}.log") for g in GROUPS}
    end = matched_step or max([p["step"] for points in curves.values() for p in points] or [1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420">',
           '<rect width="800" height="420" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#222">',
           '<text x="70" y="28">Student/teacher top-16 overlap (before each update)</text>']
    for tick in range(6):
        y = 350 - 60 * tick
        svg += [f'<path d="M70 {y}H760" stroke="#ddd"/>', f'<text x="30" y="{y+5}">{tick/5:.1f}</text>']
    for group, color, label_y in [("nonthinking", "#d55e00", 385), ("grpo", "#0072b2", 408)]:
        points = [p for p in curves[group] if p["step"] <= end]
        coords = " ".join(f"{70 + 690 * p['step'] / end:.2f},{350 - 300 * p['overlap']:.2f}" for p in points)
        svg += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="70" y="{label_y}" fill="{color}">{group}</text>']
    svg += [f'<text x="600" y="385">Training step: 0 to {end}</text>', '</g></svg>']
    (out / "overlap.svg").write_text("\n".join(svg) + "\n")


def draw_dynamic_metrics(out, curves, matched_step):
    """Prefer Matplotlib for research plots; keep a dependency-free SVG fallback."""
    if draw_dynamic_metrics_matplotlib(out, curves, matched_step):
        return
    (out / "dynamic_metrics.png").unlink(missing_ok=True)
    end = matched_step or max([p["step"] for rows in curves.values() for p in rows] or [1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="940" viewBox="0 0 1120 940">',
           '<rect width="1120" height="940" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#222">',
           '<text x="50" y="25" font-size="18">OPD dynamic metrics (before each update)</text>',
           '<text x="50" y="47" fill="#d55e00">Non-thinking teacher</text>',
           '<text x="270" y="47" fill="#0072b2">GRPO teacher</text>']
    for panel, (name, (_, label)) in enumerate(list(DYNAMIC_METRICS.items())[:5]):
        x0, y0 = 75 + (panel % 2) * 550, 95 + (panel // 2) * 280
        width, height = 440, 190
        values = [row[name] for rows in curves.values() for row in rows
                  if row["step"] <= end and row[name] is not None]
        if name == "overlap_ratio":
            low, high = 0.0, 1.0
        elif name == "overlap_token_advantage":
            low, high = min([0.0, *values]), max([0.0, *values])
            span = high - low or 0.01
            low, high = low - span * 0.1, high + span * 0.1
        else:
            low, high = 0.0, max([0.0, *values]) * 1.1 or 1.0
        svg.append(f'<text x="{x0}" y="{y0 - 17}">{label}</text>')
        for tick in range(5):
            y = y0 + height - height * tick / 4
            value = low + (high - low) * tick / 4
            x = x0 + width * tick / 4
            svg += [f'<path d="M{x0} {y:.2f}h{width}" stroke="#ddd"/>',
                    f'<text x="{x0-58}" y="{y+4:.2f}">{value:.3g}</text>',
                    f'<text x="{x:.2f}" y="{y0+height+20}">{end*tick/4:.3g}</text>']
        for group, color in [("nonthinking", "#d55e00"), ("grpo", "#0072b2")]:
            segments, segment = [], []
            for row in curves[group]:
                if row["step"] > end:
                    continue
                value = row[name]
                if value is None:
                    if segment:
                        segments.append(segment)
                        segment = []
                    continue
                x = x0 + width * row["step"] / end
                y = y0 + height * (high - value) / (high - low)
                segment.append(f"{x:.2f},{y:.2f}")
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>')
            if segment:
                segments.append(segment)
            for segment in segments:
                svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
        if not values:
            svg.append(f'<text x="{x0+70}" y="{y0+100}" fill="#777">Not logged; do not interpret as zero</text>')
        svg.append(f'<text x="{x0+width-90}" y="{y0+height+40}">Training step</text>')
    svg += ['<text x="625" y="720">Entropies use the full vocabulary.</text>',
            '<text x="625" y="745">Gap = mean of per-position absolute gaps.</text>',
            '<text x="625" y="770">Advantage uses renormalized overlap distributions.</text>',
            '<text x="625" y="795">See JSON/CSV for all logged steps and coverage.</text>', '</g></svg>']
    (out / "dynamic_metrics.svg").write_text("\n".join(svg) + "\n")


def draw_dynamic_metrics_matplotlib(out, curves, matched_step):
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    end = matched_step or max([p["step"] for rows in curves.values() for p in rows] or [1])
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    try:
        for ax, (name, (_, label)) in zip(axes.flat, list(DYNAMIC_METRICS.items())[:5]):
            any_values = False
            for group, color, legend in [("nonthinking", "#d55e00", "Non-thinking teacher"),
                                          ("grpo", "#0072b2", "GRPO teacher")]:
                rows = [row for row in curves[group] if row["step"] <= end]
                x = [row["step"] for row in rows]
                y = [row[name] if row[name] is not None else math.nan for row in rows]
                any_values |= any(math.isfinite(value) for value in y)
                ax.plot(x, y, color=color, marker="o", markersize=3, linewidth=1.5, label=legend)
            ax.set(title=label, xlabel="Training step (before update)", xlim=(0, end))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
            if name == "overlap_ratio":
                ax.set_ylim(0, 1)
            elif name == "overlap_token_advantage":
                ax.axhline(0, color="#777", linewidth=0.7)
            elif name in ("student_entropy", "teacher_entropy", "entropy_gap"):
                ax.set_ylim(bottom=0)
            if not any_values:
                ax.text(0.5, 0.5, "Not logged; do not interpret as zero", ha="center", transform=ax.transAxes)
        axes.flat[5].axis("off")
        axes.flat[5].text(0, 1, "Full-vocabulary entropy, natural logarithms.\n\n"
                         "Gap: mean of per-position absolute gaps.\n\n"
                         "Advantage: overlap-normalized distributions;\nempty intersections excluded.\n\n"
                         "Missing samples are left blank.\nFull recorded history is in JSON/CSV.",
                         va="top", transform=axes.flat[5].transAxes)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2)
        fig.suptitle("OPD dynamic metrics on each group's student rollouts", x=0.04, ha="left", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "dynamic_metrics.svg")
        fig.savefig(out / "dynamic_metrics.png", dpi=160)
    finally:
        plt.close(fig)
    return True


def supervise(c):
    out = Path(c["output"])
    if (out / "stages.json").exists() or any((out / group).exists() for group in GROUPS):
        raise ValueError("Output already contains a run; use a fresh --output to avoid mixing results")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", c)
    runner = BudgetRunner(c)
    matched = None
    error = None
    worker = [sys.executable, str(SELF), "--config", str(out / "config.json"), "--worker"]
    # The TOTAL deadline includes prepare/imports, GPU checks, training, merge and evaluation.
    # Stage caps reserve time for both teachers and independent final evaluation.
    limit = c["wall_limit_seconds"]
    def total_timeout(*_):
        raise TimeoutError("Total estimated compute budget reached")

    previous_alarm = signal.signal(signal.SIGALRM, total_timeout)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        code = runner.run("prepare", worker + ["prepare"], min(300, limit * 0.08))
        if code:
            raise RuntimeError("Local model/data preflight failed; see prepare.log")
        code = runner.run("eval_initial", worker + ["eval", "--model", c["student"], "--group", "initial"], limit * 0.12)
        if code:
            raise RuntimeError("Initial evaluation failed/timed out; fix this before spending on training")
        for group in GROUPS:
            code = runner.run("train_" + group, worker + ["train", "--group", group], limit * 0.28)
            if code not in (0, 124):
                raise RuntimeError(f"{group} training failed (not evidence against the hypothesis); see train_{group}.log")
        available = {g: checkpoints(out / g, c["gpus"]) for g in GROUPS}
        common = set(available[GROUPS[0]]) & set(available[GROUPS[1]])
        if not common:
            raise RuntimeError("No common saved step; inspect progress before changing the pilot scale")
        matched = max(common)
        write_json(out / "selected_checkpoint.json", {"step": matched, "selection": "largest common saved step; independent of scores"})
        for group in GROUPS:
            merged = out / f"merged_{group}"
            command = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                       "--local_dir", str(available[group][matched]), "--target_dir", str(merged)]
            if runner.run("merge_" + group, command, min(180, runner.remaining() / 4)):
                raise RuntimeError("Checkpoint merge failed/timed out: " + group)
            # Reserve half the remaining time for the other group's merge/eval.
            seconds = (runner.remaining() - 30) / 2 if group == GROUPS[0] else runner.remaining() - 30
            if runner.run("eval_" + group, worker + ["eval", "--group", group, "--model", str(merged)], seconds):
                raise RuntimeError("Final evaluation failed/timed out: " + group)
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc) or "Interrupted"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        report(c, runner, matched, error)
        print(f"\nReport: {out / 'summary.md'}\n请现在关闭租用实例，平台仍会按开机时间收费。", flush=True)
    return 1 if error else 0


def main():
    a = parser().parse_args()
    if a.worker:
        c = read_json(a.config)
        if a.worker == "prepare":
            prepare(c)
            preflight_gpu(c)
        elif a.worker == "train":
            train(c, a.group)
        else:
            evaluate(c, a.model, a.group)
        return 0
    if a.summarize_only:
        if a.dry_run:
            print(f"Would regenerate dynamic-metric reports in {a.summarize_only}")
        else:
            summarize_existing(a.summarize_only)
        return 0
    c = make_config(a)
    print(f"Budget ¥{c['budget_rmb']:.2f}, reserve ¥{c['reserve_rmb']:.2f}, already spent ¥{c['already_spent_rmb']:.2f}; "
          f"{c['billable_gpus']} billed GPUs × ¥{c['price_rmb_per_gpu_hour']}/h; "
          f"script deadline {c['wall_limit_seconds'] / 60:.1f} min", flush=True)
    if a.dry_run:
        print(json.dumps(c, ensure_ascii=False, indent=2))
        for group in GROUPS:
            print("\n" + group + " Hydra overrides:\n" + "\n".join(train_args(c, group)))
        return 0
    if a.prepare_only:
        Path(c["output"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(c["output"]) / "config.json", c)
        prepare(c)
        return 0
    return supervise(c)


if __name__ == "__main__":
    raise SystemExit(main())
