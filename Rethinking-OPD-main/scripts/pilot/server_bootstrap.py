"""One-command Linux server setup and budget-limited pilot execution.

Only the standard library is needed to start. No cloud credentials or shutdown
commands are used. The rental timer begins before environment installation.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import time

import thinking_consistency as pilot


ROOT = pilot.ROOT
SELF = Path(__file__).resolve()
SOURCES = {
    "student": "Qwen/Qwen3-1.7B-Base",
    "nonthinking": "Qwen/Qwen3-4B",
    "grpo": "Thinking-Space/Qwen3-4B-Base-GRPO",
}
FLASH_HASHES = {
    "FALSE": "15db5bb6524dcbf292c3c116aa4c2fa823b80abff0eb5bc58454107bca1ba0c2",
    "TRUE": "55331d797171973c8babf2b5e5fce5f78859b1cd298d54a136e8994147fe9e95",
}


def arguments():
    p = pilot.parser()
    p.description = __doc__
    p.add_argument("--setup-only", action="store_true", help="Install/download/CPU-check only; do not train")
    p.add_argument("--env-dir", type=Path, default=ROOT / ".pilot/env")
    p.add_argument("--python", dest="env_python", type=Path, help="Reuse this Python 3.12 environment, without installing packages")
    p.add_argument("--reinstall", action="store_true", help="Reapply the dependency recipe in the managed environment")
    p.add_argument("--task", choices=("download", "flash", "check"), help=argparse.SUPPRESS)
    p.add_argument("--gpu-check", action="store_true", help=argparse.SUPPRESS)
    return p


def local_env():
    env = os.environ.copy()
    # Do not inherit the offline launcher's flags into the online download step.
    env.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", PYTHONUNBUFFERED="1",
               HF_HUB_DOWNLOAD_TIMEOUT="120", HF_HUB_ETAG_TIMEOUT="30",
               TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               PIP_CACHE_DIR=str(ROOT / ".pilot/pip_cache"), SWANLAB_MODE="offline", WANDB_MODE="disabled")
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env.pop("PYTHONHOME", None)
    env.pop("RAY_ADDRESS", None)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "verl"), str(ROOT)))
    return env


class SetupRunner:
    """Guard pip/download processes before psutil exists in the new environment."""

    def __init__(self, c):
        self.c = c
        self.start = time.monotonic()
        self.deadline = self.start + c["wall_limit_seconds"]
        self.events = []

    def spent(self):
        return self.c["already_spent_rmb"] + (
            time.monotonic() - self.start
        ) / 3600 * self.c["billable_gpus"] * self.c["price_rmb_per_gpu_hour"]

    def run(self, name, command, env=None):
        remaining = self.deadline - time.monotonic() - 15
        if remaining <= 0:
            raise TimeoutError("Preparation exhausted the usable budget; training will not start")
        print(f"\n[{name}] remaining={remaining / 60:.1f} min; estimated spending=¥{self.spent():.2f}", flush=True)
        log_path = Path(self.c["output"]) / (name + ".log")
        begun = time.monotonic()
        code = None
        with log_path.open("wb") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env or local_env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            os.set_blocking(proc.stdout.fileno(), False)
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            try:
                while proc.poll() is None:
                    if time.monotonic() >= self.deadline - 15:
                        raise TimeoutError("Total setup+training time limit reached")
                    for key, _ in selector.select(timeout=0.3):
                        try:
                            chunk = os.read(key.fileobj.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if chunk:
                            log.write(chunk)
                            log.flush()
                            print(chunk.decode(errors="replace"), end="", flush=True)
                        else:
                            selector.unregister(key.fileobj)
                while True:
                    if time.monotonic() >= self.deadline - 15:
                        break
                    try:
                        tail = os.read(proc.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not tail:
                        break
                    log.write(tail)
                    print(tail.decode(errors="replace"), end="", flush=True)
                code = proc.returncode
            finally:
                selector.close()
                if proc.poll() is None:
                    # SIGINT lets the inner pilot clean its private Ray process tree.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                # Installer helpers sometimes outlive their immediate parent;
                # their own process group is isolated from other server jobs.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.stdout.close()
                self.events.append({"name": name, "returncode": proc.returncode,
                                    "seconds": time.monotonic() - begun})
                pilot.write_json(Path(self.c["output"]) / "bootstrap_stages.json", self.events)
        if code:
            raise RuntimeError(f"{name} failed ({code}); see {log_path}")


def safetensors_complete(path):
    """Check the index and tensor data boundaries without loading large tensors."""
    path = Path(path)
    if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        return False
    indices = list(path.glob("*.safetensors.index.json"))
    try:
        names = set()
        for index in indices:
            names.update(pilot.read_json(index)["weight_map"].values())
        if not names:
            names = {p.name for p in path.glob("*.safetensors")}
        if not names:
            return False
        for name in names:
            target = path / name
            if target.parent.resolve() != path.resolve():
                return False
            size = target.stat().st_size
            with target.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    return False
                length = struct.unpack("<Q", prefix)[0]
                if length > 100_000_000 or length + 8 >= size:
                    return False
                header = json.loads(stream.read(length))
            ends = [entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__"]
            if not ends or max(ends) != size - 8 - length:
                return False
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False
    return True


def download(c):
    from huggingface_hub import HfApi, snapshot_download

    locations = {"student": c["student"], **c["teachers"]}
    for label, repo in SOURCES.items():
        target = Path(locations[label])
        target.mkdir(parents=True, exist_ok=True)
        lock = target / ".pilot-source.json"
        source = pilot.read_json(lock) if lock.exists() else None
        if source and source["repo_id"] != repo:
            raise ValueError(f"Model directory belongs to another source: {target}")
        if source and source.get("complete") and safetensors_complete(target):
            print(f"Reusing complete {repo}@{source['revision']}: {target}", flush=True)
            continue
        # Pin the commit BEFORE download; interrupted retries use the same revision.
        if source is None:
            source = {"repo_id": repo, "revision": HfApi().model_info(repo).sha, "complete": False}
            pilot.write_json(lock, source)
        print(f"Downloading {repo}@{source['revision']} -> {target}", flush=True)
        for attempt in range(3):
            try:
                snapshot_download(repo_id=repo, revision=source["revision"], local_dir=target,
                                  max_workers=4, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                               "*.tiktoken", "*.jinja", "merges.txt", "vocab.txt"])
                if not safetensors_complete(target):
                    raise ValueError("Downloaded model is incomplete; check shard index and file lengths")
                source["complete"] = True
                source["sizes"] = {p.name: p.stat().st_size for p in target.glob("*.safetensors")}
                pilot.write_json(lock, source)
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"Download interrupted; retry {attempt + 2}/3", flush=True)
                time.sleep(3)


def flash_install():
    import torch

    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError("FlashAttention wheel selection requires Python 3.12 and torch 2.8.0")
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    name = f"flash_attn-2.8.1+cu12torch2.8cxx11abi{abi}-cp312-cp312-linux_x86_64.whl"
    url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/" + name
    print(f"Installing FlashAttention matching PyTorch ABI={abi}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", url + "#sha256=" + FLASH_HASHES[abi]], check=True)


def check_environment(c, gpu):
    import torch
    from hydra import compose, initialize_config_dir

    expected = {"torch": "2.8.0", "vllm": "0.11.0", "transformers": "4.56.2", "flash-attn": "2.8.1"}
    for package, version in expected.items():
        installed = importlib.metadata.version(package)
        if installed.split("+")[0] != version:
            raise RuntimeError(f"{package}={installed}; this recipe requires {version}")
    with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
        for group in pilot.GROUPS:
            compose(config_name="ppo_trainer", overrides=pilot.train_args(c, group))
    from verl.utils.reward_score.ttrl_math import compute_score
    from scripts.val.eval.utils import grade_answer_verl

    if not compute_score(r"\boxed{42}", "42")["acc"] or not grade_answer_verl(r"\boxed{42}", "42"):
        raise RuntimeError("Math grading smoke check failed")
    if gpu:
        pilot.preflight_gpu(c)
        importlib.import_module("vllm._C")  # Check compiled bindings before downloading weights.
        from flash_attn import flash_attn_func
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker

        for index in range(c["gpus"]):
            with torch.cuda.device(index):
                q = torch.randn(1, 16, 2, 64, device=f"cuda:{index}", dtype=torch.bfloat16)
                flash_attn_func(q, q, q)
                torch.cuda.synchronize()
                del q
        print(f"CUDA, FlashAttention, vLLM and local workers checked on {c['gpus']} GPUs", flush=True)
    else:
        print("CPU checks passed. CUDA kernels must still be checked on the GPU server.", flush=True)


def select_python(a, runner, state):
    if a.env_python:
        return a.env_python.resolve()
    target = a.env_dir.resolve()
    interpreter = target / "bin/python"
    if interpreter.exists():
        return interpreter
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Incomplete environment at {target}; choose a new --env-dir")
    candidate = shutil.which("python3.12")
    if sys.version_info[:2] == (3, 12):
        candidate = sys.executable
    if candidate:
        runner.run("create_env", [candidate, "-m", "venv", str(target)])
    else:
        # uv downloads managed CPython into the project, without changing system Python.
        tools = state / "bootstrap_tools"
        runner.run("install_uv", [sys.executable, "-m", "pip", "install", "--target", str(tools), "uv>=0.8,<1"])
        env = local_env()
        env.update(PYTHONPATH=str(tools), UV_PYTHON_INSTALL_DIR=str(state / "python"), UV_CACHE_DIR=str(state / "uv_cache"))
        runner.run("create_env", [sys.executable, "-m", "uv", "venv", "--python", "3.12", "--seed", str(target)], env)
    return interpreter


def bootstrap(a):
    if a.worker or a.prepare_only:
        raise ValueError("For this entry use --setup-only; --prepare-only belongs to the offline launcher")
    c = pilot.make_config(a)
    state = ROOT / ".pilot"
    if a.dry_run:
        print(json.dumps({"mode": "setup-only" if a.setup_only else "setup-and-train", "config": c,
                          "python": str(a.env_python or a.env_dir / "bin/python"), "sources": SOURCES,
                          "steps": ["create/reuse Python 3.12", "install pinned dependencies + matching FlashAttention",
                                    "CPU/CUDA checks", "download/reuse pinned model revisions", "prepare data",
                                    "subtract setup fees", "two-teacher pilot + independent evaluation"]}, ensure_ascii=False, indent=2))
        return 0
    if sys.version_info < (3, 10) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Server entry supports Linux x86_64, bootstrap Python >=3.10, Python/pip and internet access")
    for relative in ("datasets/dapo-math-17k.parquet", "datasets/test_data/AMC23/test.parquet"):
        path = ROOT / relative
        header = b""
        if path.is_file():
            with path.open("rb") as stream:
                header = stream.read(4)
        if header != b"PAR1":
            raise RuntimeError(f"Missing real Parquet data: {path}. Transfer the whole project including datasets (not LFS pointers).")
    out = Path(c["output"])
    if out.exists():
        raise ValueError("Use a fresh --output; download/environment caches are reused separately")
    out.mkdir(parents=True)
    runner = SetupRunner(c)
    state.mkdir(exist_ok=True)
    pilot.write_json(out / "bootstrap_config.json", c)
    config = out / "bootstrap_config.json"
    # A single lock protects the shared environment and model destinations.
    with (state / "setup.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another bootstrap is using this project; use a separate checkout") from exc
        try:
            cached = all(safetensors_complete(Path(p)) for p in [c["student"], *c["teachers"].values()])
            needed_gib = 55 if cached else 100
            if shutil.disk_usage(ROOT).free < needed_gib * 1024**3:
                raise RuntimeError(f"Need about {needed_gib} GiB free in this project filesystem for environment, weights and checkpoints")
            python = select_python(a, runner, state)
            runner.run("python_version", [str(python), "-c", "import sys; assert sys.version_info[:2]==(3,12), 'Python 3.12 required'"])
            requirements = ROOT / "scripts/pilot/requirements-server.txt"
            fingerprint = hashlib.sha256(requirements.read_bytes() + b"flash2.8.1-local-verl-v1").hexdigest()
            marker = a.env_dir.resolve() / ".pilot-ready.json"
            ready = marker.exists() and pilot.read_json(marker).get("recipe") == fingerprint
            if not a.env_python and (not ready or a.reinstall):
                runner.run("install_pip", [str(python), "-m", "pip", "install", "pip>=24,<26", "setuptools>=77,<80", "wheel"])
                runner.run("install_deps", [str(python), "-m", "pip", "install", "-r", str(requirements)])
                runner.run("install_verl", [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT / "verl")])
                runner.run("install_flash", [str(python), str(SELF), "--task", "flash"])
            checks = [str(python), str(SELF), "--task", "check", "--config", str(config)]
            runner.run("check_environment", checks + ([] if a.setup_only else ["--gpu-check"]))
            runner.run("pip_freeze", [str(python), "-m", "pip", "freeze"])
            shutil.copyfile(out / "pip_freeze.log", out / "pip-freeze.txt")
            runner.run("pip_check", [str(python), "-m", "pip", "check"])
            if not a.env_python:
                pilot.write_json(marker, {"recipe": fingerprint})
            runner.run("download_models", [str(python), str(SELF), "--task", "download", "--config", str(config)])
            # Freeze provenance of this run even if the source cache changes later.
            pilot.write_json(out / "model_sources.json", {
                label: pilot.read_json(Path(path) / ".pilot-source.json")
                for label, path in {"student": c["student"], **c["teachers"]}.items()
            })
            env = local_env()
            env["PYTHON"] = str(python)
            spent = runner.spent()
            pilot_args = ["--output", str(out), "--student", c["student"],
                          "--teacher-nonthinking", c["teachers"]["nonthinking"], "--teacher-grpo", c["teachers"]["grpo"],
                          "--gpus", str(c["gpus"]), "--billable-gpus", str(c["billable_gpus"]),
                          "--price", str(c["price_rmb_per_gpu_hour"]), "--budget", str(c["budget_rmb"]),
                          "--reserve", str(c["reserve_rmb"]), "--already-spent", str(spent)]
            for key in ("steps", "save_every", "batch", "rollouts", "response_length", "train_size", "eval_size", "eval_samples", "seed"):
                pilot_args += ["--" + key.replace("_", "-"), str(c[key])]
            if a.setup_only:
                pilot_args.append("--prepare-only")
            else:
                available = runner.deadline - time.monotonic()
                if available < 30 * 60:
                    raise RuntimeError("Less than 30 minutes remain after setup; models are cached, but training was skipped to preserve budget")
            runner.run("prepare_data" if a.setup_only else "pilot", ["bash", str(ROOT / "pilot_thinking_consistency.sh"), *pilot_args], env)
            pilot.write_json(out / "bootstrap_result.json", {"status": "prepared" if a.setup_only else "finished",
                             "estimated_spent_rmb": runner.spent(), "setup_only": a.setup_only})
            print(f"\nCompleted. Results: {out}\nGPU 实例仍需手动关机。", flush=True)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            pilot.write_json(out / "bootstrap_result.json", {"status": "failed_or_budget_stopped", "error": str(exc),
                             "estimated_spent_rmb": runner.spent()})
            print(f"\nSTOP: {exc}\nLogs: {out}\nGPU 实例仍需手动关机。", file=sys.stderr, flush=True)
            return 1


def main():
    a = arguments().parse_args()
    if a.task:
        if a.task == "flash":
            flash_install()
        elif a.task == "download":
            download(pilot.read_json(a.config))
        else:
            check_environment(pilot.read_json(a.config), a.gpu_check)
        return 0
    return bootstrap(a)


if __name__ == "__main__":
    raise SystemExit(main())
