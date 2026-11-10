# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Driver-only analysis jobs; no model hooks, RNG use, or analysis dependencies."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def analysis_due(config, step):
    return bool(config and step in config.get("optimizer_steps", []))


def write_analysis_job(config, checkpoint, step):
    manifest = Path(config["probe_manifest"])
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != config["probe_manifest_hash"]:
        raise ValueError("Probe manifest changed during training")
    checkpoint = Path(checkpoint).resolve()
    native = config.get("method") == "sft" and (checkpoint / "native_sft.json").is_file()
    if not (checkpoint / "data.pt").is_file() or (not native and not list((checkpoint / "actor").glob("model_*.pt"))):
        raise ValueError("Analysis job requires a successfully saved training checkpoint")
    if native:
        from scripts.research.controlled_sft import verify_checkpoint
        if verify_checkpoint(checkpoint)["step"] != step:
            raise ValueError("Native SFT checkpoint step mismatch")
    if step == 0 and config.get("controlled_comparison", False):
        from analysis.checkpoints import verify_initial_adapter
        directory = Path(config["analysis_config"]).parent
        training = json.loads((directory / "training_config.json").read_text())
        fingerprint = verify_initial_adapter(checkpoint, training)
        for method in ("opsd", "sft", "grpo"):
            other = directory.parent / method / "initial_adapter.json"
            if other.is_file() and json.loads(other.read_text()) != fingerprint:
                raise ValueError("Initial LoRA differs across methods; refusing the first update")
        from scripts.research.common import write_json
        write_json(directory / "initial_adapter.json", fingerprint)
    job = {"optimizer_step": step, "checkpoint_path": str(checkpoint), "probe_manifest_hash": actual,
           "analysis_config": config["analysis_config"], "method": config["method"], "status": "pending"}
    out = Path(config["job_dir"]) / f"step_{step:06d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if json.loads(out.read_text()) != job:
            raise ValueError("Existing analysis job differs")
        return
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(job, indent=2) + "\n")
    temp.replace(out)


def finish_checkpoint(config, checkpoint, step):
    """Analyze while training is paused, then rotate; return non-training seconds."""
    if analysis_due(config, step):
        write_analysis_job(config, checkpoint, step)
    if config.get("defer_analysis_until_training_end", False):
        if not config.get("controlled_comparison") or config.get("method") != "sft":
            raise ValueError("Deferred analysis is restricted to native controlled SFT")
        from scripts.research.common import file_hash, write_json
        from scripts.research.retention import _saved_checkpoint, archive_adapter

        _saved_checkpoint(Path(checkpoint))
        analysis_path = Path(config["analysis_config"])
        write_json(Path(checkpoint) / "research_checkpoint.json", {
            "config_sha256": file_hash(analysis_path), "method": "sft", "step": step,
        })
        if analysis_due(config, step):
            archive_adapter(Path(checkpoint), analysis_path.parent, step)
        # Keep ALL pending checkpoints; rotation is allowed only after analysis.
        return 0.0
    if config.get("checkpoint_retention") != "latest":
        return 0.0
    project = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(project), str(project / "verl"), env.get("PYTHONPATH", "")])
    # A CPU-only Ray driver can have an empty CUDA_VISIBLE_DEVICES even though
    # its workers own GPUs. Restore the allocation captured by the launcher.
    if "cuda_visible_devices" in config:
        if config["cuda_visible_devices"] is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    command = [sys.executable, "-m", "scripts.research.retention", "--analysis-config", config["analysis_config"],
               "--checkpoint", str(checkpoint), "--step", str(step)]
    started = time.monotonic()
    subprocess.run(command, cwd=project, env=env, check=True)
    elapsed = time.monotonic() - started
    timings = Path(config["job_dir"]).parent / "checkpoint_maintenance.jsonl"
    with timings.open("a") as stream:
        stream.write(json.dumps({"optimizer_step": step, "analysis_and_rotation_seconds": elapsed}) + "\n")
    return elapsed
