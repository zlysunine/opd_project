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
from pathlib import Path


def analysis_due(config, step):
    return bool(config and step in config.get("optimizer_steps", []))


def write_analysis_job(config, checkpoint, step):
    manifest = Path(config["probe_manifest"])
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != config["probe_manifest_hash"]:
        raise ValueError("Probe manifest changed during training")
    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "data.pt").is_file() or not list((checkpoint / "actor").glob("model_*.pt")):
        raise ValueError("Analysis job requires a successfully saved training checkpoint")
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
