"""CPU coverage for the online setup entry; no real downloads or GPU imports."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("server", Path(__file__).with_name("server_bootstrap.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def tiny_model(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text("{}")
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


class ServerChecks(unittest.TestCase):
    def config(self, directory):
        root = Path(directory)
        return server.pilot.make_config(server.arguments().parse_args([
            "--output", str(root / "run"), "--student", str(root / "student"),
            "--teacher-nonthinking", str(root / "nonthinking"), "--teacher-grpo", str(root / "grpo"),
        ]))

    def test_truncated_weights_or_missing_shard_are_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_model(root)
            self.assertTrue(server.safetensors_complete(root))
            weights = root / "model.safetensors"
            weights.write_bytes(weights.read_bytes()[:-1])
            self.assertFalse(server.safetensors_complete(root))
            tiny_model(root)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
            self.assertFalse(server.safetensors_complete(root))

    def test_download_pins_commits_and_reuses_complete_models_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            calls = []
            fake = types.ModuleType("huggingface_hub")

            class Api:
                def model_info(self, repo):
                    calls.append(("resolve", repo))
                    return types.SimpleNamespace(sha="fixed-commit")

            def snapshot(**kwargs):
                calls.append(("download", kwargs["repo_id"]))
                self.assertEqual(kwargs["revision"], "fixed-commit")
                tiny_model(kwargs["local_dir"])

            fake.HfApi = Api
            fake.snapshot_download = snapshot
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake}), contextlib.redirect_stdout(io.StringIO()):
                server.download(c)
                self.assertEqual(len(calls), 6)
                server.download(c)
                self.assertEqual(len(calls), 6)
            source = server.pilot.read_json(Path(c["student"]) / ".pilot-source.json")
            self.assertTrue(source["complete"])
            self.assertEqual(source["revision"], "fixed-commit")

    def test_interrupted_download_reuses_original_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            locations = {"student": c["student"], **c["teachers"]}
            for label, path in locations.items():
                server.pilot.write_json(Path(path) / ".pilot-source.json", {
                    "repo_id": server.SOURCES[label], "revision": "old-pinned-commit", "complete": False,
                })
            fake = types.ModuleType("huggingface_hub")
            fake.HfApi = mock.Mock(side_effect=AssertionError("Should not resolve HEAD again"))

            def snapshot(**kwargs):
                self.assertEqual(kwargs["revision"], "old-pinned-commit")
                tiny_model(kwargs["local_dir"])

            fake.snapshot_download = snapshot
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake}), contextlib.redirect_stdout(io.StringIO()):
                server.download(c)

    def test_prepare_time_is_charged_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            Path(c["output"]).mkdir()
            with mock.patch.object(server.time, "monotonic", side_effect=[100, 1900]):
                runner = server.SetupRunner(c)
                # Half an hour, two paid cards.
                self.assertAlmostEqual(runner.spent(), 5.59)

    def test_budget_timeout_stops_a_setup_process(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            Path(c["output"]).mkdir()
            c["wall_limit_seconds"] = 15.6  # Leaves 0.6s before the cleanup reserve.
            runner = server.SetupRunner(c)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
                runner.run("fake_download", [sys.executable, "-c", "import time; time.sleep(60)"])
            stages = server.pilot.read_json(Path(c["output"]) / "bootstrap_stages.json")
            self.assertIsNotNone(stages[0]["returncode"])

    def test_setup_only_plan_has_no_filesystem_or_network_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            a = server.arguments().parse_args(["--dry-run", "--setup-only", "--output", str(Path(directory) / "absent")])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(server.bootstrap(a), 0)
            self.assertFalse(Path(a.output).exists())

    def test_exited_parent_with_child_stdout_cannot_block_budget_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.config(directory)
            Path(c["output"]).mkdir()
            runner = server.SetupRunner(c)
            # The parent exits but its child inherits stdout. A blocking read()
            # after parent exit would hang here for 60s and bypass the timer.
            command = [sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"]
            start = time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run("parent_exit", command)
            self.assertLess(time.monotonic() - start, 5)

    def test_online_entry_passes_setup_fees_to_the_offline_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = server.arguments().parse_args([
                "--output", str(root / "run"), "--python", sys.executable,
                "--student", str(root / "student"), "--teacher-nonthinking", str(root / "nonthinking"),
                "--teacher-grpo", str(root / "grpo"), "--already-spent", "2",
            ])
            commands = []

            def fake_run(runner, name, command, env=None):
                commands.append((name, command))
                if name == "pip_freeze":
                    (Path(runner.c["output"]) / "pip_freeze.log").write_text("torch==2.8.0\n")
                if name == "download_models":
                    locations = {"student": runner.c["student"], **runner.c["teachers"]}
                    for label, location in locations.items():
                        server.pilot.write_json(Path(location) / ".pilot-source.json", {
                            "repo_id": server.SOURCES[label], "revision": "fixed", "complete": True,
                        })
                # Simulate 10 minutes spent installing/downloading on rented GPUs.
                if name == "download_models":
                    runner.start -= 600
                    runner.deadline -= 600

            # Keep even bootstrap locks out of the real project during this test.
            with mock.patch.object(server, "ROOT", root), \
                 mock.patch.object(server.shutil, "disk_usage", return_value=types.SimpleNamespace(free=200 * 1024**3)), \
                 mock.patch.object(server.SetupRunner, "run", fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                for relative in ("datasets/dapo-math-17k.parquet", "datasets/test_data/AMC23/test.parquet"):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"PAR1")
                recipe = root / "scripts/pilot/requirements-server.txt"
                recipe.parent.mkdir(parents=True)
                recipe.write_text("torch==2.8.0\n")
                self.assertEqual(server.bootstrap(a), 0)
            forwarded = next(command for name, command in commands if name == "pilot")
            spent = float(forwarded[forwarded.index("--already-spent") + 1])
            self.assertGreaterEqual(spent, 2 + 2 * 5.59 / 6)
            self.assertLess(spent, 4)
            self.assertEqual(forwarded[forwarded.index("--budget") + 1], "30.0")


if __name__ == "__main__":
    unittest.main()
