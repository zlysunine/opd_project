"""CPU checks for cost accounting, process isolation and fair comparisons.

Run with a Python environment containing psutil:
    python -m unittest discover -s scripts/pilot -p 'test_*.py' -v
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil


spec = importlib.util.spec_from_file_location("pilot", Path(__file__).with_name("thinking_consistency.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotChecks(unittest.TestCase):
    def config(self, directory, *extra):
        return pilot.make_config(pilot.parser().parse_args(["--output", str(directory), *extra]))

    def test_budget_accounts_for_every_rented_gpu_and_previous_fees(self):
        c = self.config("/tmp/not-created", "--gpus", "2", "--billable-gpus", "8", "--already-spent", "3")
        modeled = c["wall_limit_seconds"] / 3600 * 8 * c["price_rmb_per_gpu_hour"]
        self.assertAlmostEqual(modeled + 3 + c["reserve_rmb"], 30)
        self.assertLess(c["wall_limit_seconds"], 1800)

    def test_invalid_budget_never_starts_a_run(self):
        for args in [("--budget", "31"), ("--price", "nan"), ("--already-spent", "26"),
                     ("--billable-gpus", "1"), ("--train-size", "8"), ("--steps", "31")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.config("/tmp/not-created", *args)

    def test_interrupted_checkpoint_cannot_win_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (10, 20):
                actor = root / f"global_step_{step}" / "actor"
                (actor / "huggingface").mkdir(parents=True)
                (actor / "huggingface/config.json").write_text("{}")
                for rank in range(2):
                    (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(b"test")
            # Step 20 has files, but the save barrier/tracker was never completed.
            (root / "latest_checkpointed_iteration.txt").write_text("10")
            self.assertEqual(set(pilot.checkpoints(root, 2)), {10})

    def test_timeout_cleans_detached_children_without_touching_other_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.config(root)
            runner = pilot.BudgetRunner(c)
            sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            pidfile = root / "child.pid"
            code = (
                "import subprocess,sys,time,pathlib; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)"
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = runner.run("timeout", [sys.executable, "-c", code], 1.5)
                self.assertEqual(status, 124)
                child_pid = int(pidfile.read_text())
                self.assertTrue(not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
                self.assertIsNone(sibling.poll())
                self.assertLess(time.monotonic() - runner.start, 10)
            finally:
                sibling.terminate()
                sibling.wait()

    def test_report_does_not_call_overlap_only_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = self.config(root)
            runner = pilot.BudgetRunner(c)
            for label, overlap in [("nonthinking", 0.55), ("grpo", 0.70)]:
                (root / f"train_{label}.log").write_text(
                    f"step:1 - training/global_step:1 - val-topk/overlap_ratio:{overlap}\n"
                )
            for label in ("initial", "nonthinking", "grpo"):
                pilot.write_json(root / f"eval_{label}.json", {
                    "avg_accuracy": 0.5, "per_question": [0, 1], "truncation_rate": 0,
                    "mean_tokens": 200, "grading_timeouts": 0,
                })
            with contextlib.redirect_stdout(io.StringIO()):
                pilot.report(c, runner, 10)
            summary = json.loads((root / "summary.json").read_text())
            self.assertIn("未得到清晰", summary["verdict"])
            self.assertEqual(summary["grpo_minus_nonthinking"]["gap"], 0)
            self.assertTrue((root / "overlap.svg").exists())


if __name__ == "__main__":
    unittest.main()
