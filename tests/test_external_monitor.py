from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExternalMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = REPO_ROOT / ".test_artifacts" / self._testMethodName
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

    def test_external_monitor_writes_metrics_json_and_patches_result(self) -> None:
        result_path = self.workdir / "result.json"
        metrics_path = self.workdir / "metrics.json"
        result_path.write_text(
            json.dumps(
                {
                    "design_name": "d1",
                    "total_runtime_sec": 123.0,
                    "peak_memory_kb": 456.0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "external_monitor.py"),
            f"--metrics-json={metrics_path}",
            f"--patch-json={result_path}",
            "--",
            sys.executable,
            "-c",
            "import time; payload = [0] * 200000; time.sleep(0.05); print(len(payload))",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["returncode"], 0)
        self.assertIsInstance(metrics["runtime_sec"], float)
        self.assertGreater(metrics["runtime_sec"], 0.0)

        patched = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(patched["total_runtime_sec"], metrics["runtime_sec"])
        self.assertEqual(patched["peak_memory_kb"], metrics["peak_memory_kb"])

    def test_external_monitor_patches_single_result_inside_aggregate_json(self) -> None:
        result_path = self.workdir / "aggregate.json"
        result_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "benchmark": "design_1/input.blif",
                            "runtime_sec": 1.0,
                            "peak_memory_kb": 2.0,
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "external_monitor.py"),
            f"--patch-json={result_path}",
            "--",
            sys.executable,
            "-c",
            "print('ok')",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        patched = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertIn("runtime_sec", patched["results"][0])
        self.assertIn("peak_memory_kb", patched["results"][0])
        self.assertNotEqual(patched["results"][0]["runtime_sec"], 1.0)


if __name__ == "__main__":
    unittest.main()
