import csv
import json
from pathlib import Path
import tempfile
import unittest

from legged_gym.scripts.run_navigation_benchmark import (
    load_resume_records,
    validate_resume_manifest,
    write_summary,
)


class NavigationBenchmarkSummaryTest(unittest.TestCase):
    def test_failed_routes_only_contribute_to_safety_metrics(self):
        rows = [
            {
                "method": "mppi", "success": True, "elapsed_s": 10.0,
                "distance_m": 8.0, "body_contact_steps": 0,
                "stall_escape_count": 0,
            },
            {
                "method": "mppi", "success": False, "elapsed_s": 60.0,
                "distance_m": 4.0, "body_contact_steps": 6,
                "stall_escape_count": 2, "failure_reason": "Timeout",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_summary(rows, Path(tmp))
            with open(Path(tmp) / "summary.json", encoding="utf-8") as fh:
                summary = json.load(fh)[0]
            with open(Path(tmp) / "summary.csv", encoding="utf-8") as fh:
                csv_rows = list(csv.DictReader(fh))

        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["elapsed_s_mean"], 10.0)
        self.assertEqual(summary["distance_m_mean"], 8.0)
        self.assertEqual(summary["body_contact_steps_mean"], 3.0)
        self.assertEqual(summary["stall_escape_count_mean"], 1.0)
        self.assertEqual(json.loads(summary["failure_reasons"]), {"Timeout": 1})
        self.assertEqual(len(csv_rows), 1)

    def test_resume_loads_only_complete_successful_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [
                {"method": "mppi", "seed": "7", "process_returncode": "0"},
                {"method": "fmm_fixed", "seed": "7", "process_returncode": "1"},
            ]
            with (out / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (out / "mppi_seed7.json").write_text(
                json.dumps({"success": True, "elapsed_s": 4.0}), encoding="utf-8")
            (out / "mppi_seed7.log").write_text("normal", encoding="utf-8")
            (out / "fmm_fixed_seed7.json").write_text(
                json.dumps({"success": True}), encoding="utf-8")
            (out / "fmm_fixed_seed7.log").write_text("normal", encoding="utf-8")
            records = load_resume_records(out)

        self.assertEqual(set(records), {("mppi", 7)})
        self.assertIs(records[("mppi", 7)]["success"], True)
        self.assertEqual(records[("mppi", 7)]["process_returncode"], 0)

    def test_resume_rejects_changed_experiment(self):
        existing = {"scene": "easy", "methods": ["mppi"],
                    "seed_start": 1, "num_seeds": 30, "vio": "ideal"}
        changed = dict(existing, scene="hard")
        with self.assertRaisesRegex(ValueError, "scene"):
            validate_resume_manifest(existing, changed)


if __name__ == "__main__":
    unittest.main()
