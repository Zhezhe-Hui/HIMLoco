"""Offline checks for source relocation and preservation of board data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('go2_deployment', ROOT/'deploy/deploy_runtime.py')
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


class LayoutTest(unittest.TestCase):
    def test_deployment_manifest_resolves_without_local_experiment_data(self):
        paths = deployment.runtime_files(ROOT)
        self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue(all('runs' not in path.parts and 'data' not in path.parts for path in paths))
        self.assertNotIn(ROOT/'go2_runtime/calibration.local.json', paths)
        spec = importlib.util.spec_from_file_location('layout_runtime', ROOT/'go2_runtime/run_ours.py')
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        self.assertEqual(runtime.ROOT, ROOT)
        mapper, mppi, config = runtime.algorithms()
        self.assertTrue(hasattr(mppi, 'FootprintMPPIController'))

    def test_legacy_migration_preserves_original_calibration_and_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root/'deploy/ours'
            (old/'runs/trial').mkdir(parents=True)
            (old/'runs/trial/telemetry.jsonl').write_bytes(b'original telemetry\n')
            (old/'calibration.local.json').write_bytes(b'original calibration')
            web = root/'real_robot_deploy/web_console/data'
            web.mkdir(parents=True)
            (web/'calibration.web.json').write_bytes(b'board measurement')
            deployment.migrate_layout(root)
            self.assertFalse((root/'real_robot_deploy').exists())
            self.assertEqual((root/'go2_runtime/runs/trial/telemetry.jsonl').read_bytes(), b'original telemetry\n')
            self.assertEqual((root/'go2_runtime/calibration.local.json').read_bytes(), b'original calibration')
            self.assertEqual((root/'go2_web/data/calibration.web.json').read_bytes(), b'board measurement')
            deployment.migrate_layout(root)  # Repeating maintenance does not remove data.

    def test_coexisting_layouts_are_rejected_without_partial_move(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'deploy/ours').mkdir(parents=True)
            (root/'go2_web').mkdir()
            (root/'real_robot_deploy/web_console').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                deployment.migrate_layout(root)
            self.assertTrue((root/'deploy/ours').exists())
            self.assertFalse((root/'go2_runtime').exists())


if __name__ == '__main__':
    unittest.main()
