"""Local control checks only; hardware acceptance is a separate case run."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import yaml

spec=importlib.util.spec_from_file_location('official_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_manifest_detects_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'a.py').write_text('before')
            before=runner.manifest(root)
            (root/'a.py').write_text('after')
            self.assertNotEqual(before,runner.manifest(root))

    def test_manifest_ignores_cache_but_not_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'__pycache__').mkdir()
            (root/'__pycache__'/'a.pyc').write_bytes(b'cache')
            (root/'lib.so').write_bytes(b'library')
            self.assertEqual(set(runner.manifest(root)),{'lib.so'})

    def test_baseline_and_template_identity(self):
        case=Path(__file__).resolve().parents[1]
        image=yaml.safe_load((case/'image.baseline.yaml').read_text())
        compose=yaml.safe_load((case/'demo/docker-compose.yml').read_text())
        self.assertEqual(image['image']['id'],runner.IMAGE)
        service=compose['services']['runtime-dev']
        self.assertEqual(service['image'],image['image']['ref'])
        self.assertFalse(service.get('privileged',False))
        self.assertEqual(service['restart'],'no')
        self.assertEqual(len(service['devices']),4)
        self.assertIn('${FLAGRT_SOURCE:?required}:/code:ro',service['volumes'])


if __name__=='__main__':
    unittest.main()
