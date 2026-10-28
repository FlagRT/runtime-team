"""Host-only tests: python -m unittest discover -s <this directory>."""
import unittest
from pathlib import Path
from prepare import apply_exact, prepare


class ExactPatchTests(unittest.TestCase):
    def test_replacement(self):
        patch = '--- a/f\n+++ b/f\n@@ -1,2 +1,3 @@\n a\n-b\n+c\n+d\n'
        self.assertEqual(apply_exact('a\nb\n', patch), 'a\nc\nd\n')

    def test_reject_context_mismatch(self):
        patch = '--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n'
        with self.assertRaises(ValueError):
            apply_exact('different\n', patch)

    def test_reject_wrong_sdk(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            prepare(b'unknown compiler', 'epilogue')

    def test_reject_wrong_hunk_count(self):
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            apply_exact('old\n', '--- a/f\n+++ b/f\n@@ -1,2 +1 @@\n-old\n+new\n')

    def test_diagnostic_profile_is_explicit(self):
        patch = (Path(__file__).with_name('03-sdnn-ttxir-diagnostics.patch')
                 .read_text(encoding='utf-8'))
        self.assertIn('XPU_SDNN_SKIP_EARLY_COMBINE', patch)
        self.assertIn('XPU_SDNN_PRESERVE_TTXIR_ERRORS', patch)


if __name__ == '__main__':
    unittest.main()
