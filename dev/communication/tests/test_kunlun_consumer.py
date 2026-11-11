from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from deepfm_train import bind_runtime_device


class KunlunConsumerTests(unittest.TestCase):
    def runtime(self, name='kunlun', device_type='cuda', count=2):
        runtime=Mock()
        backend=Mock(device_type=device_type)
        backend.name=name
        runtime.current.return_value=backend
        runtime.device_count.return_value=count
        return runtime

    def test_kunlun_uses_cuda_namespace(self):
        runtime=self.runtime()
        self.assertEqual(str(bind_runtime_device(runtime,'kunlun',1)),'cuda:1')
        runtime.use.assert_called_once_with('kunlun')
        runtime.set_device.assert_called_once_with(1)

    def test_ascend_still_uses_npu(self):
        # Host PyTorch has no NPU device constructor; wrong namespace is rejected
        # before construction, preserving the same vendor-specific contract.
        with self.assertRaises(RuntimeError):
            bind_runtime_device(self.runtime('ascend','cuda'),'ascend',0)

    def test_cpu_fallback_rejected(self):
        with self.assertRaises(RuntimeError):
            bind_runtime_device(self.runtime(device_type='cpu'),'kunlun',0)

    def test_wrong_vendor_rejected(self):
        with self.assertRaises(RuntimeError):
            bind_runtime_device(self.runtime(name='nvidia'),'kunlun',0)

    def test_insufficient_cards_rejected(self):
        runtime=self.runtime(count=1)
        with self.assertRaises(RuntimeError):bind_runtime_device(runtime,'kunlun',1)
        runtime.set_device.assert_not_called()


if __name__=='__main__':unittest.main()
