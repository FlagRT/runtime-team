"""Local wiring checks; synthetic CPU provider, no PPU qualification."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from admission import OPERATORS,configuration,check_model_routes
from flagrt_adapter.adapter import FrameworkAdapter
from flagrt_adapter.config import load_config
from flagrt_adapter.policy import Environment


class Controls(unittest.TestCase):
    def test_configuration_roundtrip(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            for mode in ('off','unknown'):
                config=configuration(mode);config['embedding_records']=None
                path=root/(mode+'.json');path.write_text(json.dumps(config))
                parsed,records=load_config(path)
                self.assertEqual(parsed,config);self.assertEqual(records,())
            with self.assertRaises(ValueError):configuration('candidate')

    def test_no_legacy_model_approval(self):
        self.assertIsNone(configuration('unknown')['workload_reviews'])
        self.assertTrue(configuration('on')['require_target'])
        self.assertEqual(len(OPERATORS['embedding']['signatures']),2)

    def test_model_route_checker_rejects_missing_or_extra_target(self):
        expected=[['embedding','flaggems','forward','completed',2],
                  ['embedding','flaggems','backward','completed',2],
                  ['relu','native','forward','operator_disabled',3],
                  ['sigmoid','native','forward','operator_disabled',1]]
        check_model_routes(expected,'on')
        with self.assertRaises(AssertionError):check_model_routes(expected[:2],'on')
        with self.assertRaises(AssertionError):check_model_routes(expected+[['sigmoid','flaggems','forward','completed',1]],'on')

    def test_public_deepfm_cpu_mixed_scope(self):
        # The same fixed public reference source, no injected model approximation.
        source=Path(__file__).resolve().parents[3]/'model-defs/deepfm/model.py'
        if not source.exists():
            # Repo currently uses a differently named subdirectory; locate the
            # exact hash, not an arbitrary similarly named model.
            import hashlib
            root=Path(__file__).resolve().parents[3]/'model-defs'
            matches=[p for p in root.rglob('model.py') if hashlib.sha256(p.read_bytes()).hexdigest()==
                '319ca8968ee79f233a70338991cdb0f2cdd91dcb1d61329a14d3018a14434eef']
            self.assertEqual(len(matches),1);source=matches[0]
        spec=importlib.util.spec_from_file_location('public_deepfm_cpu',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.DeepFM([1000,2000,500,300,50])
        env=Environment(*(['synthetic']*7),backend='ppu',device_type='cuda')
        gems=SimpleNamespace(embedding=lambda w,i,*opts:torch.ops.aten.embedding.default(w,i,*opts),
            embedding_backward=lambda g,i,n,*opts:torch.ops.aten.embedding_backward.default(g,i,n,*opts))
        runtime=SimpleNamespace(synchronize=lambda _:None)
        for mode,reason in [('on',None),('off','experimental_disabled'),('unknown','workload_unverified')]:
            policy=SimpleNamespace(operator='embedding',reason=lambda *_,reason=reason:reason,strict=False)
            model.zero_grad(set_to_none=True)
            with FrameworkAdapter({'embedding':policy},env,runtime,gems) as scope:
                model(torch.zeros((32,5),dtype=torch.int64)).sum().backward()
            check_model_routes([list(k)+[v] for k,v in scope.operator_counts.items()],mode)


if __name__=='__main__':unittest.main()
