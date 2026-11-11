"""PPU measured identity; reuses public runtime. Does not register kernels."""
from collections import Counter
from dataclasses import asdict
import importlib.metadata as md
import logging
import os

class Hits(logging.Handler):
    def __init__(self, operator='sigmoid'):
        super().__init__()
        self.counts=Counter()
        self.operator=operator
    def emit(self, record):
        message=record.getMessage()
        if message in ('GEMS SIGMOID FORWARD','GEMS SIGMOID BACKWARD'):
            self.counts[message]+=1

def inspect_environment(args, report):
    import torch
    import torch.nn.functional as F
    import triton
    import runtime
    import flag_gems as gems
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.policy import Environment, OperatorPolicy
    from flagrt_adapter.records import load_records
    from flagrt_adapter.startup import package_content_identity

    runtime.use('ppu')
    assert runtime.device_count() == 1, 'Expected one mapped physical PPU'
    runtime.set_device(0)
    chip = torch.cuda.get_device_name(0)
    assert 'PPU' in chip, 'Refuse a non-PPU CUDA-compatible device'
    provider, detail = package_content_identity(gems.__file__)
    actual = Environment(chip=chip, torch=torch.__version__,
                         backend_version=os.environ['PPU_DRIVER_ID'],
                         gems_revision=provider, image_id=os.environ['CASE_IMAGE_ID'],
                         triton=triton.__version__, compiler_build=md.version('flagtree'),
                         backend='ppu', device_type='cuda')
    assert actual.complete()
    report.update(adapter_version=md.version('flagrt-framework-adapter'),environment=asdict(actual), provider=detail,
                  versions={**{n: md.version(n) for n in ['flag-gems', 'flagtree', 'torch',
                            'flagrt-framework-adapter']}, 'triton-module': triton.__version__},
                  provider_route=os.environ.get('GEMS_VENDOR'),
                  runtime_version=runtime.__component_version__)
    print('ENVIRONMENT', report['environment'], flush=True)
