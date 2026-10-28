"""Test-only instrumentation around unchanged installed training CLI and public entry."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--scenario', choices=('control', 'forward', 'backward'), required=True)
    parser.add_argument('--guarded', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    output = Path(os.environ['DC_OUT_DIR']); output.mkdir(parents=True, exist_ok=True)
    path = output / f'events-rank{rank}.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    lock = threading.RLock()

    def emit(kind, **fields):
        with lock:
            os.write(fd, (json.dumps(dict(event=kind, rank=rank, pid=os.getpid(),
                monotonic=time.monotonic(), **fields))+'\n').encode())
            os.fsync(fd)

    emit('start', restart_count=os.environ.get('TORCHELASTIC_RESTART_COUNT', '0'))
    def terminated(signum, frame):
        emit('signal', number=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminated)
    import torch
    import torch.distributed as dist
    import flag_gems
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter import train_cli
    if args.guarded:
        from flagrt_adapter.step_guard import StepCoordinator
        original_ready = StepCoordinator.ready
        def ready(self, phase, step):
            emit('step_gate_enter', phase=phase, step=step)
            return original_ready(self, phase, step)
        StepCoordinator.ready = ready

    original_init = dist.init_process_group
    def init(*a, **kw):
        # Explicit test harness timeout, not a change to the public source.
        kw['timeout'] = timedelta(seconds=60)
        result = original_init(*a, **kw)
        emit('group_ready', backend=str(dist.get_backend()))
        return result
    dist.init_process_group = init
    original_reduce = dist.all_reduce
    def reduce(tensor, *a, **kw):
        if tensor.numel() > 1:
            emit('gradient_collective_enter')
            # Persistent marker establishes peer blocked/entering gradient sync
            # before the injecting rank raises; it is NOT a stop protocol.
            marker=output/f'gradient-enter-rank{rank}'
            if not marker.exists(): marker.touch()
        return original_reduce(tensor, *a, **kw)
    dist.all_reduce = reduce
    original_step = torch.optim.AdamW.step
    def step(self, *a, **kw):
        emit('optimizer_enter')
        value = original_step(self, *a, **kw)
        emit('optimizer_done')
        return value
    torch.optim.AdamW.step = step
    original_record = ElementwiseAdapter.record
    def record(self, backend, stage, reason, operator='silu'):
        emit('adapter', backend=backend, stage=stage, reason=reason, operator=operator)
        return original_record(self, backend, stage, reason, operator)
    ElementwiseAdapter.record = record
    current = threading.local()
    original_target = ElementwiseAdapter.target
    def target(self, stage, *a, **kw):
        current.stage = stage
        try: return original_target(self, stage, *a, **kw)
        finally: current.stage = None
    ElementwiseAdapter.target = target
    original_context = flag_gems.use_gems
    injected = False
    @contextmanager
    def context(*a, **kw):
        nonlocal injected
        with original_context(*a, **kw):
            if rank == 0 and not injected and current.stage == args.scenario:
                deadline=time.monotonic()+120
                peer_marker=(output/'step-control/rank1/backward-0' if args.guarded
                             else output/'gradient-enter-rank1')
                while not peer_marker.exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Peer did not reach expected gate/collective; injection invalid')
                    time.sleep(.05)
                injected=True
                emit('injected', stage=args.scenario)
                raise RuntimeError('INJECTED_ADAPTER_FAILURE_'+args.scenario)
            yield
    flag_gems.use_gems = context
    before=None
    try:
        # Loading torch_npu here only establishes dispatcher tables for the audit.
        import torch_npu
        before={op:torch._C._dispatch_dump_table('aten::'+op) for op in ('silu','silu_backward')}
        sys.argv=[sys.argv[0], '--config', args.config]
        train_cli.main()
        emit('completed')
    except BaseException as exc:
        emit('raised', exception=type(exc).__name__, message=str(exc))
        raise
    finally:
        emit('finally', dispatcher_restored=before is not None and before == {
            op:torch._C._dispatch_dump_table('aten::'+op) for op in before})
        os.close(fd)


if __name__ == '__main__': main()
