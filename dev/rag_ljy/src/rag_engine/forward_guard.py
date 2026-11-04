"""Last-moment checks on the model thread, using exact padded input shapes."""
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from .memory_probe import validate_memory_stats

logger = logging.getLogger(__name__)
request_context = ContextVar('memory_request_id', default=None)


class ForwardMemoryGuard:
    def __init__(self, estimator, stats, reclaim, *, safety_mib=5120, timeout_s=30, allocator=None):
        self.estimator, self.stats, self.reclaim = estimator, stats, reclaim
        self.safety_mib, self.timeout_s = safety_mib, timeout_s
        self.allocator = allocator

    def cache_capacity(self):
        """Only wholly inactive segments are counted as reclaimable, never weights."""
        if self.allocator is None or not callable(getattr(self.allocator, 'memory_snapshot', None)):
            return {'reclaimable_mib':0, 'largest_mib':0, 'sampled_monotonic':time.monotonic()}
        segments = self.allocator.memory_snapshot()
        sizes = [s['total_size'] / 1024**2 for s in segments
                 if s.get('device', 0) == 0 and s.get('active_size', 1) == 0 and s.get('allocated_size', 1) == 0]
        return {'reclaimable_mib':sum(sizes), 'largest_mib':max(sizes, default=0), 'sampled_monotonic':time.monotonic()}

    def release_cached(self):
        self.estimator.cache_snapshot = {'reclaimable_mib':0, 'sampled_monotonic':float('inf')}
        self.reclaim()
        self.estimator.cache_snapshot = self.cache_capacity()

    @contextmanager
    def observe_forward(self, estimate):
        """Measure this process's allocation peaks independently of device sampling.

        The reranker owns one model worker. Remote embedding is a different
        process, so these allocator peaks are attributable to this forward.
        Vendor allocations outside the Torch allocator still need device reads.
        """
        module = self.allocator
        if module is None:
            yield
            return
        module.synchronize()
        allocated = module.memory_allocated()
        reserved = module.memory_reserved()
        module.reset_peak_memory_stats()
        try:
            yield
        finally:
            module.synchronize()
            peak_allocated = module.max_memory_allocated()
            peak_reserved = module.max_memory_reserved()
            incremental = max(0, peak_allocated-allocated, peak_reserved-reserved) / (1024**2)
            row = {'tokens': estimate['tokens'], 'batch': estimate['batch'],
                   'request_id': request_context.get(),
                   'estimated_mib': estimate['estimated_mib'],
                   'analytical_components_mib': sum(estimate.get('components', {}).values()),
                   'factor': estimate.get('factor'), 'workspace_mib': estimate.get('workspace_mib'),
                   'model_fingerprint': getattr(self.estimator, 'fingerprint', None),
                   'allocator_incremental_peak_mib': incremental,
                   'allocated_before_mib': allocated/(1024**2), 'reserved_before_mib': reserved/(1024**2),
                   'peak_allocated_mib': peak_allocated/(1024**2), 'peak_reserved_mib': peak_reserved/(1024**2),
                   'scope': 'reranker process torch allocator; vendor/device-wide allocations require separate telemetry'}
            import json
            logger.info('event=forward_memory_observation observation=%s', json.dumps(row, sort_keys=True))
            self.estimator.cache_snapshot = self.cache_capacity()
            if incremental > estimate['estimated_mib']:
                self.estimator.fault = 'Reranker allocator peak exceeded the forward estimate; stop new admission and recalibrate'

    def choose_batch(self, lengths, requested):
        """Return a fitting prefix; never allocate an input tensor to test fit."""
        if not lengths or requested < 1 or any(n < 1 for n in lengths):
            raise ValueError('Execution guard needs nonempty positive sequence lengths and batch')
        end = time.monotonic() + self.timeout_s
        if getattr(self.estimator, 'fault', None):
            raise RuntimeError('Previous forward exceeded estimate; execution is stopped before allocation')
        reclaimed = False
        while True:
            reading = validate_memory_stats(self.stats())
            if not all(k in reading for k in ('total_mb', 'used_mb', 'free_mb')):
                raise RuntimeError('Execution guard requires complete fresh device memory stats')
            batch = min(requested, len(lengths))
            while batch:
                tokens = max(lengths[:batch])
                estimate = self.estimator.forward_estimate(tokens, batch)
                cache = self.cache_capacity()
                required_growth = max(0, estimate['estimated_mib'] - cache['reclaimable_mib'])
                # If the largest tensor cannot reuse an existing free segment,
                # it must also fit contiguously in currently free device memory.
                largest_tensor = max(estimate.get('components', {}).get('output_mib', 0),
                                     estimate.get('components', {}).get('activations_mib', 0))
                reusable_largest = (largest_tensor <= cache['largest_mib']
                                    and cache['reclaimable_mib'] - cache['largest_mib'] >= estimate['estimated_mib'] - largest_tensor)
                if not reusable_largest:
                    required_growth = max(required_growth, largest_tensor + estimate.get('workspace_mib', 0))
                if reading['free_mb'] - required_growth >= self.safety_mib:
                    logger.info('event=forward_memory_plan requested_batch=%d actual_batch=%d tokens=%d estimated_mib=%d free_mib=%s cache_reclaimable_mib=%s required_growth_mib=%s',
                                requested, batch, tokens, estimate['estimated_mib'], reading['free_mb'], cache['reclaimable_mib'], required_growth)
                    return batch, estimate
                if (not reclaimed and cache['reclaimable_mib'] > 0
                        and reading['free_mb'] + cache['reclaimable_mib'] - estimate['estimated_mib'] >= self.safety_mib):
                    # A larger real tensor is blocked by cached-block shape, not
                    # device capacity. Reclaim before reducing the execution batch.
                    self.release_cached()
                    reclaimed = True
                    break
                batch //= 2
            if batch:
                continue
            if not reclaimed:
                # Only releases unused allocator cache; never allocates fake memory.
                self.release_cached()
                reclaimed = True
                continue
            if time.monotonic() >= end:
                raise RuntimeError('No safe single-item forward after cache reclaim; no device allocation attempted')
            time.sleep(.1)
