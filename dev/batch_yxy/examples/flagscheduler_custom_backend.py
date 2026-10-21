"""Custom Python backend example: flagscheduler.backends factory contract.

Use backend='python:flagscheduler_custom_backend.ExampleBackend' and add this
examples/ directory to PYTHONPATH. This demo is not a real inference adapter.
"""
from flagscheduler.contracts import ExecutionOutput, positive_int


class ExampleBackend:
    def __init__(self, binding):
        self.options = binding.options

    def count_cost(self, payload):
        return positive_int(payload["sample_count"], "sample_count")

    def execute(self, inputs):
        return [ExecutionOutput(item.request_id, {"example": True, "score": 0.5}) for item in inputs]

    def close(self):
        pass
