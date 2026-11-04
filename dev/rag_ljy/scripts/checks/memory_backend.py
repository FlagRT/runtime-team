"""Validate the actual backend reported by the API, never just a requested label."""
def require_backend(health, expected):
    actual = health.get('retrieval_backend')
    if actual != expected:
        raise RuntimeError(f'Expected retrieval backend {expected}, got {actual}')
    return actual
