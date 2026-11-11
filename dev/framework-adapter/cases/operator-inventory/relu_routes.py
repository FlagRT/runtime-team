"""Exact routing oracle for DeepFM: replace three ReLUs, keep sigmoid native."""
from collections import Counter


def check_model_routes(routes,mode):
    actual=Counter({tuple(x[:-1]):x[-1] for x in routes})
    if mode in ('native','native_repeat'):
        expected=Counter()
    else:
        expected=Counter({('sigmoid','native','forward','operator_disabled'):1})
        if mode in ('candidate','on'):
            expected.update({('relu','flaggems','forward','completed'):3,
                             ('relu','flaggems','backward','completed'):3})
        else:
            reason={'off':'experimental_disabled','unknown':'workload_unverified'}[mode]
            expected[('relu','native','forward',reason)]=3
    if actual!=expected:raise AssertionError(f'Unexpected model routes: {actual}; expected {expected}')
