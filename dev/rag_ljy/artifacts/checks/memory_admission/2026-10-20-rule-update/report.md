# Memory admission rule update — 2026-10-20

Changed the pressure latch to strictly `used + outstanding reservations > high × total`.
It clears at or below the low watermark. While not latched, admission requires only
`free - outstanding reservations - new budget >= safety margin`, alongside existing
telemetry validation and request-count limits. New budgets are no longer compared
against the high watermark. Default thresholds remain 85%/75%, margin 1024 MiB.

Validation: full suite **173 passed**, one Starlette/httpx deprecation warning, 2.30 s.
Command (repository root):
```
/tmp/rag-ljy-memory-admission-20261020/bin/python -m pytest dev/rag_ljy/tests -q -o faulthandler_timeout=20
```
Added coverage for crossing the watermark with a fitting request, subsequent pause
and recovery, exact 85% not triggering pause, exact safety margin acceptance, and
budget rejection without a pressure latch. Existing budget/concurrency fixtures now
use a 150 MiB margin so insufficient free capacity, rather than projected watermark,
is the reason for rejection.

These are controlled-memory and fake-model tests, not live NPU pressure tests.
No services were restarted. The running query API needs a restart to load the change.
Earlier reports remain historical evidence for the previous rule.
