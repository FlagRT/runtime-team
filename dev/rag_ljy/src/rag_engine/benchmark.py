"""Single-flight search benchmark with raw samples and successful-only stats."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from .benchmark_data import BenchmarkQuery
from .timing import TIMING_FIELDS
from .retrieval import validate_execution_mode


def percentile(values: Sequence[float], percent: float) -> float:
    """Linear interpolation between sorted observations (no NumPy dependency)."""
    if not values or not 0 <= percent <= 100:
        raise ValueError("Percentile requires values and a percentage in 0..100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    stages = {}
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in successful if field in row]
        if values:
            stages[field] = {
                "samples": len(values), "mean": statistics.fmean(values),
                "p50": percentile(values, 50), "p95": percentile(values, 95),
                "p99": percentile(values, 99), "min": min(values), "max": max(values),
            }
    failures = len(rows) - len(successful)
    return {
        "attempted": len(rows), "successful": len(successful), "failed": failures,
        "error_rate": failures / len(rows) if rows else 0.0,
        "latency_ms": stages,
    }


def run_benchmark(
    queries: Sequence[BenchmarkQuery],
    measure_query: Callable[[BenchmarkQuery, dict[str, float]], list[dict]],
    output_dir: Path,
    *,
    warmup: int = 10,
    repeats: int = 3,
    seed: int = 42,
    metadata: dict[str, Any] | None = None,
    execution_mode: str = "sequential",
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Load models outside this function. Warm up, then measure serial requests.

    New directories only; no artifacts are overwritten. CSV is flushed per
    sample, outside the measured call. Errors remain visible but are excluded
    from successful latency percentiles. Summary is also saved on interruption.
    """
    if not queries or warmup < 0 or repeats < 1:
        raise ValueError("Require queries, non-negative warmup, and positive repeats")
    validate_execution_mode(execution_mode)
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Duplicate benchmark query IDs")
    output_dir.mkdir(parents=True, exist_ok=False)
    serialized_queries = json.dumps([asdict(query) for query in queries], ensure_ascii=False)
    info = dict(metadata or {})
    info.update({
        "execution_mode": execution_mode,
        "search_order": ["bm25", "dense"] if execution_mode == "sequential" else None,
        "search_channels": ["bm25", "dense"],
        "search_workers": 2 if execution_mode == "concurrent" else 0,
        "query_count": len(queries), "queries_sha256": hashlib.sha256(
            serialized_queries.encode("utf-8")
        ).hexdigest(),
        "warmup": warmup, "repeats": repeats, "seed": seed,
        "percentile_method": "linear interpolation",
        "timing_clock": "time.perf_counter_ns",
        "search_ms_definition": (
            "before submitting both searches until both complete; excludes RRF"
            if execution_mode == "concurrent" else
            "before BM25 starts until dense returns; excludes RRF"
        ),
        "full_query_total_definition": "embedding through reranker return; excludes startup, HTTP and artifact I/O",
        "retrieval_only_total_definition": "search + RRF; excludes precomputed embeddings",
    })
    (output_dir / "queries.jsonl").write_text(
        "".join(json.dumps(asdict(query), ensure_ascii=False) + "\n" for query in queries),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    state = "warming_up"
    summary = {"status": state, "metadata": info, **summarize_rows(rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    try:
        for index in range(warmup):
            measure_query(queries[index % len(queries)], {})
            progress(f"Warmup {index + 1}/{warmup} (not recorded)")
        state = "running"
        fields = ["repeat", "position", "query_id", "status", "error_type", "result_count", *TIMING_FIELDS]
        with (output_dir / "raw.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for repeat in range(repeats):
                order = list(queries)
                random.Random(seed + repeat).shuffle(order)
                for position, query in enumerate(order, 1):
                    timings: dict[str, float] = {}
                    row = {
                        "repeat": repeat + 1, "position": position,
                        "query_id": query.query_id, "status": "ok", "error_type": "",
                        "result_count": 0,
                    }
                    start = perf_counter_ns()
                    try:
                        results = measure_query(query, timings)
                        row["result_count"] = len(results)
                    except Exception as error:
                        row["status"] = "error"
                        # Avoid writing credentials/connection details from exception text.
                        row["error_type"] = type(error).__name__
                        progress(f"Query {query.query_id} failed: {type(error).__name__}")
                    finally:
                        timings.setdefault("total_ms", (perf_counter_ns() - start) / 1_000_000)
                    row.update({field: timings[field] for field in TIMING_FIELDS if field in timings})
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    if position % 10 == 0 or position == len(order):
                        progress(f"Repeat {repeat + 1}/{repeats}: {position}/{len(order)} queries")
        state = "complete"
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    except Exception:
        state = "aborted"
        raise
    finally:
        summary = {"status": state, "metadata": info, **summarize_rows(rows)}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary
