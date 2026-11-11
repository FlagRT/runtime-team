import csv
import json

import pytest

from rag_engine.benchmark import percentile, run_benchmark
from rag_engine.benchmark_data import BenchmarkQuery, load_queries, select_qrels_queries


QUERIES = [BenchmarkQuery("q1", "first query"), BenchmarkQuery("q2", "second query")]


def measure(_query, timings):
    timings.update(bm25_ms=40.0, dense_ms=60.0, search_ms=100.0, rrf_ms=2.0, total_ms=102.0)
    return [{}]


def test_benchmark_excludes_warmup_and_keeps_model_owner_reusable(tmp_path):
    calls = []

    def reused_measure(query, timings):
        calls.append(query.query_id)
        return measure(query, timings)

    output = tmp_path / "run"
    summary = run_benchmark(
        QUERIES, reused_measure, output, warmup=3, repeats=2,
        metadata={"scope": "retrieval-only"}, progress=lambda _: None,
    )
    assert len(calls) == 3 + 2 * 2
    assert summary["status"] == "complete"
    assert summary["attempted"] == summary["successful"] == 4
    assert summary["failed"] == 0
    assert summary["metadata"]["search_order"] == ["bm25", "dense"]
    assert summary["metadata"]["execution_mode"] == "sequential"
    assert summary["latency_ms"]["search_ms"]["mean"] == 100.0
    assert "embedding_ms" not in summary["latency_ms"]
    assert json.loads((output / "summary.json").read_text()) == summary
    with (output / "raw.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["embedding_ms"] == "" and row["rerank_ms"] == "" for row in rows)
    assert load_queries(output / "queries.jsonl") == QUERIES


def test_failures_are_recorded_but_excluded_from_successful_percentiles(tmp_path):
    def sometimes_fails(query, timings):
        if query.query_id == "q2":
            timings.update(search_ms=999.0)
            raise RuntimeError("SECRET password in connection details")
        return measure(query, timings)

    output = tmp_path / "run"
    summary = run_benchmark(
        QUERIES, sometimes_fails, output, warmup=0, repeats=1, progress=lambda _: None,
    )
    assert summary["successful"] == summary["failed"] == 1
    assert summary["error_rate"] == 0.5
    assert summary["latency_ms"]["search_ms"]["mean"] == 100.0
    assert summary["latency_ms"]["search_ms"]["samples"] == 1
    csv_text = (output / "raw.csv").read_text()
    assert "RuntimeError" in csv_text
    assert "SECRET" not in csv_text


def test_benchmark_never_overwrites_existing_run(tmp_path):
    output = tmp_path / "run"
    run_benchmark(QUERIES, measure, output, warmup=0, repeats=1, progress=lambda _: None)
    original = (output / "raw.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(QUERIES, measure, output, warmup=0, repeats=1, progress=lambda _: None)
    assert (output / "raw.csv").read_bytes() == original


def test_query_order_and_fingerprint_are_reproducible(tmp_path):
    summaries = []
    orders = []
    for name in ("a", "b"):
        order = []

        def record(query, timings):
            order.append(query.query_id)
            return measure(query, timings)

        summaries.append(run_benchmark(
            QUERIES, record, tmp_path / name, warmup=0, repeats=3,
            seed=17, progress=lambda _: None,
        ))
        orders.append(order)
    assert orders[0] == orders[1]
    assert summaries[0]["metadata"]["queries_sha256"] == summaries[1]["metadata"]["queries_sha256"]


def test_concurrent_benchmark_is_not_mislabeled_as_sequential(tmp_path):
    summary = run_benchmark(
        QUERIES, measure, tmp_path / "run", warmup=0, repeats=1,
        execution_mode="concurrent", progress=lambda _: None,
    )
    metadata = summary["metadata"]
    assert metadata["execution_mode"] == "concurrent"
    assert metadata["search_order"] is None
    assert metadata["search_workers"] == 2
    assert "both complete" in metadata["search_ms_definition"]


def test_interrupt_saves_completed_samples_and_partial_summary(tmp_path):
    calls = 0

    def interrupt(query, timings):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return measure(query, timings)

    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(QUERIES, interrupt, output, warmup=0, repeats=1, progress=lambda _: None)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert summary["successful"] == 1


def test_warmup_failure_aborts_instead_of_claiming_a_completed_benchmark(tmp_path):
    def fail(*_args):
        raise RuntimeError("warmup failed")

    output = tmp_path / "run"
    with pytest.raises(RuntimeError):
        run_benchmark(QUERIES, fail, output, warmup=1, progress=lambda _: None)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "aborted"
    assert summary["attempted"] == 0


def test_percentiles_are_documented_linear_interpolation():
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([4, 1, 3, 2], 95) == pytest.approx(3.85)
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 95)


def test_query_loader_accepts_prepared_and_beir_records(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"q1","query":"first query"}\n'
        '{"_id":"q2","text":"second query"}\n', encoding="utf-8",
    )
    assert load_queries(path) == QUERIES


@pytest.mark.parametrize("text", [
    '[]\n',
    '{"query_id":"q1","query":" "}\n',
    '{"query":"missing ID"}\n',
    '{"query_id":"q1","query":"x"}\n' * 2,
])
def test_invalid_query_records_are_rejected(tmp_path, text):
    path = tmp_path / "queries.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(path)


def test_qrels_selects_only_split_queries_and_preserves_ids(tmp_path):
    qrels = tmp_path / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq2\tMED-10\t2\n", encoding="utf-8")
    assert select_qrels_queries(QUERIES, qrels) == [QUERIES[1]]
    qrels.write_text("query-id\tcorpus-id\tscore\nmissing\tMED-10\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        select_qrels_queries(QUERIES, qrels)
