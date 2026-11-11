"""Memory pressure, concurrent budgets and delayed-observation accounting."""

from types import SimpleNamespace

import pytest

from rag_engine.admission import MemoryAdmissionConfig, MemoryAdmissionController


def reading(used=300, *, started=100, **updates):
    sample = {"status": "ok", "unit": "MiB", "scope": "device", "age_seconds": 0,
              "sample_started_monotonic": started, "stale": False,
              "stats": {"total_mb": 1000, "used_mb": used, "free_mb": 1000 - used}}
    sample.update(updates)
    return sample


def make_controller(**options):
    clock = SimpleNamespace(now=100)
    config = MemoryAdmissionConfig(**{
        "enabled": True, "request_budget_mib": 200, "safety_margin_mib": 150,
        "high_watermark": .85, "low_watermark": .75, **options})
    return MemoryAdmissionController(config, clock=lambda: clock.now), clock


@pytest.mark.parametrize("options", [
    {"enabled": True}, {"request_budget_mib": 0}, {"request_budget_mib": -1},
    {"request_budget_mib": float("nan")}, {"request_budget_mib": True},
    {"safety_margin_mib": -1}, {"safety_margin_mib": float("inf")},
    {"high_watermark": 1}, {"high_watermark": float("nan")},
    {"high_watermark": .7, "low_watermark": .7}, {"low_watermark": 0},
    {"low_watermark": .9}, {"max_sample_age_s": 0}, {"max_sample_age_s": float("inf")},
])
def test_invalid_admission_configuration(options):
    with pytest.raises(ValueError):
        MemoryAdmissionConfig(**options)


def test_same_sample_cannot_be_spent_by_unlimited_requests():
    admission, _ = make_controller()
    sample = reading()
    assert admission.reserve("one", sample)["accepted"]
    assert admission.reserve("two", sample)["accepted"]
    rejected = admission.reserve("three", sample)
    assert not rejected["accepted"] and rejected["reason"] == "request_budget"
    state = admission.snapshot(sample)
    assert state["reserved_requests"] == 2 and state["reserved_mib"] == 400
    assert state["accepted"] == 2 and state["rejected"] == 1


def test_pressure_requires_low_watermark_before_recovery():
    admission, _ = make_controller(request_budget_mib=50)
    assert admission.reserve("high", reading(900))["reason"] == "memory_pressure"
    # Dropping below the high mark alone does not reopen admission.
    assert admission.reserve("middle", reading(800))["reason"] == "memory_pressure"
    assert admission.reserve("low", reading(750))["accepted"]
    assert admission.snapshot(reading(750))["reserved_mib"] == 50


def test_reservations_do_not_trigger_or_latch_watermark():
    admission, _ = make_controller(request_budget_mib=100, safety_margin_mib=50)
    assert admission.reserve("fits", reading(840))["accepted"]
    state = admission.snapshot(reading(840))
    assert state["effective_used_percent"] == 94
    assert state["measured_used_percent"] == 84
    assert state["reason"] == "request_budget"
    assert admission.reserve("budget-short", reading(840))["reason"] == "request_budget"
    assert admission.reserve("free-recovered", reading(700))["accepted"]


def test_exact_high_watermark_enters_pressure_pause():
    admission, _ = make_controller(request_budget_mib=50, safety_margin_mib=50)
    assert admission.reserve("exactly-85-percent", reading(850))["reason"] == "memory_pressure"
    assert admission.reserve("middle", reading(849))["reason"] == "memory_pressure"
    assert admission.reserve("low", reading(750))["accepted"]


def test_recovery_uses_measured_usage_even_with_reservations():
    admission, _ = make_controller(request_budget_mib=100, safety_margin_mib=50)
    assert admission.reserve("active", reading(700))["accepted"]
    assert admission.snapshot(reading(850))["reason"] == "memory_pressure"
    assert admission.snapshot(reading(751))["reason"] == "memory_pressure"
    state = admission.snapshot(reading(750))
    assert state["reserved_mib"] == 100 and state["status"] == "ready"
    assert admission.reserve("new", reading(750))["accepted"]


def test_default_five_gib_margin_boundary():
    a = MemoryAdmissionController(MemoryAdmissionConfig(enabled=True, estimate_requests=True), clock=lambda: 100)
    sample = reading(stats={"total_mb": 62748, "used_mb": 40000, "free_mb": 22748})
    assert a.reserve("one-too-many", sample, request_budget_mib=17629)["reason"] == "request_budget"
    assert a.reserve("exact-five-gib-left", sample, request_budget_mib=17628)["accepted"]


def test_exact_safety_margin_is_accepted():
    admission, _ = make_controller(request_budget_mib=100, safety_margin_mib=60)
    assert admission.reserve("exact-margin", reading(840))["accepted"]


def test_margin_rejection_does_not_latch_until_low_watermark():
    admission, _ = make_controller(request_budget_mib=100, safety_margin_mib=100)
    assert admission.reserve("insufficient-margin", reading(810))["reason"] == "request_budget"
    # Recovery of free memory is sufficient; no low-watermark latch was set.
    assert admission.reserve("fits-without-low-watermark", reading(800))["accepted"]


def test_free_memory_margin_is_enforced_separately_from_watermark():
    admission, _ = make_controller(request_budget_mib=500, safety_margin_mib=350)
    result = admission.reserve("large", reading(200))
    assert not result["accepted"] and result["reason"] == "request_budget"
    assert result["reserved_mib"] == 0


def test_completed_budget_waits_for_read_started_after_completion():
    admission, clock = make_controller(request_budget_mib=300)
    sample = reading()
    assert admission.reserve("one", sample)["accepted"]
    clock.now = 101
    admission.release("one")
    admission.release("one")  # Duplicate callback must not duplicate accounting.
    state = admission.snapshot(sample)
    assert state["reserved_requests"] == 0 and state["pending_release_mib"] == 300
    assert not admission.reserve("two", sample)["accepted"]
    clock.now = 102
    # A read that finished later but started BEFORE completion is insufficient.
    assert admission.snapshot(reading(started=100.5))["pending_release_mib"] == 300
    assert admission.snapshot(reading(started=101))["pending_release_mib"] == 300
    assert admission.reserve("fresh", reading(started=102))["accepted"]
    assert admission.snapshot(reading(started=102))["pending_release_mib"] == 0


def test_live_request_reservation_survives_all_new_samples():
    admission, clock = make_controller()
    assert admission.reserve("waiting-for-rerank", reading())["accepted"]
    clock.now = 101
    state = admission.snapshot(reading(500, started=101))
    # Measured allocations do not silently replace the full downstream budget.
    assert state["active_reserved_mib"] == 200
    assert state["effective_used_percent"] == 70
    assert not admission.reserve("extra", reading(500, started=101))["accepted"]


@pytest.mark.parametrize("sample, reason", [
    ({"status": "pending"}, "telemetry_unavailable"),
    ({"status": "unavailable"}, "telemetry_unavailable"),
    (reading(status="partial"), "telemetry_unavailable"),
    (reading(stale=True), "stale_telemetry"),
    (reading(age_seconds=4), "stale_telemetry"),
    (reading(started=96), "stale_telemetry"),
    (reading(started=101), "invalid_telemetry"),
    (reading(age_seconds=float("nan")), "invalid_telemetry"),
    (reading(age_seconds=True), "invalid_telemetry"),
    (reading(stats={"total_mb": 1000, "used_mb": 300}), "invalid_telemetry"),
    (reading(stats={"total_mb": 1000, "used_mb": 300, "free_mb": 900}), "invalid_telemetry"),
    (reading(scope="process"), "unverified_memory_scope"),
    (reading(scope="backend_reported_unverified"), "unverified_memory_scope"),
    (reading(unit="bytes"), "invalid_memory_unit"),
])
def test_unusable_readings_reject_without_allocating_budget(sample, reason):
    admission, _ = make_controller()
    decision = admission.reserve("bad-reading", sample)
    assert not decision["accepted"] and decision["reason"] == reason
    assert decision["reserved_mib"] == 0
    assert admission.reserve("recovered", reading())["accepted"]


def test_unverified_scope_requires_explicit_configuration():
    admission, _ = make_controller(allow_unverified_scope=True)
    assert admission.reserve("verified-externally", reading(scope="backend_reported_unverified"))["accepted"]
    assert not admission.reserve("process-only", reading(scope="process"))["accepted"]


def test_failed_read_cannot_retire_completed_budget():
    admission, clock = make_controller()
    assert admission.reserve("one", reading())["accepted"]
    clock.now = 101
    admission.release("one")
    clock.now = 102
    state = admission.snapshot(reading(started=102, status="unavailable"))
    assert state["pending_release_mib"] == 200
    assert admission.snapshot(reading(started=102))["pending_release_mib"] == 0


def test_duplicate_request_id_cannot_replace_a_reservation():
    admission, _ = make_controller()
    assert admission.reserve("same", reading())["accepted"]
    with pytest.raises(ValueError, match="already owns"):
        admission.reserve("same", reading())


def test_disabled_controller_needs_no_telemetry_or_memory_budget():
    admission = MemoryAdmissionController(MemoryAdmissionConfig())
    assert admission.snapshot(None) == {"enabled": False, "status": "disabled"}
    assert admission.reserve("one", None)["accepted"]
    admission.release("one")
    assert admission.snapshot(None) == {"enabled": False, "status": "disabled"}


def test_dynamic_budgets_are_independent_and_preserved_until_fresh_sample():
    clock = SimpleNamespace(now=100)
    a = MemoryAdmissionController(MemoryAdmissionConfig(enabled=True, estimate_requests=True,
        safety_margin_mib=100), clock=lambda: clock.now)
    assert a.reserve('small', reading(), request_budget_mib=100)['accepted']
    assert a.reserve('large', reading(), request_budget_mib=400)['accepted']
    r = a.reserve('too-large', reading(), request_budget_mib=101)
    assert not r['accepted'] and r['reason'] == 'request_budget'
    assert r['request_budget_mib'] == 101
    a.release('large')
    assert a.snapshot(reading())['pending_release_mib'] == 400
    clock.now = 101
    assert a.snapshot(reading(started=101))['reserved_mib'] == 100


@pytest.mark.parametrize('budget', [None, 0, -1, True, float('nan'), float('inf')])
def test_dynamic_missing_invalid_estimates_never_fall_back_to_fixed_budget(budget):
    a = MemoryAdmissionController(MemoryAdmissionConfig(enabled=True, estimate_requests=True))
    r = a.reserve('bad', {}, request_budget_mib=budget)
    assert not r['accepted'] and r['reason'] == 'invalid_request_estimate'
    assert r['reserved_mib'] == 0


def test_unavailable_estimator_is_audited_without_reservation():
    a, _ = make_controller()
    d = a.reject_unestimated('no-estimate', 'memory_estimate_unavailable')
    assert d['accepted'] is False and d['reserved_mib'] == 0
    assert a.rejected == 1 and a.last_decision == d


def test_cache_credit_cannot_be_spent_twice_and_reclaim_promotes_reservation():
    a, _ = make_controller(request_budget_mib=100, safety_margin_mib=150)
    assert a.reserve('credited', reading(600), cache_credit_mib=200)['accepted']
    assert a.reserve('double-credit', reading(600), cache_credit_mib=200)['reason'] == 'invalid_cache_credit'
    before = a.snapshot(reading(600))['available_after_reservations_mib']
    after = a.snapshot(reading(400))
    assert after['reserved_mib'] == 300
    assert after['available_after_reservations_mib'] == before
    assert a.reserve('overspend', reading(400), request_budget_mib=151)['reason'] == 'request_budget'
    assert a.snapshot(reading(500))['reserved_mib'] == 300  # never undo promotion
    a.release('credited')
    assert a.snapshot(reading(400))['pending_release_mib'] == 300
