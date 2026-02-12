"""Tests for calibration monitor."""
from __future__ import annotations

import pytest

from dojo.reward.calibration import CalibrationMonitor


def test_record_aligned() -> None:
    monitor = CalibrationMonitor()
    entry = monitor.record("ep1", "elicitation", outcome_score=0.7, heuristic_score=0.6)
    assert entry.flag == "aligned"


def test_record_performative() -> None:
    monitor = CalibrationMonitor()
    entry = monitor.record("ep1", "elicitation", outcome_score=0.2, heuristic_score=0.9)
    assert entry.flag == "performative"


def test_record_unconventional() -> None:
    monitor = CalibrationMonitor()
    entry = monitor.record("ep1", "elicitation", outcome_score=0.9, heuristic_score=0.2)
    assert entry.flag == "unconventional"


def test_record_with_judge_score() -> None:
    monitor = CalibrationMonitor()
    # Judge score overrides heuristic for flag computation
    entry = monitor.record(
        "ep1", "elicitation",
        outcome_score=0.2,
        heuristic_score=0.4,
        judge_score=0.9,
    )
    assert entry.flag == "performative"  # judge=0.9, outcome=0.2 -> high diff


def test_report_empty() -> None:
    monitor = CalibrationMonitor()
    report = monitor.report()
    assert report.total_entries == 0
    assert report.aligned_rate == 0.0


def test_report_with_entries() -> None:
    monitor = CalibrationMonitor()
    monitor.record("ep1", "elicitation", 0.7, 0.6)
    monitor.record("ep2", "elicitation", 0.5, 0.5)
    monitor.record("ep3", "decomposition", 0.2, 0.8)

    report = monitor.report()
    assert report.total_entries == 3
    assert report.aligned_rate > 0
    assert "elicitation" in report.per_episode_type
    assert "decomposition" in report.per_episode_type


def test_get_flagged_entries() -> None:
    monitor = CalibrationMonitor()
    monitor.record("ep1", "elicitation", 0.7, 0.6)  # aligned
    monitor.record("ep2", "elicitation", 0.2, 0.9)  # performative
    monitor.record("ep3", "elicitation", 0.9, 0.2)  # unconventional

    performative = monitor.get_flagged_entries("performative")
    assert len(performative) == 1
    assert performative[0].episode_id == "ep2"


def test_correlation_requires_minimum_entries() -> None:
    monitor = CalibrationMonitor()
    monitor.record("ep1", "e", 0.5, 0.5)
    report = monitor.report()
    assert report.outcome_behavioral_correlation is None

    monitor.record("ep2", "e", 0.6, 0.6)
    monitor.record("ep3", "e", 0.7, 0.7)
    report = monitor.report()
    assert report.outcome_behavioral_correlation is not None


def test_correlation_positive() -> None:
    monitor = CalibrationMonitor()
    # Perfect positive correlation
    for i in range(10):
        v = i / 10.0
        monitor.record(f"ep{i}", "e", v, v)

    report = monitor.report()
    assert report.outcome_behavioral_correlation is not None
    assert report.outcome_behavioral_correlation > 0.9


def test_heuristic_judge_correlation() -> None:
    monitor = CalibrationMonitor()
    for i in range(10):
        v = i / 10.0
        monitor.record(f"ep{i}", "e", 0.5, v, judge_score=v * 0.9)

    report = monitor.report()
    assert report.heuristic_judge_correlation is not None
    assert report.heuristic_judge_correlation > 0.9


def test_clear() -> None:
    monitor = CalibrationMonitor()
    monitor.record("ep1", "e", 0.5, 0.5)
    assert len(monitor._entries) == 1
    monitor.clear()
    assert len(monitor._entries) == 0
