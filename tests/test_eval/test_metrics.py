"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest

from dojo.eval.metrics import (
    EvalSummary,
    MetricsCalculator,
    PrimaryMetrics,
    TransferScore,
)


def test_primary_metrics_mean() -> None:
    m = PrimaryMetrics(
        elicitation_quality=0.8,
        sufficiency_calibration=0.6,
        decomposition_accuracy=0.7,
        intent_match=0.9,
    )
    assert 0.0 < m.mean() < 1.0


def test_primary_metrics_to_dict() -> None:
    m = PrimaryMetrics(elicitation_quality=0.8)
    d = m.to_dict()
    assert "elicitation_quality" in d
    assert d["elicitation_quality"] == 0.8


def test_transfer_score_ratio() -> None:
    ts = TransferScore(behavior_code="B-01", team_score=0.8, solo_score=0.7)
    assert abs(ts.ratio - 0.875) < 0.01
    assert ts.transfers_well


def test_transfer_score_poor() -> None:
    ts = TransferScore(behavior_code="B-01", team_score=0.8, solo_score=0.3)
    assert not ts.transfers_well


def test_transfer_score_zero_team() -> None:
    ts = TransferScore(behavior_code="B-01", team_score=0.0, solo_score=0.5)
    assert ts.ratio == 0.0


def test_compute_primary_empty() -> None:
    calc = MetricsCalculator()
    result = calc.compute_primary([])
    assert result.elicitation_quality == 0.0


def test_compute_primary() -> None:
    calc = MetricsCalculator()
    results = [
        {
            "phase_scores": {"elicitation": 0.8, "planning": 0.7, "verification": 0.9},
            "phases": [{"phase": "elicitation", "actions": [
                {"type": "ask_question"}, {"type": "ask_question"}, {"type": "ask_question"},
            ]}],
        },
        {
            "phase_scores": {"elicitation": 0.6, "planning": 0.5, "verification": 0.7},
            "phases": [{"phase": "elicitation", "actions": [
                {"type": "ask_question"}, {"type": "ask_question"},
            ]}],
        },
    ]
    primary = calc.compute_primary(results)
    assert primary.elicitation_quality == 0.7  # (0.8+0.6)/2
    assert primary.decomposition_accuracy == 0.6  # (0.7+0.5)/2


def test_compute_transfer_scores() -> None:
    calc = MetricsCalculator()
    team = {"B-01": 0.8, "B-02": 0.6, "B-05": 0.9}
    solo = {"B-01": 0.7, "B-02": 0.5, "B-05": 0.8}
    scores = calc.compute_transfer_scores(team, solo)
    assert len(scores) == 3
    b01 = next(s for s in scores if s.behavior_code == "B-01")
    assert abs(b01.ratio - 0.875) < 0.01


def test_summarize() -> None:
    calc = MetricsCalculator()
    results = [
        {
            "task_id": "t1",
            "phase_scores": {"elicitation": 0.8},
            "overall_score": 0.75,
            "phases": [],
        }
    ]
    summary = calc.summarize(results)
    assert summary.num_tasks == 1
    assert "t1" in summary.per_task_scores


def test_eval_summary_mean_transfer() -> None:
    summary = EvalSummary(
        num_tasks=10,
        primary=PrimaryMetrics(),
        secondary=None,
        transfer_scores=[
            TransferScore("B-01", 1.0, 0.9),
            TransferScore("B-02", 1.0, 0.7),
        ],
    )
    assert abs(summary.mean_transfer - 0.8) < 0.01
    assert summary.well_transferring_count == 1
