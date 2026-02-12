"""Tests for evaluation runner."""
from __future__ import annotations

import pytest

from dojo.eval.runner import EvalRunner, EvalConfig
from dojo.eval.solo_env import SoloTask


@pytest.fixture
def task_bank() -> list:
    return [
        SoloTask(
            task_id=f"task-{i}",
            description=f"Implement feature {i}",
            ambiguity="medium",
            domain="api",
            complexity="medium",
            codebase="greenfield",
            expected_questions=[f"Question {j}" for j in range(3)],
        )
        for i in range(20)
    ]


def test_quick_eval_no_tasks() -> None:
    runner = EvalRunner(config=EvalConfig(quick_num_tasks=5))
    summary = runner.quick_eval()
    assert summary.num_tasks == 5


def test_quick_eval_with_task_bank(task_bank) -> None:
    runner = EvalRunner(task_bank=task_bank, config=EvalConfig(quick_num_tasks=10))
    summary = runner.quick_eval()
    assert summary.num_tasks == 10
    assert len(summary.per_task_scores) == 10


def test_full_eval(task_bank) -> None:
    runner = EvalRunner(task_bank=task_bank, config=EvalConfig(full_num_tasks=15))
    summary = runner.full_eval()
    assert summary.num_tasks == 15


def test_comprehensive_eval_with_transfer(task_bank) -> None:
    runner = EvalRunner(task_bank=task_bank, config=EvalConfig(comprehensive_num_tasks=10))
    team_scores = {"B-01": 0.8, "B-02": 0.7}
    solo_scores = {"B-01": 0.7, "B-02": 0.6}
    summary = runner.comprehensive_eval(
        team_scores=team_scores,
        solo_behavior_scores=solo_scores,
    )
    assert len(summary.transfer_scores) == 2
    assert summary.mean_transfer > 0


def test_eval_history(task_bank) -> None:
    runner = EvalRunner(task_bank=task_bank, config=EvalConfig(quick_num_tasks=5))
    runner.quick_eval()
    runner.quick_eval()
    assert len(runner.history) == 2


def test_primary_metrics_populated(task_bank) -> None:
    runner = EvalRunner(task_bank=task_bank, config=EvalConfig(quick_num_tasks=5))
    summary = runner.quick_eval()
    primary = summary.primary
    assert primary.elicitation_quality >= 0.0
    assert primary.decomposition_accuracy >= 0.0
