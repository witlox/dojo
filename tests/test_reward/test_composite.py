"""Tests for composite reward calculator."""
from __future__ import annotations

import pytest

from dojo.reward.composite_reward import (
    CompositeReward,
    CompositeRewardCalculator,
    StageWeights,
)


def test_stage_weights_for_stage_1() -> None:
    w = StageWeights.for_stage(1)
    assert w.outcome == 0.30
    assert w.behavioral == 0.70


def test_stage_weights_for_stage_4() -> None:
    w = StageWeights.for_stage(4)
    assert w.outcome == 0.60
    assert w.behavioral == 0.40


def test_stage_weights_default_for_unknown() -> None:
    w = StageWeights.for_stage(99)
    assert w.outcome == 0.30  # Defaults to stage 1


def test_compute_basic() -> None:
    calc = CompositeRewardCalculator(stage=1)
    reward = calc.compute(
        outcome_reward=0.8,
        behavioral_heuristic=0.6,
    )
    assert isinstance(reward, CompositeReward)
    assert 0.0 <= reward.total <= 1.0
    assert reward.outcome == 0.8
    assert reward.behavioral_heuristic == 0.6
    assert reward.behavioral_judge is None


def test_compute_uses_judge_when_available() -> None:
    calc = CompositeRewardCalculator(stage=1)
    reward = calc.compute(
        outcome_reward=0.5,
        behavioral_heuristic=0.3,
        behavioral_judge=0.9,
    )
    # Stage 1: behavioral weight 0.7, so judge score 0.9 should dominate
    assert reward.behavioral_judge == 0.9
    # total = 0.3*0.5 + 0.7*0.9 = 0.15 + 0.63 = 0.78
    assert abs(reward.total - 0.78) < 0.05


def test_compute_efficiency_penalty() -> None:
    calc = CompositeRewardCalculator(
        stage=1,
        baseline_actions={"elicitation": 10.0},
    )
    reward = calc.compute(
        outcome_reward=0.5,
        behavioral_heuristic=0.5,
        actions_taken=20,
        episode_type="elicitation",
    )
    assert reward.efficiency_penalty < 0


def test_compute_no_efficiency_penalty_below_baseline() -> None:
    calc = CompositeRewardCalculator(
        stage=1,
        baseline_actions={"elicitation": 10.0},
    )
    reward = calc.compute(
        outcome_reward=0.5,
        behavioral_heuristic=0.5,
        actions_taken=5,
        episode_type="elicitation",
    )
    assert reward.efficiency_penalty == 0.0


def test_compute_phase_bonus() -> None:
    calc = CompositeRewardCalculator(stage=1)
    reward = calc.compute(
        outcome_reward=0.5,
        behavioral_heuristic=0.5,
        phase_transitions=["elicitation_to_planning", "checkpoint_pivot"],
    )
    assert reward.phase_bonus > 0


def test_compute_clamps_to_unit() -> None:
    calc = CompositeRewardCalculator(stage=1)
    reward = calc.compute(
        outcome_reward=1.0,
        behavioral_heuristic=1.0,
        phase_transitions=["elicitation_to_planning", "checkpoint_pivot"],
    )
    assert reward.total <= 1.0

    reward2 = calc.compute(
        outcome_reward=0.0,
        behavioral_heuristic=0.0,
        actions_taken=100,
        episode_type="elicitation",
    )
    assert reward2.total >= 0.0


def test_set_stage_updates_weights() -> None:
    calc = CompositeRewardCalculator(stage=1)
    assert calc.weights.behavioral == 0.70
    calc.set_stage(4)
    assert calc.weights.behavioral == 0.40


def test_all_13_episode_types_produce_valid_rewards() -> None:
    """Ensure rewards are computable for all episode types."""
    episode_types = [
        "elicitation", "decomposition", "implementation", "self_monitoring",
        "research", "triage", "recovery", "scope_change",
        "borrowing_arrival", "cross_team_dependency",
        "knowledge_handoff", "onboarding_support", "compensation",
    ]
    for stage in range(1, 5):
        calc = CompositeRewardCalculator(stage=stage)
        for ep_type in episode_types:
            reward = calc.compute(
                outcome_reward=0.5,
                behavioral_heuristic=0.5,
                episode_type=ep_type,
            )
            assert 0.0 <= reward.total <= 1.0, f"Invalid reward for {ep_type} stage {stage}"
