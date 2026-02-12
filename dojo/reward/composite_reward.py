"""Composite reward calculator combining AAT signals with dojo's judge evaluator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageWeights:
    """Outcome/behavioral weight schedule per curriculum stage."""

    outcome: float
    behavioral: float
    efficiency_penalty_weight: float = 0.1
    phase_bonus_weight: float = 0.1

    @staticmethod
    def for_stage(stage: int) -> "StageWeights":
        """Get weight schedule for a curriculum stage.

        Stage 1: behavioral-heavy (learning patterns)
        Stage 4: outcome-heavy (patterns must produce results)
        """
        schedules = {
            1: StageWeights(outcome=0.30, behavioral=0.70),
            2: StageWeights(outcome=0.40, behavioral=0.60),
            3: StageWeights(outcome=0.50, behavioral=0.50),
            4: StageWeights(outcome=0.60, behavioral=0.40),
        }
        return schedules.get(stage, schedules[1])


@dataclass
class CompositeReward:
    """Full reward breakdown from composite calculation."""

    total: float
    outcome: float
    behavioral_heuristic: float
    behavioral_judge: Optional[float]
    efficiency_penalty: float
    phase_bonus: float
    components: Dict[str, float] = field(default_factory=dict)


# Phase transition bonuses
PHASE_BONUSES: Dict[str, float] = {
    "elicitation_to_planning": 0.10,
    "planning_to_execution": 0.10,
    "checkpoint_continue": 0.05,
    "checkpoint_pivot": 0.15,
    "checkpoint_escalate": 0.10,
}


class CompositeRewardCalculator:
    """Combines all reward signals into a single composite reward.

    Sources:
    - AAT's RewardCalculator: outcome signals (velocity, coverage, completion)
    - AAT's BehavioralScorer: fast heuristic behavioral signals
    - Dojo's JudgeEvaluator: Claude Opus behavioral quality (periodic)
    - Efficiency penalty: penalizes unnecessary actions
    - Phase bonus: sparse bonus for correct phase transitions
    """

    def __init__(
        self,
        stage: int = 1,
        baseline_actions: Optional[Dict[str, float]] = None,
    ) -> None:
        self.stage = stage
        self.weights = StageWeights.for_stage(stage)
        self.baseline_actions = baseline_actions or {}

    def set_stage(self, stage: int) -> None:
        """Update the curriculum stage and weight schedule."""
        self.stage = stage
        self.weights = StageWeights.for_stage(stage)

    def compute(
        self,
        outcome_reward: float,
        behavioral_heuristic: float,
        behavioral_judge: Optional[float] = None,
        actions_taken: int = 0,
        episode_type: str = "",
        phase_transitions: Optional[List[str]] = None,
        components: Optional[Dict[str, float]] = None,
    ) -> CompositeReward:
        """Compute composite reward from all sources.

        Args:
            outcome_reward: From AAT's RewardCalculator (0-1).
            behavioral_heuristic: From AAT's BehavioralScorer (0-1).
            behavioral_judge: From Dojo's JudgeEvaluator (0-1, when available).
            actions_taken: Number of actions in the episode.
            episode_type: Episode type string for baseline lookup.
            phase_transitions: List of phase transition keys for bonus.
            components: Additional reward component breakdown.

        Returns:
            CompositeReward with full breakdown.
        """
        # Behavioral reward: use judge when available, otherwise heuristic
        behavioral_reward = behavioral_judge if behavioral_judge is not None else behavioral_heuristic

        # Weighted combination
        weighted_outcome = self.weights.outcome * outcome_reward
        weighted_behavioral = self.weights.behavioral * behavioral_reward

        # Efficiency penalty
        efficiency_penalty = self._compute_efficiency_penalty(actions_taken, episode_type)

        # Phase bonus
        phase_bonus = self._compute_phase_bonus(phase_transitions or [])

        total = weighted_outcome + weighted_behavioral + efficiency_penalty + phase_bonus
        total = max(0.0, min(1.0, total))

        return CompositeReward(
            total=total,
            outcome=outcome_reward,
            behavioral_heuristic=behavioral_heuristic,
            behavioral_judge=behavioral_judge,
            efficiency_penalty=efficiency_penalty,
            phase_bonus=phase_bonus,
            components=components or {},
        )

    def _compute_efficiency_penalty(self, actions_taken: int, episode_type: str) -> float:
        """Compute efficiency penalty relative to baseline."""
        baseline = self.baseline_actions.get(episode_type, 0.0)
        if baseline <= 0 or actions_taken <= 0:
            return 0.0

        excess = max(0, actions_taken - baseline) / baseline
        return -self.weights.efficiency_penalty_weight * excess

    def _compute_phase_bonus(self, transitions: List[str]) -> float:
        """Compute sparse phase transition bonus."""
        bonus = 0.0
        for t in transitions:
            bonus += PHASE_BONUSES.get(t, 0.0)
        return bonus * self.weights.phase_bonus_weight
