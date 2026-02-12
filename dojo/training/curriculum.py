"""Curriculum manager for stage progression across 4 training stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraduationCriteria:
    """Criteria that must be met to advance to the next stage."""

    metrics: Dict[str, float]  # metric_name -> minimum_threshold

    def is_met(self, current_metrics: Dict[str, float]) -> bool:
        """Check if all criteria are met."""
        for metric, threshold in self.metrics.items():
            if current_metrics.get(metric, 0.0) < threshold:
                return False
        return True

    def unmet_criteria(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Return criteria that are not yet met with their gaps."""
        gaps = {}
        for metric, threshold in self.metrics.items():
            current = current_metrics.get(metric, 0.0)
            if current < threshold:
                gaps[metric] = threshold - current
        return gaps


# Episode types per stage
STAGE_EPISODE_TYPES: Dict[int, List[str]] = {
    1: ["elicitation", "decomposition", "implementation", "self_monitoring", "research"],
    2: ["triage", "recovery", "scope_change"],
    3: ["borrowing_arrival", "cross_team_dependency"],
    4: ["knowledge_handoff", "onboarding_support", "compensation"],
}

# Graduation criteria per stage
STAGE_GRADUATION: Dict[int, GraduationCriteria] = {
    1: GraduationCriteria(metrics={
        "elicitation_quality": 0.70,
        "decomposition_quality": 0.65,
        "self_correction_rate": 0.50,
    }),
    2: GraduationCriteria(metrics={
        "triage_accuracy": 0.70,
        "velocity_drop": -0.30,  # Negative means "less than 30% drop"
    }),
    3: GraduationCriteria(metrics={
        "actions_to_productive": -3.0,  # Negative means "fewer than 3"
        "orientation_score": 0.70,
    }),
    4: GraduationCriteria(metrics={
        "no_regression": 0.0,  # No regression on stage 1 metrics
        "velocity_dip": -0.20,
    }),
}

# Target episode counts per stage
STAGE_EPISODE_COUNTS: Dict[int, int] = {
    1: 5000,
    2: 1200,
    3: 700,
    4: 700,
}


class CurriculumManager:
    """Manages training stage progression through the 4-stage curriculum.

    Uses AAT's ScenarioCatalog to generate episode configs per stage.
    Tracks graduation criteria and advances stages when met.
    """

    def __init__(
        self,
        start_stage: int = 1,
        stages: Optional[List[int]] = None,
    ) -> None:
        self._current_stage = start_stage
        self._stages = stages or [1, 2, 3, 4]
        self._stage_idx = self._stages.index(start_stage) if start_stage in self._stages else 0
        self._metrics: Dict[str, List[float]] = {}
        self._episode_count = 0
        self._stage_episode_count = 0

    @property
    def current_stage(self) -> int:
        return self._current_stage

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def stage_episode_count(self) -> int:
        return self._stage_episode_count

    def episode_types_for_stage(self, stage: Optional[int] = None) -> List[str]:
        """Get episode types for a stage."""
        s = stage or self._current_stage
        return STAGE_EPISODE_TYPES.get(s, [])

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of scenario configs for the current stage.

        Uses AAT's ScenarioCatalog when available, otherwise returns
        minimal config dicts for episode generation.

        Args:
            batch_size: Number of episodes to generate.
            seed: Random seed.

        Returns:
            List of scenario config dicts.
        """
        try:
            from src.rl import ScenarioCatalog
            catalog = ScenarioCatalog()
            scenarios = catalog.generate_curriculum(
                stage=self._current_stage,
                num_episodes=batch_size,
                seed=seed,
            )
            return [_scenario_to_dict(s) for s in scenarios]
        except ImportError:
            # Fallback without AAT
            return self._generate_fallback_batch(batch_size, seed)

    def _generate_fallback_batch(
        self, batch_size: int, seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate batch without AAT dependency (for testing)."""
        import random
        rng = random.Random(seed)
        episode_types = self.episode_types_for_stage()
        batch = []
        for _ in range(batch_size):
            ep_type = rng.choice(episode_types)
            difficulty = rng.uniform(0.3, 0.9)
            batch.append({
                "episode_type": ep_type,
                "stage": self._current_stage,
                "difficulty": difficulty,
                "target_agent_slot": "dev_mid_backend",
            })
        return batch

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value for graduation tracking."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

    def record_episode(self) -> None:
        """Record that an episode was completed."""
        self._episode_count += 1
        self._stage_episode_count += 1

    def current_metrics(self) -> Dict[str, float]:
        """Get averaged current metrics (last 20 values)."""
        result = {}
        for name, values in self._metrics.items():
            recent = values[-20:] if len(values) > 20 else values
            result[name] = sum(recent) / len(recent) if recent else 0.0
        return result

    def check_graduation(self) -> bool:
        """Check if current stage graduation criteria are met."""
        criteria = STAGE_GRADUATION.get(self._current_stage)
        if criteria is None:
            return True  # No criteria = auto-graduate

        current = self.current_metrics()

        # Handle negative thresholds (meaning "less than")
        adjusted = {}
        for metric, threshold in criteria.metrics.items():
            if threshold < 0:
                # For negative thresholds, the metric should be <= abs(threshold)
                val = current.get(metric, 0.0)
                adjusted[metric] = -abs(threshold) if val <= abs(threshold) else 0.0
            else:
                adjusted[metric] = current.get(metric, 0.0)

        return criteria.is_met(adjusted)

    def advance_stage(self) -> int:
        """Advance to the next curriculum stage.

        Returns:
            The new stage number, or -1 if all stages complete.
        """
        self._stage_idx += 1
        if self._stage_idx >= len(self._stages):
            return -1

        self._current_stage = self._stages[self._stage_idx]
        self._stage_episode_count = 0
        return self._current_stage

    def is_complete(self) -> bool:
        """Check if all stages have been completed."""
        return self._stage_idx >= len(self._stages) - 1 and self.check_graduation()


def _scenario_to_dict(scenario: Any) -> Dict[str, Any]:
    """Convert a ScenarioConfig to a plain dict."""
    if hasattr(scenario, "__dict__"):
        return {
            "episode_type": getattr(scenario, "episode_type", ""),
            "stage": getattr(scenario, "stage", 1),
            "difficulty": getattr(scenario, "difficulty", 0.5),
            "target_agent_slot": getattr(scenario, "target_agent_slot", "dev_mid_backend"),
            "expected_behaviors": getattr(scenario, "expected_behaviors", []),
            "phases": getattr(scenario, "phases", []),
            "duration_minutes": getattr(scenario, "duration_minutes", 5),
        }
    return dict(scenario) if isinstance(scenario, dict) else {}
