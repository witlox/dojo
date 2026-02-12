"""Evaluation metrics and transfer scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PrimaryMetrics:
    """Primary evaluation metrics (transfer indicators)."""

    elicitation_quality: float = 0.0
    sufficiency_calibration: float = 0.0
    research_effectiveness: float = 0.0
    decomposition_accuracy: float = 0.0
    self_correction_rate: float = 0.0
    adaptation_quality: float = 0.0
    intent_match: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "elicitation_quality": self.elicitation_quality,
            "sufficiency_calibration": self.sufficiency_calibration,
            "research_effectiveness": self.research_effectiveness,
            "decomposition_accuracy": self.decomposition_accuracy,
            "self_correction_rate": self.self_correction_rate,
            "adaptation_quality": self.adaptation_quality,
            "intent_match": self.intent_match,
        }

    def mean(self) -> float:
        vals = list(self.to_dict().values())
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class SecondaryMetrics:
    """Secondary metrics (behavioral health)."""

    question_count_calibration: float = 0.0
    action_efficiency: float = 0.0
    escalation_appropriateness: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "question_count_calibration": self.question_count_calibration,
            "action_efficiency": self.action_efficiency,
            "escalation_appropriateness": self.escalation_appropriateness,
        }


@dataclass
class TransferScore:
    """Per-behavior transfer score: solo_metric / team_metric."""

    behavior_code: str
    team_score: float
    solo_score: float

    @property
    def ratio(self) -> float:
        if self.team_score <= 0:
            return 0.0
        return self.solo_score / self.team_score

    @property
    def transfers_well(self) -> bool:
        """Transfer score > 0.8 indicates good transfer."""
        return self.ratio > 0.8


@dataclass
class EvalSummary:
    """Summary of a complete evaluation run."""

    num_tasks: int
    primary: PrimaryMetrics
    secondary: SecondaryMetrics
    transfer_scores: List[TransferScore]
    per_task_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_transfer(self) -> float:
        if not self.transfer_scores:
            return 0.0
        return sum(ts.ratio for ts in self.transfer_scores) / len(self.transfer_scores)

    @property
    def well_transferring_count(self) -> int:
        return sum(1 for ts in self.transfer_scores if ts.transfers_well)


class MetricsCalculator:
    """Computes evaluation metrics from solo evaluation results."""

    def compute_primary(
        self,
        eval_results: List[Dict[str, Any]],
    ) -> PrimaryMetrics:
        """Compute primary metrics from evaluation results.

        Args:
            eval_results: List of EvalResult-like dicts from solo evaluations.

        Returns:
            Aggregated PrimaryMetrics.
        """
        if not eval_results:
            return PrimaryMetrics()

        elicitation_scores: List[float] = []
        sufficiency_scores: List[float] = []
        decomposition_scores: List[float] = []
        intent_scores: List[float] = []

        for result in eval_results:
            phase_scores = result.get("phase_scores", {})
            elicitation_scores.append(phase_scores.get("elicitation", 0.0))
            sufficiency_scores.append(self._compute_sufficiency(result))
            decomposition_scores.append(phase_scores.get("planning", 0.0))
            intent_scores.append(phase_scores.get("verification", 0.0))

        return PrimaryMetrics(
            elicitation_quality=_mean(elicitation_scores),
            sufficiency_calibration=_mean(sufficiency_scores),
            decomposition_accuracy=_mean(decomposition_scores),
            intent_match=_mean(intent_scores),
        )

    def compute_transfer_scores(
        self,
        team_scores: Dict[str, float],
        solo_scores: Dict[str, float],
    ) -> List[TransferScore]:
        """Compute per-behavior transfer scores.

        Args:
            team_scores: Per-behavior scores from team training.
            solo_scores: Per-behavior scores from solo evaluation.

        Returns:
            List of TransferScore for each behavior.
        """
        transfer_scores = []
        all_behaviors = set(team_scores.keys()) | set(solo_scores.keys())

        for behavior in sorted(all_behaviors):
            team = team_scores.get(behavior, 0.0)
            solo = solo_scores.get(behavior, 0.0)
            transfer_scores.append(TransferScore(
                behavior_code=behavior,
                team_score=team,
                solo_score=solo,
            ))

        return transfer_scores

    def summarize(
        self,
        eval_results: List[Dict[str, Any]],
        team_scores: Optional[Dict[str, float]] = None,
        solo_scores: Optional[Dict[str, float]] = None,
    ) -> EvalSummary:
        """Generate a complete evaluation summary.

        Args:
            eval_results: Results from solo evaluation tasks.
            team_scores: Optional per-behavior team training scores.
            solo_scores: Optional per-behavior solo scores.

        Returns:
            EvalSummary with all metrics.
        """
        primary = self.compute_primary(eval_results)
        secondary = SecondaryMetrics()  # Filled by judge evaluator separately

        transfer = []
        if team_scores and solo_scores:
            transfer = self.compute_transfer_scores(team_scores, solo_scores)

        per_task = {}
        for result in eval_results:
            task_id = result.get("task_id", "unknown")
            per_task[task_id] = result.get("overall_score", 0.0)

        return EvalSummary(
            num_tasks=len(eval_results),
            primary=primary,
            secondary=secondary,
            transfer_scores=transfer,
            per_task_scores=per_task,
        )

    def _compute_sufficiency(self, result: Dict[str, Any]) -> float:
        """Heuristic sufficiency calibration from eval result."""
        phases = result.get("phases", [])
        for phase in phases:
            if isinstance(phase, dict) and phase.get("phase") == "elicitation":
                num_questions = len([
                    a for a in phase.get("actions", [])
                    if a.get("type") == "ask_question"
                ])
                # Rough heuristic: 2-5 questions is usually right
                if 2 <= num_questions <= 5:
                    return 1.0
                elif 1 <= num_questions <= 7:
                    return 0.7
                else:
                    return 0.3
        return 0.5


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
