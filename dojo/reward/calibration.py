"""Cross-validation monitor for reward function drift detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CalibrationEntry:
    """Single calibration data point."""

    episode_id: str
    episode_type: str
    outcome_score: float
    heuristic_score: float
    judge_score: Optional[float]
    flag: str  # "aligned", "performative", "unconventional", "unknown"


@dataclass
class CalibrationReport:
    """Summary statistics from calibration monitoring."""

    total_entries: int
    aligned_rate: float
    performative_rate: float
    unconventional_rate: float
    heuristic_judge_correlation: Optional[float]
    outcome_behavioral_correlation: Optional[float]
    per_episode_type: Dict[str, Dict[str, float]]


class CalibrationMonitor:
    """Detects reward function drift by comparing reward channels.

    Tracks agreement between:
    - Judge scores and outcome scores
    - Heuristic scores and judge scores
    - Reward distribution stability over time

    Flags episodes as:
    - "performative": high behavioral + low outcome (going through motions)
    - "unconventional": low behavioral + high outcome (effective but different)
    - "aligned": both agree
    """

    def __init__(
        self,
        performative_threshold: float = 0.4,
        unconventional_threshold: float = 0.4,
    ) -> None:
        self.performative_threshold = performative_threshold
        self.unconventional_threshold = unconventional_threshold
        self._entries: List[CalibrationEntry] = []

    def record(
        self,
        episode_id: str,
        episode_type: str,
        outcome_score: float,
        heuristic_score: float,
        judge_score: Optional[float] = None,
    ) -> CalibrationEntry:
        """Record a calibration data point and compute flag.

        Args:
            episode_id: Unique episode identifier.
            episode_type: Episode type string.
            outcome_score: From AAT's RewardCalculator (0-1).
            heuristic_score: From AAT's BehavioralScorer (0-1).
            judge_score: From Dojo's JudgeEvaluator (0-1), when available.

        Returns:
            CalibrationEntry with computed flag.
        """
        behavioral = judge_score if judge_score is not None else heuristic_score
        flag = self._compute_flag(outcome_score, behavioral)

        entry = CalibrationEntry(
            episode_id=episode_id,
            episode_type=episode_type,
            outcome_score=outcome_score,
            heuristic_score=heuristic_score,
            judge_score=judge_score,
            flag=flag,
        )
        self._entries.append(entry)
        return entry

    def report(self) -> CalibrationReport:
        """Generate a calibration summary report."""
        n = len(self._entries)
        if n == 0:
            return CalibrationReport(
                total_entries=0,
                aligned_rate=0.0,
                performative_rate=0.0,
                unconventional_rate=0.0,
                heuristic_judge_correlation=None,
                outcome_behavioral_correlation=None,
                per_episode_type={},
            )

        flag_counts = {"aligned": 0, "performative": 0, "unconventional": 0, "unknown": 0}
        per_type: Dict[str, Dict[str, List[float]]] = {}

        for entry in self._entries:
            flag_counts[entry.flag] = flag_counts.get(entry.flag, 0) + 1

            if entry.episode_type not in per_type:
                per_type[entry.episode_type] = {
                    "outcome": [], "heuristic": [], "judge": [],
                }
            per_type[entry.episode_type]["outcome"].append(entry.outcome_score)
            per_type[entry.episode_type]["heuristic"].append(entry.heuristic_score)
            if entry.judge_score is not None:
                per_type[entry.episode_type]["judge"].append(entry.judge_score)

        # Compute correlations
        heuristic_judge_corr = self._compute_heuristic_judge_correlation()
        outcome_behavioral_corr = self._compute_outcome_behavioral_correlation()

        # Per-type averages
        per_type_summary: Dict[str, Dict[str, float]] = {}
        for etype, scores in per_type.items():
            per_type_summary[etype] = {
                "avg_outcome": _mean(scores["outcome"]),
                "avg_heuristic": _mean(scores["heuristic"]),
                "count": float(len(scores["outcome"])),
            }
            if scores["judge"]:
                per_type_summary[etype]["avg_judge"] = _mean(scores["judge"])

        return CalibrationReport(
            total_entries=n,
            aligned_rate=flag_counts["aligned"] / n,
            performative_rate=flag_counts["performative"] / n,
            unconventional_rate=flag_counts["unconventional"] / n,
            heuristic_judge_correlation=heuristic_judge_corr,
            outcome_behavioral_correlation=outcome_behavioral_corr,
            per_episode_type=per_type_summary,
        )

    def get_flagged_entries(self, flag: str) -> List[CalibrationEntry]:
        """Get all entries with a specific flag."""
        return [e for e in self._entries if e.flag == flag]

    def clear(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()

    def _compute_flag(self, outcome: float, behavioral: float) -> str:
        """Determine the flag for an outcome/behavioral score pair."""
        diff = behavioral - outcome

        if diff > self.performative_threshold:
            return "performative"  # high behavioral, low outcome
        elif diff < -self.unconventional_threshold:
            return "unconventional"  # low behavioral, high outcome
        else:
            return "aligned"

    def _compute_heuristic_judge_correlation(self) -> Optional[float]:
        """Pearson correlation between heuristic and judge scores."""
        pairs = [
            (e.heuristic_score, e.judge_score)
            for e in self._entries
            if e.judge_score is not None
        ]
        if len(pairs) < 3:
            return None
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return _pearson(xs, ys)

    def _compute_outcome_behavioral_correlation(self) -> Optional[float]:
        """Pearson correlation between outcome and behavioral scores."""
        if len(self._entries) < 3:
            return None
        xs = [e.outcome_score for e in self._entries]
        ys = [e.heuristic_score for e in self._entries]
        return _pearson(xs, ys)


def _mean(values: List[float]) -> float:
    """Compute mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x == 0 or var_y == 0:
        return None

    return cov / (var_x ** 0.5 * var_y ** 0.5)
