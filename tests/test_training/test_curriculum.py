"""Tests for curriculum manager."""
from __future__ import annotations

import pytest

from dojo.training.curriculum import (
    CurriculumManager,
    GraduationCriteria,
    STAGE_EPISODE_TYPES,
    STAGE_GRADUATION,
)


def test_stage_episode_types_coverage() -> None:
    all_types = set()
    for types in STAGE_EPISODE_TYPES.values():
        all_types.update(types)
    assert len(all_types) == 13


def test_graduation_criteria_met() -> None:
    criteria = GraduationCriteria(metrics={"quality": 0.7, "speed": 0.5})
    assert criteria.is_met({"quality": 0.8, "speed": 0.6})
    assert not criteria.is_met({"quality": 0.6, "speed": 0.6})


def test_graduation_unmet_criteria() -> None:
    criteria = GraduationCriteria(metrics={"quality": 0.7, "speed": 0.5})
    gaps = criteria.unmet_criteria({"quality": 0.5, "speed": 0.6})
    assert "quality" in gaps
    assert "speed" not in gaps


def test_curriculum_start_stage() -> None:
    cm = CurriculumManager(start_stage=2)
    assert cm.current_stage == 2


def test_curriculum_episode_types() -> None:
    cm = CurriculumManager(start_stage=1)
    types = cm.episode_types_for_stage()
    assert "elicitation" in types
    assert len(types) == 5


def test_generate_batch_fallback() -> None:
    cm = CurriculumManager(start_stage=1)
    batch = cm._generate_fallback_batch(10, seed=42)
    assert len(batch) == 10
    for item in batch:
        assert "episode_type" in item
        assert item["stage"] == 1
        assert item["episode_type"] in STAGE_EPISODE_TYPES[1]


def test_record_metric() -> None:
    cm = CurriculumManager()
    cm.record_metric("quality", 0.5)
    cm.record_metric("quality", 0.7)
    metrics = cm.current_metrics()
    assert abs(metrics["quality"] - 0.6) < 0.01


def test_record_episode() -> None:
    cm = CurriculumManager()
    cm.record_episode()
    cm.record_episode()
    assert cm.episode_count == 2
    assert cm.stage_episode_count == 2


def test_advance_stage() -> None:
    cm = CurriculumManager(start_stage=1)
    new_stage = cm.advance_stage()
    assert new_stage == 2
    assert cm.current_stage == 2
    assert cm.stage_episode_count == 0


def test_advance_past_last_stage() -> None:
    cm = CurriculumManager(start_stage=4)
    result = cm.advance_stage()
    assert result == -1


def test_custom_stages() -> None:
    cm = CurriculumManager(start_stage=1, stages=[1, 3])
    cm.advance_stage()
    assert cm.current_stage == 3


def test_is_complete() -> None:
    cm = CurriculumManager(start_stage=1, stages=[1])
    # Record enough good metrics
    for _ in range(25):
        cm.record_metric("elicitation_quality", 0.8)
        cm.record_metric("decomposition_quality", 0.7)
        cm.record_metric("self_correction_rate", 0.6)
    assert cm.check_graduation()
