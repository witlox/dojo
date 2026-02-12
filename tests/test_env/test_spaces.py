"""Tests for gym space conversions."""
from __future__ import annotations

import numpy as np
import pytest

from dojo.env.spaces import (
    ACTION_INDEX,
    INDEX_TO_ACTION,
    INDEX_TO_PHASE,
    PHASE_INDEX,
    action_from_gym,
    build_action_space,
    build_observation_space,
    observation_to_gym,
)


def test_phase_index_mapping() -> None:
    assert len(PHASE_INDEX) == 5
    assert PHASE_INDEX["planning"] == 0
    assert PHASE_INDEX["development"] == 1
    assert INDEX_TO_PHASE[0] == "planning"


def test_action_index_mapping() -> None:
    assert len(ACTION_INDEX) == 6
    assert ACTION_INDEX["no_op"] == 0
    assert ACTION_INDEX["inject_disturbance"] == 1
    assert INDEX_TO_ACTION[0] == "no_op"


def test_build_observation_space() -> None:
    space = build_observation_space()
    assert isinstance(space, dict.__class__) or hasattr(space, "spaces")
    assert "sprint_num" in space.spaces
    assert "phase" in space.spaces
    assert "num_agents" in space.spaces
    assert "velocity_ratio" in space.spaces


def test_build_action_space() -> None:
    space = build_action_space()
    assert space.n == 6


def test_observation_to_gym_minimal() -> None:
    obs_dict = {
        "sprint_num": 1,
        "phase": "development",
        "agents": [{"agent_id": "a1"}, {"agent_id": "a2"}],
        "kanban": {"todo": [{"id": "c1"}], "in_progress": [], "done": []},
        "sprint_metrics": {"velocity_ratio": 0.8, "coverage_score": 0.6},
        "disturbances_active": ["flaky_test"],
        "meta_learnings_count": 3,
    }
    result = observation_to_gym(obs_dict)
    assert result["sprint_num"] == 1
    assert result["phase"] == 1  # development
    assert result["num_agents"] == 2
    assert result["num_cards"] == 1
    assert result["num_disturbances_active"] == 1
    assert result["meta_learnings_count"] == 3
    assert isinstance(result["velocity_ratio"], np.floating)


def test_observation_to_gym_empty() -> None:
    result = observation_to_gym({})
    assert result["sprint_num"] == 0
    assert result["phase"] == 0
    assert result["num_agents"] == 0


def test_observation_to_gym_clamps_values() -> None:
    obs_dict = {
        "sprint_num": 100,
        "phase": "unknown_phase",
        "agents": [{"agent_id": f"a{i}"} for i in range(20)],
        "sprint_metrics": {"velocity_ratio": 10.0, "coverage_score": 2.0},
        "meta_learnings_count": 200,
        "disturbances_active": list(range(20)),
    }
    result = observation_to_gym(obs_dict)
    assert result["sprint_num"] == 19
    assert result["phase"] == 0  # unknown maps to 0
    assert result["num_agents"] == 12
    assert result["velocity_ratio"] <= 5.0
    assert result["coverage_score"] <= 1.0
    assert result["meta_learnings_count"] == 99
    assert result["num_disturbances_active"] == 9


def test_observation_space_contains_valid_obs() -> None:
    space = build_observation_space()
    obs = observation_to_gym({
        "sprint_num": 1,
        "phase": "planning",
        "agents": [],
        "kanban": {},
        "sprint_metrics": {"velocity_ratio": 0.5, "coverage_score": 0.5},
        "disturbances_active": [],
        "meta_learnings_count": 0,
    })
    assert space.contains(obs)


def test_action_from_gym_no_op() -> None:
    result = action_from_gym(0)
    assert result is None


def test_action_from_gym_inject_disturbance() -> None:
    result = action_from_gym(1, {"disturbance_type": "scope_creep", "severity": 0.8})
    assert result is not None
    assert result.disturbance_type == "scope_creep"
    assert result.severity == 0.8


def test_action_from_gym_defaults() -> None:
    result = action_from_gym(1)  # inject_disturbance with defaults
    assert result is not None
    assert result.disturbance_type == "flaky_test"
    assert result.severity == 0.5


def test_action_from_gym_all_types() -> None:
    for idx in range(6):
        result = action_from_gym(idx)
        if idx == 0:
            assert result is None
        else:
            assert result is not None
