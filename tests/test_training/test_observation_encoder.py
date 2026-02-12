"""Tests for observation encoder."""
from __future__ import annotations

import pytest

from dojo.training.observation_encoder import ObservationEncoder


def test_encode_minimal() -> None:
    enc = ObservationEncoder()
    obs = {"sprint_num": 0, "phase": "planning", "agents": [], "kanban": {}}
    result = enc.encode(obs)
    assert "Sprint 0" in result
    assert "planning" in result


def test_encode_with_episode_type() -> None:
    enc = ObservationEncoder()
    obs = {"sprint_num": 1, "phase": "development", "agents": [], "kanban": {}}
    result = enc.encode(obs, episode_type="elicitation", stage=1)
    assert "clarifying questions" in result
    assert "[Stage 1]" in result


def test_encode_with_scenario() -> None:
    enc = ObservationEncoder()
    obs = {"sprint_num": 0, "phase": "planning", "agents": [], "kanban": {}}
    scenario = {
        "episode_type": "decomposition",
        "difficulty": 0.7,
        "expected_behaviors": ["B-05", "B-06"],
    }
    result = enc.encode(obs, scenario=scenario)
    assert "B-05" in result or "decomposition" in result


def test_encode_with_behavioral_hints() -> None:
    enc = ObservationEncoder(include_behavioral_hints=True)
    obs = {"sprint_num": 0, "phase": "planning", "agents": [], "kanban": {}}
    scenario = {"expected_behaviors": ["B-01", "B-02"]}
    result = enc.encode(obs, scenario=scenario)
    assert "Focus areas" in result
    assert "B-01" in result


def test_encode_truncation() -> None:
    enc = ObservationEncoder(max_prompt_length=50)
    obs = {
        "sprint_num": 0,
        "phase": "planning",
        "agents": [{"agent_id": f"agent_{i}", "role_id": "dev", "seniority": "mid"} for i in range(20)],
        "kanban": {},
    }
    result = enc.encode(obs, episode_type="implementation", stage=2)
    assert len(result) <= 50


def test_all_episode_type_contexts() -> None:
    enc = ObservationEncoder()
    obs = {"sprint_num": 0, "phase": "planning", "agents": [], "kanban": {}}
    episode_types = [
        "elicitation", "decomposition", "implementation", "self_monitoring",
        "research", "triage", "recovery", "scope_change",
        "borrowing_arrival", "cross_team_dependency",
        "knowledge_handoff", "onboarding_support", "compensation",
    ]
    for ep_type in episode_types:
        result = enc.encode(obs, episode_type=ep_type, stage=1)
        assert len(result) > 0
