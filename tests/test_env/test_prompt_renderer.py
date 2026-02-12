"""Tests for prompt rendering."""
from __future__ import annotations

from dojo.env.prompt_renderer import render_observation


def test_render_minimal() -> None:
    obs = {"sprint_num": 0, "phase": "planning", "agents": [], "kanban": {}}
    result = render_observation(obs)
    assert "Sprint 0" in result
    assert "planning" in result


def test_render_with_scenario() -> None:
    obs = {"sprint_num": 1, "phase": "development", "agents": [], "kanban": {}}
    scenario = {
        "episode_type": "elicitation",
        "difficulty": 0.7,
        "expected_behaviors": ["B-01", "B-02"],
        "backlog_stories": [{"title": "Add login"}],
    }
    result = render_observation(obs, scenario)
    assert "elicitation" in result
    assert "B-01" in result
    assert "Add login" in result


def test_render_with_agents() -> None:
    obs = {
        "sprint_num": 1,
        "phase": "development",
        "agents": [
            {
                "agent_id": "ahmed",
                "role_id": "dev_senior_backend",
                "seniority": "senior",
                "is_swapped": False,
                "is_onboarding": False,
            },
            {
                "agent_id": "bob",
                "role_id": "dev_mid_frontend",
                "seniority": "mid",
                "is_swapped": True,
                "is_onboarding": False,
            },
        ],
        "kanban": {},
    }
    result = render_observation(obs)
    assert "ahmed" in result
    assert "senior" in result
    assert "swapped" in result


def test_render_with_disturbances() -> None:
    obs = {
        "sprint_num": 2,
        "phase": "development",
        "agents": [],
        "kanban": {},
        "disturbances_active": ["flaky_test", "scope_creep"],
    }
    result = render_observation(obs)
    assert "flaky_test" in result
    assert "scope_creep" in result


def test_render_with_kanban() -> None:
    obs = {
        "sprint_num": 1,
        "phase": "development",
        "agents": [],
        "kanban": {
            "todo": [{"id": "c1"}, {"id": "c2"}],
            "in_progress": [{"id": "c3"}],
            "done": [],
        },
    }
    result = render_observation(obs)
    assert "todo: 2 cards" in result
    assert "in_progress: 1 cards" in result
