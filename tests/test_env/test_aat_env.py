"""Tests for AATEnv gym wrapper."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from dojo.env.aat_env import AATEnv, EpisodeConfig


@pytest.fixture
def mock_aat_components():
    """Patch AAT imports with mocks for unit testing."""
    # Build mock scenario
    mock_scenario = MagicMock()
    mock_scenario.phases = ["planning", "development"]
    mock_scenario.expected_behaviors = ["B-01", "B-02"]

    # Mock ScenarioCatalog
    mock_catalog = MagicMock()
    mock_catalog.generate.return_value = mock_scenario

    # Mock ObservationExtractor
    mock_obs = MagicMock()
    mock_obs_dict = {
        "sprint_num": 0,
        "phase": "planning",
        "agents": [{"agent_id": "a1", "role_id": "dev_mid_backend", "seniority": "mid"}],
        "kanban": {"todo": [{"id": "c1"}]},
        "sprint_metrics": {"velocity_ratio": 0.5, "coverage_score": 0.5},
        "disturbances_active": [],
        "meta_learnings_count": 0,
    }
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=mock_obs)
    mock_extractor.to_dict.return_value = mock_obs_dict

    # Mock PhaseResult
    mock_phase_result = MagicMock()
    mock_phase_result.decisions = [{"action": "ask_question", "content": "What is the scope?"}]
    mock_phase_result.phase = "planning"

    # Mock PhaseRunner
    mock_phase_runner = MagicMock()
    mock_phase_runner.run_phase = AsyncMock(return_value=mock_phase_result)

    # Mock RewardSignal
    mock_reward_signal = MagicMock()
    mock_reward_signal.total = 0.65

    # Mock RewardCalculator
    mock_reward_calc = MagicMock()
    mock_reward_calc.compute_phase_reward.return_value = mock_reward_signal

    # Mock BehavioralScorer
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = (0.7, ["B-01"])

    # Mock ActionExecutor
    mock_action_exec = MagicMock()
    mock_action_exec.execute = AsyncMock(return_value={"success": True})

    # Mock EpisodeRunner
    mock_runner = MagicMock()

    # Mock RewardWeights
    mock_weights = MagicMock()

    with patch.dict("sys.modules", {}):
        with patch("dojo.env.aat_env.AATEnv._ensure_initialized") as mock_init:
            def setup_mocks(self=None):
                pass

            mock_init.side_effect = setup_mocks

            env = AATEnv(
                episode_config=EpisodeConfig(
                    stage=1,
                    episode_type="elicitation",
                    difficulty=0.5,
                ),
            )

            # Inject mocks directly
            env._runner = mock_runner
            env._phase_runner = mock_phase_runner
            env._catalog = mock_catalog
            env._obs_extractor = mock_extractor
            env._reward_calc = mock_reward_calc
            env._scorer = mock_scorer
            env._action_exec = mock_action_exec

            yield env, {
                "scenario": mock_scenario,
                "catalog": mock_catalog,
                "extractor": mock_extractor,
                "phase_runner": mock_phase_runner,
                "reward_calc": mock_reward_calc,
                "scorer": mock_scorer,
                "action_exec": mock_action_exec,
                "runner": mock_runner,
                "obs_dict": mock_obs_dict,
                "phase_result": mock_phase_result,
                "reward_signal": mock_reward_signal,
            }


def test_env_has_correct_spaces() -> None:
    with patch("dojo.env.aat_env.AATEnv._ensure_initialized"):
        env = AATEnv()
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.n == 6


@pytest.mark.asyncio
async def test_reset_returns_valid_observation(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    obs, info = await env.reset_async()
    assert "sprint_num" in obs
    assert "phase" in obs
    assert "scenario" in info
    assert env.observation_space.contains(obs)


@pytest.mark.asyncio
async def test_step_returns_correct_tuple(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    await env.reset_async()
    obs, reward, terminated, truncated, info = await env.step_async(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "phase" in info
    assert "behavioral_score" in info
    assert env.observation_space.contains(obs)


@pytest.mark.asyncio
async def test_step_with_action(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    await env.reset_async()
    # Inject disturbance action
    obs, reward, terminated, truncated, info = await env.step_async(1)
    mocks["action_exec"].execute.assert_called_once()
    assert info["action_result"] is not None


@pytest.mark.asyncio
async def test_full_episode_loop(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    obs, info = await env.reset_async()

    total_reward = 0.0
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = await env.step_async(0)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    # Scenario has 2 phases, so should take 2 steps
    assert steps == 2
    assert terminated
    assert total_reward > 0


@pytest.mark.asyncio
async def test_step_after_termination_raises(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    await env.reset_async()
    # Run through all phases
    await env.step_async(0)
    await env.step_async(0)
    # Now it should be terminated
    with pytest.raises(RuntimeError, match="Episode already ended"):
        await env.step_async(0)


@pytest.mark.asyncio
async def test_reset_with_options(mock_aat_components) -> None:
    env, mocks = mock_aat_components
    obs, info = await env.reset_async(
        options={"episode_type": "decomposition", "difficulty": 0.8}
    )
    assert info["episode_type"] == "decomposition"
    assert info["difficulty"] == 0.8


def test_episode_config_defaults() -> None:
    cfg = EpisodeConfig()
    assert cfg.stage == 1
    assert cfg.episode_type == "elicitation"
    assert cfg.difficulty == 0.5


def test_render_human_mode() -> None:
    with patch("dojo.env.aat_env.AATEnv._ensure_initialized"):
        env = AATEnv(render_mode="human")
        env._current_scenario = MagicMock()
        env._current_scenario.phases = ["planning"]
        env.render()  # Should not raise
