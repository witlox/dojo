"""Tests for training orchestrator."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dojo.orchestrator import OrchestratorConfig, TrainingOrchestrator


@pytest.fixture
def config() -> OrchestratorConfig:
    return OrchestratorConfig(
        base_model="test-model",
        output_dir="/tmp/dojo-test-orchestrator",
        stages=[1],
        mock_mode=True,
        ppo_update_every_n=4,
        judge_every_n=10,
        eval_quick_every_n=50,
        eval_full_every_n=100,
        checkpoint_every_n=50,
    )


def _make_mock_env():
    """Create a mock AATEnv that returns realistic observations."""
    mock_env = MagicMock()

    mock_obs = {
        "sprint_num": 0,
        "phase": "planning",
        "agents": [{"agent_id": "a1", "role_id": "dev_mid_backend", "seniority": "mid"}],
        "kanban": {"todo": [{"id": "c1"}]},
        "sprint_metrics": {"velocity_ratio": 0.5, "coverage_score": 0.5},
        "disturbances_active": [],
        "meta_learnings_count": 0,
    }
    mock_info = {
        "scenario": MagicMock(),
        "episode_type": "elicitation",
        "stage": 1,
        "difficulty": 0.5,
        "raw_observation": mock_obs,
    }
    mock_step_info = {
        "phase": "planning",
        "behavioral_score": 0.6,
        "behaviors_detected": ["B-01"],
        "raw_observation": mock_obs,
    }

    mock_env.reset_async = AsyncMock(return_value=(mock_obs, mock_info))

    # First step returns non-terminal, second step returns terminal
    mock_env.step_async = AsyncMock(
        side_effect=[
            (mock_obs, 0.5, False, False, mock_step_info),
            (mock_obs, 0.3, True, False, {**mock_step_info, "behavioral_score": 0.7}),
        ]
    )

    return mock_env


def test_orchestrator_config_defaults() -> None:
    config = OrchestratorConfig()
    assert config.base_model == "deepseek-ai/deepseek-coder-v2-lite-instruct"
    assert config.stages == [1, 2, 3, 4]
    assert config.mock_mode is False
    assert config.ppo_update_every_n == 16


def test_orchestrator_config_custom() -> None:
    config = OrchestratorConfig(
        base_model="custom-model",
        stages=[1, 2],
        mock_mode=True,
    )
    assert config.base_model == "custom-model"
    assert config.stages == [1, 2]
    assert config.mock_mode is True


def test_orchestrator_init(config) -> None:
    orch = TrainingOrchestrator(config)
    assert orch.config is config
    assert orch.curriculum is not None
    assert orch.reward_calc is not None
    assert orch.judge is not None
    assert orch.calibration is not None
    assert orch.buffer is not None
    assert orch.eval_runner is not None
    assert orch.trainer is None


def test_orchestrator_init_default_config() -> None:
    orch = TrainingOrchestrator()
    assert orch.config is not None
    assert orch.config.stages == [1, 2, 3, 4]


def test_orchestrator_components_initialized(config) -> None:
    orch = TrainingOrchestrator(config)
    assert orch.curriculum.current_stage == 1
    assert len(orch._episode_rewards) == 0
    assert len(orch._behavioral_scores) == 0
    assert len(orch._training_log) == 0


def test_run_single_episode(config) -> None:
    orch = TrainingOrchestrator(config)
    mock_env = _make_mock_env()

    with patch("dojo.orchestrator.AATEnv", return_value=mock_env):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                orch.run_single_episode(
                    episode_type="elicitation",
                    stage=1,
                    difficulty=0.5,
                    verbose=True,
                )
            )
        finally:
            loop.close()

    assert "reward" in result
    assert "behavioral_score" in result
    assert "episode_type" in result
    assert result["episode_type"] == "elicitation"


def test_run_single_episode_records_metrics(config) -> None:
    orch = TrainingOrchestrator(config)
    mock_env = _make_mock_env()

    with patch("dojo.orchestrator.AATEnv", return_value=mock_env):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                orch.run_single_episode(episode_type="elicitation", stage=1)
            )
        finally:
            loop.close()

    assert len(orch._episode_rewards) == 1
    assert len(orch._behavioral_scores) == 1


def test_run_single_episode_different_types(config) -> None:
    orch = TrainingOrchestrator(config)

    loop = asyncio.new_event_loop()
    try:
        for etype in ["elicitation", "decomposition", "implementation"]:
            mock_env = _make_mock_env()
            with patch("dojo.orchestrator.AATEnv", return_value=mock_env):
                result = loop.run_until_complete(
                    orch.run_single_episode(episode_type=etype, stage=1)
                )
                assert result["episode_type"] == etype
    finally:
        loop.close()

    assert len(orch._episode_rewards) == 3


def test_ppo_update_empty_buffer(config) -> None:
    orch = TrainingOrchestrator(config)
    # Should not raise when buffer is empty
    orch._do_ppo_update()
    assert orch.buffer.total_steps == 0


def test_ppo_update_with_data(config) -> None:
    from dojo.training.trajectory_buffer import Trajectory

    orch = TrainingOrchestrator(config)
    orch.buffer.add(Trajectory(
        episode_id="test-1",
        episode_type="elicitation",
        stage=1,
        prompts=["prompt1"],
        responses=["response1"],
        rewards=[0.5],
    ))

    # Should not raise — logs but doesn't train without GPU
    orch._do_ppo_update()
    # Buffer should be cleared after update
    assert orch.buffer.total_steps == 0


def test_calibration_after_episode(config) -> None:
    orch = TrainingOrchestrator(config)
    mock_env = _make_mock_env()

    with patch("dojo.orchestrator.AATEnv", return_value=mock_env):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                orch.run_single_episode(episode_type="elicitation", stage=1)
            )
        finally:
            loop.close()

    report = orch.calibration.report()
    assert report.total_entries == 1


def test_run_episode_accumulates_reward(config) -> None:
    orch = TrainingOrchestrator(config)
    mock_env = _make_mock_env()

    with patch("dojo.orchestrator.AATEnv", return_value=mock_env):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                orch.run_single_episode(episode_type="elicitation", stage=1)
            )
        finally:
            loop.close()

    # 0.5 + 0.3 from the mock side_effect
    assert result["reward"] == pytest.approx(0.8)


def test_run_stage_with_mocked_episode(config) -> None:
    """Test that _run_stage completes by mocking _run_episode directly."""
    orch = TrainingOrchestrator(config)

    call_count = [0]

    def early_graduation():
        call_count[0] += 1
        return call_count[0] > 3

    orch.curriculum.check_graduation = early_graduation

    # Mock _run_episode to return a realistic result without needing AAT
    async def mock_run_episode(env, scenario):
        return {
            "reward": 0.6,
            "behavioral_score": 0.7,
            "episode_type": scenario.get("episode_type", "elicitation"),
        }

    orch._run_episode = mock_run_episode

    with patch("dojo.orchestrator.AATEnv"):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(orch._run_stage(1))
        finally:
            loop.close()

    assert "episodes" in result
    assert "duration_seconds" in result
    assert "mean_reward" in result
    assert result["episodes"] > 0
