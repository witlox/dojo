"""Tests for CLI entry point."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dojo.cli import main


def test_no_command_shows_help() -> None:
    result = main([])
    assert result == 1


def test_help_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_train_requires_base_model() -> None:
    with pytest.raises(SystemExit):
        main(["train", "--output", "/tmp/test"])


def test_train_requires_output() -> None:
    with pytest.raises(SystemExit):
        main(["train", "--base-model", "test-model"])


def test_evaluate_requires_model() -> None:
    with pytest.raises(SystemExit):
        main(["evaluate", "--output", "/tmp/test"])


def test_evaluate_requires_output() -> None:
    with pytest.raises(SystemExit):
        main(["evaluate", "--model", "test-model"])


def _mock_orchestrator_run():
    """Create a mock orchestrator whose run() returns realistic results."""
    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(return_value={
        "stage_1": {
            "episodes": 10,
            "duration_seconds": 5.0,
            "mean_reward": 0.65,
            "adapter_path": None,
        },
        "final_eval": {
            "num_tasks": 5,
            "primary_mean": 0.7,
            "mean_transfer": 0.0,
        },
        "total_episodes": 10,
        "calibration": {
            "aligned_rate": 1.0,
            "performative_rate": 0.0,
        },
    })
    return mock_orch


def test_train_mock_mode() -> None:
    mock_orch = _mock_orchestrator_run()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dojo.orchestrator.TrainingOrchestrator", return_value=mock_orch):
            result = main([
                "train",
                "--base-model", "test-model",
                "--output", tmpdir,
                "--stages", "1",
                "--mock",
            ])
        assert result == 0

        results_file = os.path.join(tmpdir, "training_results.json")
        assert os.path.exists(results_file)

        with open(results_file) as f:
            data = json.load(f)
        assert "stage_1" in data
        assert "final_eval" in data
        assert "total_episodes" in data


def test_train_custom_stages() -> None:
    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(return_value={
        "stage_1": {"episodes": 5, "duration_seconds": 2.0, "mean_reward": 0.6, "adapter_path": None},
        "stage_2": {"episodes": 5, "duration_seconds": 2.0, "mean_reward": 0.7, "adapter_path": None},
        "final_eval": {"num_tasks": 5, "primary_mean": 0.7, "mean_transfer": 0.0},
        "total_episodes": 10,
        "calibration": {"aligned_rate": 1.0, "performative_rate": 0.0},
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dojo.orchestrator.TrainingOrchestrator", return_value=mock_orch):
            result = main([
                "train",
                "--base-model", "test-model",
                "--output", tmpdir,
                "--stages", "1,2",
                "--mock",
            ])
        assert result == 0

        with open(os.path.join(tmpdir, "training_results.json")) as f:
            data = json.load(f)
        assert "stage_1" in data
        assert "stage_2" in data


def test_evaluate_basic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = main([
            "evaluate",
            "--model", "test-model",
            "--output", tmpdir,
            "--num-tasks", "3",
        ])
        assert result == 0

        results_file = os.path.join(tmpdir, "eval_results.json")
        assert os.path.exists(results_file)

        with open(results_file) as f:
            data = json.load(f)
        assert "num_tasks" in data
        assert "primary_metrics" in data
        assert data["num_tasks"] == 3


def test_episode_mock() -> None:
    mock_orch = MagicMock()
    mock_orch.run_single_episode = AsyncMock(return_value={
        "reward": 0.5,
        "behavioral_score": 0.6,
        "episode_type": "elicitation",
    })

    with patch("dojo.orchestrator.TrainingOrchestrator", return_value=mock_orch):
        result = main([
            "episode",
            "--type", "elicitation",
            "--stage", "1",
            "--difficulty", "0.5",
            "--mock",
        ])
    assert result == 0


def test_episode_different_types() -> None:
    for etype in ["elicitation", "decomposition"]:
        mock_orch = MagicMock()
        mock_orch.run_single_episode = AsyncMock(return_value={
            "reward": 0.5,
            "behavioral_score": 0.6,
            "episode_type": etype,
        })

        with patch("dojo.orchestrator.TrainingOrchestrator", return_value=mock_orch):
            result = main([
                "episode",
                "--type", etype,
                "--stage", "1",
                "--mock",
            ])
        assert result == 0


def test_verbose_flag() -> None:
    mock_orch = _mock_orchestrator_run()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dojo.orchestrator.TrainingOrchestrator", return_value=mock_orch):
            result = main([
                "-v",
                "train",
                "--base-model", "test-model",
                "--output", tmpdir,
                "--stages", "1",
                "--mock",
            ])
        assert result == 0


def test_train_parses_stages_correctly() -> None:
    """Verify that comma-separated stages are parsed into a list of ints."""
    captured_config = {}

    def capture_config(config):
        captured_config["stages"] = config.stages
        mock = MagicMock()
        mock.run = AsyncMock(return_value={
            "stage_1": {}, "stage_3": {},
            "final_eval": {"num_tasks": 1, "primary_mean": 0.5, "mean_transfer": 0.0},
            "total_episodes": 5,
            "calibration": {"aligned_rate": 1.0, "performative_rate": 0.0},
        })
        return mock

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dojo.orchestrator.TrainingOrchestrator", side_effect=capture_config):
            main([
                "train",
                "--base-model", "m",
                "--output", tmpdir,
                "--stages", "1,3",
                "--mock",
            ])

    assert captured_config["stages"] == [1, 3]
