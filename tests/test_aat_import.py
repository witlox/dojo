"""Verify all 28 AAT RL API symbols are importable."""
from __future__ import annotations

import pytest


def test_episode_harness_imports() -> None:
    from src.rl import EpisodeRunner, EpisodeResult

    assert EpisodeRunner is not None
    assert EpisodeResult is not None


def test_scenario_imports() -> None:
    from src.rl import ScenarioCatalog, ScenarioConfig, EPISODE_TYPES

    assert ScenarioCatalog is not None
    assert ScenarioConfig is not None
    assert EPISODE_TYPES is not None


def test_observation_imports() -> None:
    from src.rl import ObservationExtractor, Observation, AgentObservation

    assert ObservationExtractor is not None
    assert Observation is not None
    assert AgentObservation is not None


def test_reward_imports() -> None:
    from src.rl import RewardCalculator, RewardSignal, RewardWeights

    assert RewardCalculator is not None
    assert RewardSignal is not None
    assert RewardWeights is not None


def test_behavioral_imports() -> None:
    from src.rl import BehavioralScorer, BehavioralCode, BEHAVIORAL_CODES

    assert BehavioralScorer is not None
    assert BehavioralCode is not None
    assert BEHAVIORAL_CODES is not None


def test_action_imports() -> None:
    from src.rl import (
        ActionExecutor,
        InjectDisturbance,
        SwapAgentRole,
        ModifyBacklog,
        ModifyTeamComposition,
        AdjustSprintParams,
        ACTION_SPACE_SPEC,
    )

    assert ActionExecutor is not None
    assert InjectDisturbance is not None
    assert SwapAgentRole is not None
    assert ModifyBacklog is not None
    assert ModifyTeamComposition is not None
    assert AdjustSprintParams is not None
    assert ACTION_SPACE_SPEC is not None


def test_checkpoint_imports() -> None:
    from src.rl import CheckpointManager, Checkpoint

    assert CheckpointManager is not None
    assert Checkpoint is not None


def test_config_imports() -> None:
    from src.rl import ExperimentConfigBuilder, ExperimentConfig

    assert ExperimentConfigBuilder is not None
    assert ExperimentConfig is not None


def test_phase_runner_imports() -> None:
    from src.rl import PhaseRunner, PhaseResult

    assert PhaseRunner is not None
    assert PhaseResult is not None


def test_runtime_registration_import() -> None:
    from src.rl import register_runtime

    assert register_runtime is not None


def test_all_28_symbols_importable() -> None:
    """Verify the complete set of 28 exported symbols."""
    from src.rl import (
        EpisodeRunner,
        EpisodeResult,
        ScenarioCatalog,
        ScenarioConfig,
        EPISODE_TYPES,
        ObservationExtractor,
        Observation,
        AgentObservation,
        RewardCalculator,
        RewardSignal,
        RewardWeights,
        BehavioralScorer,
        BehavioralCode,
        BEHAVIORAL_CODES,
        ActionExecutor,
        InjectDisturbance,
        SwapAgentRole,
        ModifyBacklog,
        ModifyTeamComposition,
        AdjustSprintParams,
        ACTION_SPACE_SPEC,
        CheckpointManager,
        Checkpoint,
        ExperimentConfigBuilder,
        ExperimentConfig,
        PhaseRunner,
        PhaseResult,
        register_runtime,
    )

    symbols = [
        EpisodeRunner, EpisodeResult, ScenarioCatalog, ScenarioConfig,
        EPISODE_TYPES, ObservationExtractor, Observation, AgentObservation,
        RewardCalculator, RewardSignal, RewardWeights, BehavioralScorer,
        BehavioralCode, BEHAVIORAL_CODES, ActionExecutor, InjectDisturbance,
        SwapAgentRole, ModifyBacklog, ModifyTeamComposition, AdjustSprintParams,
        ACTION_SPACE_SPEC, CheckpointManager, Checkpoint, ExperimentConfigBuilder,
        ExperimentConfig, PhaseRunner, PhaseResult, register_runtime,
    ]
    assert len(symbols) == 28


def test_behavioral_codes_has_30_entries() -> None:
    from src.rl import BEHAVIORAL_CODES

    assert len(BEHAVIORAL_CODES) == 30, (
        f"Expected 30 behavioral codes, got {len(BEHAVIORAL_CODES)}"
    )


def test_episode_types_has_13_entries() -> None:
    from src.rl import EPISODE_TYPES

    assert len(EPISODE_TYPES) == 13, (
        f"Expected 13 episode types, got {len(EPISODE_TYPES)}"
    )


def test_action_space_spec_has_5_entries() -> None:
    from src.rl import ACTION_SPACE_SPEC

    assert len(ACTION_SPACE_SPEC) == 5, (
        f"Expected 5 action space entries, got {len(ACTION_SPACE_SPEC)}"
    )
