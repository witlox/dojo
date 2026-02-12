"""Tests for PPO trainer (unit tests with mocks — no GPU required)."""
from __future__ import annotations

import pytest

from dojo.training.ppo_trainer import DojoTrainer, TrainerConfig, TrainStepResult


def test_trainer_config_defaults() -> None:
    cfg = TrainerConfig()
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.quantization_bits == 4
    assert cfg.learning_rate == 1e-5
    assert cfg.batch_size == 4
    assert cfg.ppo_epochs == 4
    assert cfg.cliprange == 0.2


def test_trainer_config_custom() -> None:
    cfg = TrainerConfig(
        base_model="test-model",
        lora_rank=32,
        learning_rate=5e-5,
    )
    assert cfg.base_model == "test-model"
    assert cfg.lora_rank == 32
    assert cfg.learning_rate == 5e-5


def test_trainer_creation() -> None:
    trainer = DojoTrainer(TrainerConfig(base_model="test"))
    assert not trainer._initialized
    assert trainer.step_count == 0


def test_train_step_result_fields() -> None:
    result = TrainStepResult(
        loss=0.5,
        policy_loss=0.3,
        value_loss=0.2,
        entropy=0.1,
        kl_divergence=0.01,
        num_samples=4,
    )
    assert result.loss == 0.5
    assert result.num_samples == 4


def test_save_adapter_requires_initialization() -> None:
    trainer = DojoTrainer()
    with pytest.raises(RuntimeError, match="not initialized"):
        trainer.save_adapter("test")
