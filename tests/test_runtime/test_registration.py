"""Tests for runtime registration with AAT."""
from __future__ import annotations

import pytest

from dojo.runtime.registration import register_candidate_runtime


def test_register_candidate_runtime() -> None:
    register_candidate_runtime()

    from src.agents.runtime.factory import registered_runtime_types

    types = registered_runtime_types()
    assert "training_candidate" in types


def test_registered_runtime_creates_instance() -> None:
    register_candidate_runtime()

    from unittest.mock import MagicMock

    from src.agents.runtime.factory import _RUNTIME_REGISTRY

    factory = _RUNTIME_REGISTRY["training_candidate"]
    config = {"endpoint": "mock://", "model": "test"}
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "test"
    tool.parameters = {"properties": {}, "required": []}
    runtime = factory(config, [tool])

    from dojo.runtime.candidate_runtime import TrainingCandidateRuntime

    assert isinstance(runtime, TrainingCandidateRuntime)
