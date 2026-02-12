"""Tests for TrainingCandidateRuntime."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dojo.runtime.candidate_runtime import TraceEntry, TrainingCandidateRuntime


@pytest.fixture
def mock_tools():
    """Create mock tools for runtime initialization."""
    tool = MagicMock()
    tool.name = "read_file"
    tool.description = "Read a file"
    tool.parameters = {"properties": {"path": {"type": "string"}}, "required": ["path"]}
    return [tool]


@pytest.fixture
def runtime(mock_tools):
    """Create a runtime in mock mode."""
    config = {"endpoint": "mock://", "model": "test-model", "max_tokens": 1024}
    return TrainingCandidateRuntime(config, mock_tools)


def test_implements_agent_runtime() -> None:
    from src.agents.runtime.base import AgentRuntime

    assert issubclass(TrainingCandidateRuntime, AgentRuntime)


def test_runtime_init(runtime) -> None:
    assert runtime.endpoint == "mock://"
    assert runtime.model == "test-model"
    assert runtime.lora_adapter_path is None
    assert len(runtime._traces) == 0


def test_set_lora_adapter(runtime) -> None:
    assert runtime.lora_adapter_path is None
    runtime.set_lora_adapter("/path/to/lora-v1")
    assert runtime.lora_adapter_path == "/path/to/lora-v1"
    runtime.set_lora_adapter(None)
    assert runtime.lora_adapter_path is None


def test_clear_traces(runtime) -> None:
    runtime._traces.append(
        TraceEntry(
            timestamp=0, system_prompt="", user_message="", response="",
            turn=0, tool_calls=[], duration_ms=0,
        )
    )
    assert len(runtime._traces) == 1
    runtime.clear_traces()
    assert len(runtime._traces) == 0


def test_get_traces_returns_copy(runtime) -> None:
    runtime._traces.append(
        TraceEntry(
            timestamp=0, system_prompt="", user_message="", response="",
            turn=0, tool_calls=[], duration_ms=0,
        )
    )
    traces = runtime.get_traces()
    assert len(traces) == 1
    traces.clear()
    assert len(runtime._traces) == 1  # Original not affected


@pytest.mark.asyncio
async def test_mock_execute(runtime) -> None:
    result = await runtime.execute_task(
        system_prompt="You are a developer.",
        user_message="Implement a login feature.",
    )
    assert result.success is True
    assert result.turns == 1
    assert len(result.content) > 0
    assert result.metadata.get("trace_count") == 1


@pytest.mark.asyncio
async def test_mock_collects_traces(runtime) -> None:
    await runtime.execute_task("system", "task1")
    await runtime.execute_task("system", "task2")
    traces = runtime.get_traces()
    assert len(traces) == 2
    assert traces[0].user_message == "task1"
    assert traces[1].user_message == "task2"


@pytest.mark.asyncio
async def test_mock_mode_env_var(mock_tools) -> None:
    config = {"endpoint": "http://localhost:8000", "model": "test"}
    rt = TrainingCandidateRuntime(config, mock_tools)
    with patch.dict(os.environ, {"MOCK_LLM": "true"}):
        result = await rt.execute_task("system", "task")
    assert result.success is True


def test_parse_tool_calls(runtime) -> None:
    text = """I'll read the file first.

<tool_call>
  <name>read_file</name>
  <arguments>
    <path>src/main.py</path>
  </arguments>
</tool_call>"""
    calls = runtime._parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["params"]["path"] == "src/main.py"


def test_parse_tool_calls_empty(runtime) -> None:
    text = "No tool calls here, just a response."
    calls = runtime._parse_tool_calls(text)
    assert len(calls) == 0


def test_parse_tool_calls_malformed_xml(runtime) -> None:
    text = "<tool_call><name>bad<</tool_call>"
    calls = runtime._parse_tool_calls(text)
    assert len(calls) == 0


def test_trace_entry_fields() -> None:
    entry = TraceEntry(
        timestamp=1234567890.0,
        system_prompt="sys",
        user_message="user",
        response="resp",
        turn=0,
        tool_calls=[{"name": "test"}],
        duration_ms=100.0,
    )
    assert entry.timestamp == 1234567890.0
    assert entry.turn == 0
    assert len(entry.tool_calls) == 1
