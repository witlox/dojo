"""Tests for judge evaluator (mock API)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from dojo.reward.judge_evaluator import JudgeEvaluator, JudgeResult


def test_judge_result_fields() -> None:
    result = JudgeResult(
        score=0.8,
        justification="Good behavior",
        per_behavior={"B-01": 0.9, "B-02": 0.7},
        flags=[],
    )
    assert result.score == 0.8
    assert len(result.per_behavior) == 2


def test_should_evaluate_cadence() -> None:
    judge = JudgeEvaluator(evaluate_every_n=3)
    assert not judge.should_evaluate()  # 1
    assert not judge.should_evaluate()  # 2
    assert judge.should_evaluate()       # 3
    assert not judge.should_evaluate()  # 4
    assert not judge.should_evaluate()  # 5
    assert judge.should_evaluate()       # 6


def test_reset_counter() -> None:
    judge = JudgeEvaluator(evaluate_every_n=2)
    judge.should_evaluate()  # 1
    judge.reset_counter()
    assert not judge.should_evaluate()  # 1 again
    assert judge.should_evaluate()       # 2


@pytest.mark.asyncio
async def test_evaluate_without_api_key_returns_fallback() -> None:
    judge = JudgeEvaluator(api_key="")
    result = await judge.evaluate(
        decision_traces=[{"action": "ask_question", "content": "What scope?"}],
        expected_behaviors=["B-01", "B-02"],
        episode_type="elicitation",
        task_context="Build a login page",
    )
    assert result.score == 0.5
    assert "fallback" in result.flags
    assert "B-01" in result.per_behavior
    assert "B-02" in result.per_behavior


@pytest.mark.asyncio
async def test_evaluate_with_mock_api() -> None:
    mock_response = json.dumps({
        "overall_score": 0.85,
        "justification": "Strong elicitation behavior",
        "per_behavior": {"B-01": 0.9, "B-02": 0.8},
        "flags": [],
    })

    judge = JudgeEvaluator(api_key="test-key")
    with patch.object(judge, "_call_api", new_callable=AsyncMock, return_value=mock_response):
        result = await judge.evaluate(
            decision_traces=[
                {"action": "ask_question", "content": "What is the scope?"},
                {"action": "ask_question", "content": "What constraints exist?"},
            ],
            expected_behaviors=["B-01", "B-02"],
            episode_type="elicitation",
            task_context="Build a REST API",
        )
    assert result.score == 0.85
    assert result.per_behavior["B-01"] == 0.9


@pytest.mark.asyncio
async def test_evaluate_handles_api_error() -> None:
    judge = JudgeEvaluator(api_key="test-key")
    with patch.object(judge, "_call_api", new_callable=AsyncMock, side_effect=RuntimeError("API down")):
        result = await judge.evaluate(
            decision_traces=[],
            expected_behaviors=["B-01"],
            episode_type="elicitation",
            task_context="test",
        )
    assert result.score == 0.5
    assert "fallback" in result.flags


def test_parse_response_json_in_markdown() -> None:
    judge = JudgeEvaluator()
    response = """Here is my evaluation:

```json
{
  "overall_score": 0.75,
  "justification": "Good but not great",
  "per_behavior": {"B-01": 0.8},
  "flags": ["performative"]
}
```"""
    result = judge._parse_response(response, ["B-01"])
    assert result.score == 0.75
    assert "performative" in result.flags


def test_parse_response_malformed_json() -> None:
    judge = JudgeEvaluator()
    result = judge._parse_response("not json at all", ["B-01"])
    assert result.score == 0.5
    assert "fallback" in result.flags


def test_judge_score_in_valid_range() -> None:
    judge = JudgeEvaluator()
    response = json.dumps({
        "overall_score": 0.42,
        "justification": "test",
        "per_behavior": {},
        "flags": [],
    })
    result = judge._parse_response(response, [])
    assert 0.0 <= result.score <= 1.0
