"""Tests for solo evaluation environment."""
from __future__ import annotations

import pytest

from dojo.eval.solo_env import (
    EVAL_PHASE_ORDER,
    EvalPhase,
    HumanProxy,
    SoloEvalEnvironment,
    SoloTask,
)


@pytest.fixture
def sample_task() -> SoloTask:
    return SoloTask(
        task_id="test-1",
        description="Build a REST API for user management",
        ambiguity="high",
        domain="api",
        complexity="medium",
        codebase="greenfield",
        expected_questions=["What endpoints?", "Auth required?"],
        acceptance_criteria=["CRUD operations", "Tests pass"],
        scripted_answers={"endpoint": "GET/POST /users", "auth": "JWT tokens"},
    )


def test_eval_phase_order() -> None:
    assert len(EVAL_PHASE_ORDER) == 5
    assert EVAL_PHASE_ORDER[0] == EvalPhase.ELICITATION
    assert EVAL_PHASE_ORDER[-1] == EvalPhase.VERIFICATION


def test_solo_task_from_dict() -> None:
    data = {
        "task_id": "t1",
        "description": "Test task",
        "ambiguity": "low",
        "domain": "api",
        "complexity": "low",
        "codebase": "greenfield",
    }
    task = SoloTask.from_dict(data)
    assert task.task_id == "t1"


def test_human_proxy_scripted() -> None:
    proxy = HumanProxy(scripted_answers={"scope": "Just the API, no frontend"})
    response = proxy.respond("What is the scope of this task?")
    assert "API" in response


def test_human_proxy_fallback() -> None:
    proxy = HumanProxy()
    response = proxy.respond("How should I handle errors?")
    assert len(response) > 0


def test_solo_env_initial_state(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)
    assert env.current_phase == EvalPhase.ELICITATION
    assert not env.is_complete


def test_solo_env_context(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)
    ctx = env.get_context()
    assert ctx["phase"] == "elicitation"
    assert ctx["phase_number"] == 1
    assert "REST API" in ctx["task_description"]


def test_solo_env_record_question(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)
    response = env.record_action({"type": "ask_question", "content": "What endpoints do you need?"})
    assert response is not None  # Proxy should respond


def test_solo_env_advance_phase(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)
    env.record_action({"type": "ask_question", "content": "What endpoints?"})
    next_phase = env.advance_phase()
    assert next_phase == EvalPhase.RESEARCH
    assert env.current_phase == EvalPhase.RESEARCH


def test_solo_env_full_protocol(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)
    phases_visited = []

    while not env.is_complete:
        phases_visited.append(env.current_phase)
        env.record_action({"type": "action", "content": "test"})
        env.advance_phase()

    assert len(phases_visited) == 5
    assert phases_visited[0] == EvalPhase.ELICITATION
    assert phases_visited[-1] == EvalPhase.VERIFICATION


def test_solo_env_result(sample_task) -> None:
    env = SoloEvalEnvironment(sample_task)

    # Run through all phases
    for _ in range(5):
        env.record_action({"type": "ask_question", "content": "test?"})
        env.record_action({"type": "action", "content": "test"})
        env.advance_phase()

    result = env.get_result()
    assert result.task_id == "test-1"
    assert len(result.phases) == 5
    assert 0.0 <= result.overall_score <= 1.0
