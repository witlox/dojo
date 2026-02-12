"""Solo evaluation environment — tests behavioral transfer without AAT."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalPhase(str, Enum):
    ELICITATION = "elicitation"
    RESEARCH = "research"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"


EVAL_PHASE_ORDER = [
    EvalPhase.ELICITATION,
    EvalPhase.RESEARCH,
    EvalPhase.PLANNING,
    EvalPhase.EXECUTION,
    EvalPhase.VERIFICATION,
]


@dataclass
class SoloTask:
    """A single evaluation task for the solo environment."""

    task_id: str
    description: str
    ambiguity: str  # "low", "medium", "high"
    domain: str
    complexity: str  # "low", "medium", "high"
    codebase: str  # "greenfield", "small_brownfield", "large_brownfield"
    risk: str = "medium"  # "low", "medium", "high"
    expected_questions: List[str] = field(default_factory=list)
    hidden_constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    scripted_answers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoloTask":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PhaseTrace:
    """Trace of model actions during one evaluation phase."""

    phase: EvalPhase
    actions: List[Dict[str, Any]]
    duration_seconds: float = 0.0


@dataclass
class EvalResult:
    """Result from a solo evaluation episode."""

    task_id: str
    phases: List[PhaseTrace]
    phase_scores: Dict[str, float]
    overall_score: float
    behavioral_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanProxy:
    """Simulates human responses for automated evaluation.

    Uses scripted answers when available, falls back to generic responses.
    """

    def __init__(self, scripted_answers: Optional[Dict[str, str]] = None) -> None:
        self._answers = scripted_answers or {}
        self._questions_asked: List[str] = []

    def respond(self, question: str) -> str:
        """Respond to a model's question."""
        self._questions_asked.append(question)

        # Check scripted answers (keyword matching)
        for keyword, answer in self._answers.items():
            if keyword.lower() in question.lower():
                return answer

        return "That's a good question. Let me think about it and get back to you."

    @property
    def questions_asked(self) -> List[str]:
        return list(self._questions_asked)


class SoloEvalEnvironment:
    """Standalone evaluation environment (no AAT dependency).

    Implements the 5-phase solo evaluation protocol:
    1. Elicitation — model asks clarifying questions
    2. Research — model searches for information
    3. Planning — model decomposes the task
    4. Execution — model implements the solution
    5. Verification — model verifies against intent
    """

    def __init__(self, task: SoloTask) -> None:
        self.task = task
        self.proxy = HumanProxy(scripted_answers=task.scripted_answers)
        self._current_phase_idx = 0
        self._traces: List[PhaseTrace] = []
        self._current_actions: List[Dict[str, Any]] = []

    @property
    def current_phase(self) -> EvalPhase:
        if self._current_phase_idx >= len(EVAL_PHASE_ORDER):
            return EVAL_PHASE_ORDER[-1]
        return EVAL_PHASE_ORDER[self._current_phase_idx]

    @property
    def is_complete(self) -> bool:
        return self._current_phase_idx >= len(EVAL_PHASE_ORDER)

    def get_context(self) -> Dict[str, Any]:
        """Get current context for the model."""
        return {
            "phase": self.current_phase.value,
            "task_description": self.task.description,
            "phase_number": self._current_phase_idx + 1,
            "total_phases": len(EVAL_PHASE_ORDER),
            "previous_phases": [t.phase.value for t in self._traces],
        }

    def record_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Record a model action and return any response.

        Args:
            action: Dict with at least "type" and optional "content".

        Returns:
            Response string (e.g., human proxy answer), or None.
        """
        self._current_actions.append(action)

        action_type = action.get("type", "")

        # Handle questions during elicitation
        if action_type == "ask_question" and self.current_phase == EvalPhase.ELICITATION:
            content = action.get("content", "")
            return self.proxy.respond(content)

        return None

    def advance_phase(self) -> Optional[EvalPhase]:
        """Complete the current phase and advance to the next.

        Returns:
            The new phase, or None if evaluation is complete.
        """
        trace = PhaseTrace(
            phase=self.current_phase,
            actions=list(self._current_actions),
        )
        self._traces.append(trace)
        self._current_actions = []
        self._current_phase_idx += 1

        if self.is_complete:
            return None
        return self.current_phase

    def get_result(self) -> EvalResult:
        """Get the evaluation result after all phases complete.

        Phase scores are computed based on action quality heuristics.
        Full scoring requires the judge evaluator (external).
        """
        # Finalize any remaining actions
        if self._current_actions:
            self._traces.append(PhaseTrace(
                phase=self.current_phase,
                actions=list(self._current_actions),
            ))

        phase_scores = {}
        behavioral_scores: Dict[str, float] = {}

        for trace in self._traces:
            phase_name = trace.phase.value
            num_actions = len(trace.actions)

            # Basic heuristic scoring
            if trace.phase == EvalPhase.ELICITATION:
                # More questions for higher ambiguity is better
                questions = [a for a in trace.actions if a.get("type") == "ask_question"]
                phase_scores[phase_name] = min(len(questions) / 5.0, 1.0)
            elif trace.phase == EvalPhase.PLANNING:
                # Having a plan is better than not
                phase_scores[phase_name] = min(num_actions / 3.0, 1.0)
            else:
                phase_scores[phase_name] = min(num_actions / 5.0, 1.0) if num_actions > 0 else 0.0

        overall = sum(phase_scores.values()) / max(len(phase_scores), 1)

        return EvalResult(
            task_id=self.task.task_id,
            phases=self._traces,
            phase_scores=phase_scores,
            overall_score=overall,
            behavioral_scores=behavioral_scores,
        )
