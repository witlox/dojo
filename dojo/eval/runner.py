"""Evaluation runner with three cadences."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from dojo.eval.metrics import EvalSummary, MetricsCalculator
from dojo.eval.solo_env import EvalPhase, SoloEvalEnvironment, SoloTask


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    quick_num_tasks: int = 10
    full_num_tasks: int = 50
    comprehensive_num_tasks: int = 100
    task_bank_dir: Optional[str] = None


class EvalRunner:
    """Runs evaluations at three cadences.

    - quick_eval(): 10 tasks, primary metrics (every 100 training episodes)
    - full_eval(): 50 tasks, all metrics (every 500 training episodes)
    - comprehensive_eval(): 100 tasks, transfer scores (stage completion)
    """

    def __init__(
        self,
        task_bank: Optional[List[SoloTask]] = None,
        config: Optional[EvalConfig] = None,
        model_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config or EvalConfig()
        self._task_bank = task_bank or []
        self._model_fn = model_fn
        self._calculator = MetricsCalculator()
        self._history: List[EvalSummary] = []

    def load_task_bank(self, directory: str) -> None:
        """Load tasks from YAML files in a directory."""
        task_dir = Path(directory)
        if not task_dir.exists():
            return

        tasks = []
        for f in sorted(task_dir.glob("*.yaml")) + sorted(task_dir.glob("*.yml")):
            with open(f) as fh:
                data = yaml.safe_load(fh)
                if isinstance(data, dict):
                    tasks.append(SoloTask.from_dict(data))
                elif isinstance(data, list):
                    tasks.extend(SoloTask.from_dict(d) for d in data)

        self._task_bank = tasks

    def quick_eval(self) -> EvalSummary:
        """Quick evaluation: 10 tasks, primary metrics only."""
        tasks = self._select_tasks(self.config.quick_num_tasks)
        results = self._run_tasks(tasks)
        summary = self._calculator.summarize(results)
        self._history.append(summary)
        return summary

    def full_eval(self) -> EvalSummary:
        """Full evaluation: 50 tasks, all metrics."""
        tasks = self._select_tasks(self.config.full_num_tasks)
        results = self._run_tasks(tasks)
        summary = self._calculator.summarize(results)
        self._history.append(summary)
        return summary

    def comprehensive_eval(
        self,
        team_scores: Optional[Dict[str, float]] = None,
        solo_behavior_scores: Optional[Dict[str, float]] = None,
    ) -> EvalSummary:
        """Comprehensive evaluation: 100 tasks, transfer scores."""
        tasks = self._select_tasks(self.config.comprehensive_num_tasks)
        results = self._run_tasks(tasks)
        summary = self._calculator.summarize(
            results,
            team_scores=team_scores,
            solo_scores=solo_behavior_scores,
        )
        self._history.append(summary)
        return summary

    @property
    def history(self) -> List[EvalSummary]:
        return list(self._history)

    def _select_tasks(self, num: int) -> List[SoloTask]:
        """Select tasks from the task bank."""
        if not self._task_bank:
            # Generate minimal tasks for testing
            return [
                SoloTask(
                    task_id=f"auto-{i}",
                    description=f"Implement feature {i}",
                    ambiguity="medium",
                    domain="api",
                    complexity="medium",
                    codebase="greenfield",
                )
                for i in range(num)
            ]

        # Cycle through task bank
        selected = []
        for i in range(num):
            selected.append(self._task_bank[i % len(self._task_bank)])
        return selected

    def _run_tasks(self, tasks: List[SoloTask]) -> List[Dict[str, Any]]:
        """Run evaluation on a set of tasks."""
        results = []
        for task in tasks:
            result = self._run_single_task(task)
            results.append(result)
        return results

    def _run_single_task(self, task: SoloTask) -> Dict[str, Any]:
        """Run a single evaluation task through the 5-phase protocol."""
        env = SoloEvalEnvironment(task)

        # Simulate model interaction through all phases
        while not env.is_complete:
            phase = env.current_phase

            if phase == EvalPhase.ELICITATION:
                # Simulate asking questions
                for q in task.expected_questions[:5]:
                    response = env.record_action({"type": "ask_question", "content": q})
                env.record_action({"type": "signal_ready"})

            elif phase == EvalPhase.RESEARCH:
                env.record_action({"type": "search", "content": f"how to {task.description}"})
                env.record_action({"type": "evaluate_source"})

            elif phase == EvalPhase.PLANNING:
                env.record_action({"type": "create_plan", "content": "Implementation plan"})
                env.record_action({"type": "identify_risk"})
                env.record_action({"type": "signal_plan_complete"})

            elif phase == EvalPhase.EXECUTION:
                env.record_action({"type": "write_code"})
                env.record_action({"type": "run_tests"})
                env.record_action({"type": "self_correct"})

            elif phase == EvalPhase.VERIFICATION:
                env.record_action({"type": "verify_intent"})
                env.record_action({"type": "check_criteria"})

            env.advance_phase()

        eval_result = env.get_result()

        return {
            "task_id": eval_result.task_id,
            "phase_scores": eval_result.phase_scores,
            "overall_score": eval_result.overall_score,
            "phases": [
                {"phase": t.phase.value, "actions": t.actions}
                for t in eval_result.phases
            ],
        }
