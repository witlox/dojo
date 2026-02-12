"""Claude Opus judge evaluator for nuanced behavioral quality scoring."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""

    score: float
    justification: str
    per_behavior: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


# Behavioral rubric template for the judge prompt
_RUBRIC_TEMPLATE = """You are evaluating the behavioral quality of an AI agent's decisions during a software development episode.

## Episode Context
- Episode type: {episode_type}
- Task: {task_context}
- Expected behaviors: {expected_behaviors}

## Decision Trace
{decision_traces}

## Evaluation Rubric

For each expected behavior, score 0.0 to 1.0:
- 1.0: Exemplary demonstration of the behavior
- 0.7: Good demonstration with minor gaps
- 0.5: Partial demonstration
- 0.3: Weak demonstration with significant gaps
- 0.0: Behavior not demonstrated or anti-pattern exhibited

## Expected Behaviors

{behavior_descriptions}

## Instructions

Evaluate each behavior and provide:
1. A score (0.0-1.0) for each behavior
2. A brief justification for each score
3. An overall quality score (0.0-1.0)
4. Any flags: "performative" if behaviors seem superficial, "unconventional" if approach is unusual but effective

Respond in JSON format:
```json
{{
  "overall_score": 0.0,
  "justification": "...",
  "per_behavior": {{"B-01": 0.0, ...}},
  "flags": []
}}
```"""


class JudgeEvaluator:
    """Claude Opus-based behavioral quality scorer.

    Runs periodically (not every episode) to provide nuanced behavioral
    scoring that calibrates against AAT's fast heuristic BehavioralScorer.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        evaluate_every_n: int = 50,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.evaluate_every_n = evaluate_every_n
        self._episode_counter = 0

    def should_evaluate(self) -> bool:
        """Check if this episode should receive a judge evaluation."""
        self._episode_counter += 1
        return self._episode_counter % self.evaluate_every_n == 0

    def reset_counter(self) -> None:
        """Reset the episode counter (e.g., at stage transition)."""
        self._episode_counter = 0

    async def evaluate(
        self,
        decision_traces: List[Dict[str, Any]],
        expected_behaviors: List[str],
        episode_type: str,
        task_context: str,
        behavior_descriptions: Optional[Dict[str, str]] = None,
    ) -> JudgeResult:
        """Evaluate decision traces against behavioral taxonomy.

        Args:
            decision_traces: Full trace of agent decisions from the episode.
            expected_behaviors: List of expected behavior codes (e.g., ["B-01", "B-02"]).
            episode_type: Episode type string.
            task_context: Description of the task being performed.
            behavior_descriptions: Optional mapping of behavior codes to descriptions.

        Returns:
            JudgeResult with score, justification, per-behavior scores, and flags.
        """
        if not self.api_key:
            return self._fallback_result(expected_behaviors)

        # Format decision traces for the prompt
        traces_text = self._format_traces(decision_traces)
        behaviors_text = self._format_behaviors(expected_behaviors, behavior_descriptions)

        prompt = _RUBRIC_TEMPLATE.format(
            episode_type=episode_type,
            task_context=task_context[:2000],
            expected_behaviors=", ".join(expected_behaviors),
            decision_traces=traces_text[:8000],
            behavior_descriptions=behaviors_text,
        )

        try:
            response = await self._call_api(prompt)
            return self._parse_response(response, expected_behaviors)
        except Exception:
            return self._fallback_result(expected_behaviors)

    async def _call_api(self, prompt: str) -> str:
        """Call the Anthropic API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code != 200:
                raise RuntimeError(f"Anthropic API returned {response.status_code}")
            data = response.json()
            return data["content"][0]["text"]

    def _parse_response(self, response: str, expected_behaviors: List[str]) -> JudgeResult:
        """Parse judge response into JudgeResult."""
        # Extract JSON from response (may be wrapped in markdown code block)
        json_str = response
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            json_str = response[start:end]
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            json_str = response[start:end]

        try:
            data = json.loads(json_str.strip())
            return JudgeResult(
                score=float(data.get("overall_score", 0.5)),
                justification=str(data.get("justification", "")),
                per_behavior={
                    k: float(v)
                    for k, v in data.get("per_behavior", {}).items()
                },
                flags=list(data.get("flags", [])),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return self._fallback_result(expected_behaviors)

    def _fallback_result(self, expected_behaviors: List[str]) -> JudgeResult:
        """Return a neutral result when API is unavailable."""
        return JudgeResult(
            score=0.5,
            justification="Judge evaluation unavailable; using neutral score.",
            per_behavior={b: 0.5 for b in expected_behaviors},
            flags=["fallback"],
        )

    def _format_traces(self, traces: List[Dict[str, Any]]) -> str:
        """Format decision traces for the prompt."""
        lines: List[str] = []
        for i, trace in enumerate(traces[:50]):  # Cap at 50 decisions
            action = trace.get("action", trace.get("action_content", "unknown"))
            content = trace.get("content", trace.get("context", ""))
            lines.append(f"Decision {i+1}: {action}")
            if content:
                lines.append(f"  Content: {str(content)[:200]}")
        return "\n".join(lines)

    def _format_behaviors(
        self,
        codes: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format behavior descriptions for the prompt."""
        descs = descriptions or {}
        lines: List[str] = []
        for code in codes:
            desc = descs.get(code, f"Behavioral pattern {code}")
            lines.append(f"- {code}: {desc}")
        return "\n".join(lines)
