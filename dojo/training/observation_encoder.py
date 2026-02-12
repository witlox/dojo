"""Observation encoder — converts structured observations to text prompts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dojo.env.prompt_renderer import render_observation


class ObservationEncoder:
    """Converts structured AAT observations + scenario config into text prompts.

    This is the bridge between gym observations and LLM inputs.
    The structured observation contains sprint state metadata;
    the encoder renders it into a natural language prompt suitable
    for the training candidate model.
    """

    def __init__(
        self,
        include_behavioral_hints: bool = False,
        max_prompt_length: int = 4000,
    ) -> None:
        self.include_behavioral_hints = include_behavioral_hints
        self.max_prompt_length = max_prompt_length

    def encode(
        self,
        observation: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None,
        episode_type: Optional[str] = None,
        stage: int = 1,
    ) -> str:
        """Encode an observation into a text prompt.

        Args:
            observation: Raw observation dict from ObservationExtractor.to_dict().
            scenario: Optional scenario config for episode context.
            episode_type: Episode type string.
            stage: Curriculum stage (1-4).

        Returns:
            Text prompt string.
        """
        parts: List[str] = []

        # System context based on episode type
        if episode_type:
            parts.append(self._episode_type_context(episode_type, stage))

        # Rendered observation
        rendered = render_observation(observation, scenario)
        parts.append(rendered)

        # Behavioral hints (optional — can be used for curriculum guidance)
        if self.include_behavioral_hints and scenario:
            expected = scenario.get("expected_behaviors", [])
            if expected:
                parts.append(self._behavioral_hint(expected))

        prompt = "\n\n".join(parts)

        # Truncate if too long
        if len(prompt) > self.max_prompt_length:
            prompt = prompt[: self.max_prompt_length - 3] + "..."

        return prompt

    def _episode_type_context(self, episode_type: str, stage: int) -> str:
        """Generate context string for the episode type."""
        contexts = {
            "elicitation": (
                "You are in a story refinement session. Your goal is to ask "
                "clarifying questions to understand the task requirements before "
                "proceeding to implementation."
            ),
            "decomposition": (
                "You are planning the technical approach for a story. Break the "
                "work into tractable tasks, identify dependencies and risks, and "
                "select an implementation approach."
            ),
            "implementation": (
                "You are implementing a task. Write code, run tests, and iterate. "
                "Check your progress at regular intervals and decide whether to "
                "continue, pivot, or escalate."
            ),
            "self_monitoring": (
                "You are at a checkpoint during implementation. Assess your "
                "progress honestly and decide whether to continue, change approach, "
                "or ask for help."
            ),
            "research": (
                "You need to find specific technical information. Search "
                "effectively, evaluate sources critically, and synthesize findings."
            ),
            "triage": (
                "A disturbance has occurred. Assess its severity and blast radius, "
                "then decide on the appropriate response."
            ),
            "recovery": (
                "You need to recover from a disturbance. Diagnose the root cause, "
                "fix the issue without causing collateral damage, and capture learnings."
            ),
            "scope_change": (
                "New scope has been added mid-sprint. Evaluate the impact, negotiate "
                "trade-offs, and adjust the plan appropriately."
            ),
            "borrowing_arrival": (
                "You have been borrowed to an unfamiliar team. Orient yourself in "
                "their codebase and conventions, then contribute productively."
            ),
            "cross_team_dependency": (
                "You have identified a potential cross-team dependency. Analyze it, "
                "communicate with the other team, and propose a resolution."
            ),
            "knowledge_handoff": (
                "You are departing the team. Identify critical knowledge that only "
                "you hold and transfer it effectively."
            ),
            "onboarding_support": (
                "A new team member has joined. Help them onboard by assessing their "
                "skills, providing context, and assigning appropriate work."
            ),
            "compensation": (
                "A key team member has departed. Assess the impact, recover critical "
                "knowledge from artifacts, and adjust commitments."
            ),
        }
        context = contexts.get(episode_type, f"Episode type: {episode_type}")
        return f"[Stage {stage}] {context}"

    def _behavioral_hint(self, expected_behaviors: List[str]) -> str:
        """Generate optional behavioral hint."""
        return (
            "Focus areas: " + ", ".join(expected_behaviors)
        )
