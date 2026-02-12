"""Tests for backlog generator."""
from __future__ import annotations

import pytest

from dojo.data.backlog_generator import (
    BacklogGenerator,
    BacklogStory,
    DOMAINS,
    LANGUAGES,
    CODEBASE_CONTEXTS,
)


def test_generate_single_story() -> None:
    gen = BacklogGenerator(seed=42)
    story = gen.generate_story()
    assert isinstance(story, BacklogStory)
    assert len(story.title) > 0
    assert len(story.description) > 0
    assert story.domain in DOMAINS
    assert story.language in LANGUAGES
    assert story.codebase_context in CODEBASE_CONTEXTS
    assert 1 <= story.complexity_points <= 8
    assert 0.0 <= story.ambiguity <= 1.0


def test_generate_with_params() -> None:
    gen = BacklogGenerator(seed=42)
    story = gen.generate_story(
        domain="api",
        language="python",
        ambiguity=0.1,
        complexity=5,
        codebase_context="greenfield",
    )
    assert story.domain == "api"
    assert story.language == "python"
    assert story.ambiguity == 0.1
    assert story.complexity_points == 5
    assert story.codebase_context == "greenfield"


def test_high_ambiguity_short_description() -> None:
    gen = BacklogGenerator(seed=42)
    story = gen.generate_story(ambiguity=0.9)
    # High ambiguity: description should be very short (just the title)
    assert len(story.description) < 100


def test_low_ambiguity_detailed_description() -> None:
    gen = BacklogGenerator(seed=42)
    story = gen.generate_story(ambiguity=0.1)
    # Low ambiguity: should have detailed description
    assert len(story.description) > 100


def test_acceptance_criteria_vary_with_ambiguity() -> None:
    gen = BacklogGenerator(seed=42)
    clear = gen.generate_story(ambiguity=0.1)
    vague = gen.generate_story(ambiguity=0.9)
    assert len(clear.acceptance_criteria) >= len(vague.acceptance_criteria)


def test_hidden_constraints_for_complex_stories() -> None:
    gen = BacklogGenerator(seed=42)
    complex_story = gen.generate_story(complexity=8)
    simple_story = gen.generate_story(complexity=1)
    assert len(complex_story.hidden_constraints) >= len(simple_story.hidden_constraints)


def test_generate_backlog() -> None:
    gen = BacklogGenerator(seed=42)
    backlog = gen.generate_backlog(num_stories=10)
    assert len(backlog) == 10
    # Should have variety
    domains = {s.domain for s in backlog}
    assert len(domains) > 1


def test_to_dict() -> None:
    gen = BacklogGenerator(seed=42)
    story = gen.generate_story()
    d = story.to_dict()
    assert "title" in d
    assert "description" in d
    assert "domain" in d
    assert "language" in d
    assert "acceptance_criteria" in d
    # Hidden constraints should NOT be in dict (not visible to model)
    assert "hidden_constraints" not in d


def test_all_domains_have_templates() -> None:
    gen = BacklogGenerator(seed=42)
    for domain in DOMAINS:
        story = gen.generate_story(domain=domain)
        assert len(story.title) > 0


def test_reproducible_with_seed() -> None:
    gen1 = BacklogGenerator(seed=42)
    gen2 = BacklogGenerator(seed=42)
    story1 = gen1.generate_story()
    story2 = gen2.generate_story()
    assert story1.title == story2.title
    assert story1.domain == story2.domain
