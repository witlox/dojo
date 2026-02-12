"""Synthetic backlog generator for training episodes."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DOMAINS = ["api", "data_pipeline", "frontend", "cli", "library", "infrastructure"]
LANGUAGES = ["python", "typescript", "go", "rust"]
CODEBASE_CONTEXTS = ["greenfield", "small_brownfield", "large_brownfield"]

# Story templates per domain
_STORY_TEMPLATES: Dict[str, List[str]] = {
    "api": [
        "Add {operation} endpoint for {resource}",
        "Implement pagination for {resource} list endpoint",
        "Add authentication to {resource} API",
        "Create webhook handler for {event} events",
        "Add rate limiting to {resource} endpoints",
    ],
    "data_pipeline": [
        "Build ETL pipeline for {source} data",
        "Add data validation for {resource} ingestion",
        "Implement retry logic for {source} connector",
        "Create scheduled job for {operation} processing",
        "Add monitoring for {resource} pipeline",
    ],
    "frontend": [
        "Create {component} component with {feature}",
        "Add form validation for {resource} creation",
        "Implement {feature} for the dashboard",
        "Build responsive layout for {page} page",
        "Add loading states and error handling to {component}",
    ],
    "cli": [
        "Add {command} command to CLI",
        "Implement {feature} flag for {command}",
        "Create interactive {operation} wizard",
        "Add output formatting for {command} results",
        "Implement config file support for {feature}",
    ],
    "library": [
        "Create {feature} module with public API",
        "Add type-safe builder for {resource} configuration",
        "Implement {pattern} pattern for {resource}",
        "Add serialization support for {resource}",
        "Create plugin system for {feature} extensions",
    ],
    "infrastructure": [
        "Set up {service} deployment configuration",
        "Add health check endpoint for {service}",
        "Implement graceful shutdown for {service}",
        "Create Docker configuration for {service}",
        "Add logging and metrics for {component}",
    ],
}

_RESOURCES = ["users", "orders", "products", "sessions", "reports", "tasks", "events", "configs"]
_OPERATIONS = ["create", "update", "delete", "export", "import", "sync", "validate", "migrate"]
_COMPONENTS = ["table", "form", "modal", "sidebar", "header", "card", "list", "search"]
_FEATURES = ["filtering", "sorting", "caching", "logging", "auth", "search", "notifications"]
_SERVICES = ["api-gateway", "worker", "scheduler", "auth-service", "notification-service"]
_PATTERNS = ["repository", "factory", "observer", "strategy", "decorator", "adapter"]


@dataclass
class BacklogStory:
    """A single backlog story for training episodes."""

    title: str
    description: str
    domain: str
    language: str
    complexity_points: int
    ambiguity: float  # 0.0 (fully specified) to 1.0 (one-line description)
    codebase_context: str
    acceptance_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    hidden_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "language": self.language,
            "complexity_points": self.complexity_points,
            "ambiguity": self.ambiguity,
            "codebase_context": self.codebase_context,
            "acceptance_criteria": self.acceptance_criteria,
            "constraints": self.constraints,
        }


class BacklogGenerator:
    """Generates diverse synthetic backlogs for training episodes.

    Varies:
    - Domain (6 types)
    - Language (4)
    - Ambiguity (0.1 to 0.9)
    - Complexity (1-8 points)
    - Codebase context (greenfield/brownfield)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def generate_story(
        self,
        domain: Optional[str] = None,
        language: Optional[str] = None,
        ambiguity: Optional[float] = None,
        complexity: Optional[int] = None,
        codebase_context: Optional[str] = None,
    ) -> BacklogStory:
        """Generate a single backlog story.

        Args:
            domain: Domain type (random if None).
            language: Programming language (random if None).
            ambiguity: Ambiguity level 0-1 (random if None).
            complexity: Story points 1-8 (random if None).
            codebase_context: Codebase context (random if None).

        Returns:
            BacklogStory with generated content.
        """
        domain = domain or self._rng.choice(DOMAINS)
        language = language or self._rng.choice(LANGUAGES)
        ambiguity = ambiguity if ambiguity is not None else self._rng.uniform(0.1, 0.9)
        complexity = complexity or self._rng.choice([1, 2, 3, 5, 8])
        codebase_context = codebase_context or self._rng.choice(CODEBASE_CONTEXTS)

        # Generate title from templates
        templates = _STORY_TEMPLATES.get(domain, _STORY_TEMPLATES["api"])
        template = self._rng.choice(templates)
        title = self._fill_template(template)

        # Generate description based on ambiguity
        description = self._generate_description(title, ambiguity, domain)

        # Generate acceptance criteria (fewer for high ambiguity)
        num_criteria = max(1, int(5 * (1 - ambiguity)))
        criteria = self._generate_criteria(title, domain, num_criteria)

        # Generate constraints
        constraints = self._generate_constraints(domain, complexity)
        hidden = self._generate_hidden_constraints(domain, complexity)

        return BacklogStory(
            title=title,
            description=description,
            domain=domain,
            language=language,
            complexity_points=complexity,
            ambiguity=ambiguity,
            codebase_context=codebase_context,
            acceptance_criteria=criteria,
            constraints=constraints,
            hidden_constraints=hidden,
        )

    def generate_backlog(
        self,
        num_stories: int = 5,
        **kwargs: Any,
    ) -> List[BacklogStory]:
        """Generate a complete backlog with multiple stories.

        Args:
            num_stories: Number of stories to generate.
            **kwargs: Passed to generate_story for each story.

        Returns:
            List of BacklogStory instances.
        """
        return [self.generate_story(**kwargs) for _ in range(num_stories)]

    def _fill_template(self, template: str) -> str:
        """Fill a story template with random values."""
        replacements = {
            "{resource}": self._rng.choice(_RESOURCES),
            "{operation}": self._rng.choice(_OPERATIONS),
            "{component}": self._rng.choice(_COMPONENTS),
            "{feature}": self._rng.choice(_FEATURES),
            "{service}": self._rng.choice(_SERVICES),
            "{pattern}": self._rng.choice(_PATTERNS),
            "{event}": self._rng.choice(_RESOURCES),
            "{source}": self._rng.choice(["postgres", "kafka", "s3", "redis", "api"]),
            "{command}": self._rng.choice(["init", "deploy", "test", "lint", "build"]),
            "{page}": self._rng.choice(["dashboard", "settings", "profile", "admin"]),
        }
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        return result

    def _generate_description(self, title: str, ambiguity: float, domain: str) -> str:
        """Generate story description based on ambiguity level."""
        if ambiguity > 0.8:
            return title  # One-liner
        if ambiguity > 0.6:
            return f"{title}. Should work with existing infrastructure."
        if ambiguity > 0.4:
            return (
                f"{title}.\n\nThis should integrate with the existing {domain} "
                f"infrastructure. Consider error handling and edge cases."
            )
        return (
            f"{title}.\n\nDetailed requirements:\n"
            f"- Must integrate with existing {domain} components\n"
            f"- Include proper error handling and validation\n"
            f"- Add comprehensive tests\n"
            f"- Follow existing code conventions\n"
            f"- Document public API"
        )

    def _generate_criteria(self, title: str, domain: str, count: int) -> List[str]:
        """Generate acceptance criteria."""
        criteria_pool = [
            "All tests pass",
            "Code follows existing conventions",
            "Error cases handled gracefully",
            "Performance within acceptable bounds",
            "Documentation updated",
            "No regressions in existing tests",
            f"Integration with existing {domain} components works",
            "Edge cases covered",
        ]
        return self._rng.sample(criteria_pool, min(count, len(criteria_pool)))

    def _generate_constraints(self, domain: str, complexity: int) -> List[str]:
        """Generate visible constraints."""
        constraint_pool = [
            "Must be backwards-compatible",
            "No new dependencies allowed",
            f"Response time < {self._rng.choice([100, 200, 500])}ms",
            "Must work with existing auth system",
            f"Max {self._rng.choice([50, 100, 200])} lines of new code",
        ]
        num = min(max(1, complexity // 2), len(constraint_pool))
        return self._rng.sample(constraint_pool, num)

    def _generate_hidden_constraints(self, domain: str, complexity: int) -> List[str]:
        """Generate hidden constraints (not visible to model, used for evaluation)."""
        hidden_pool = [
            "Existing endpoint uses different naming convention",
            "Database has an undocumented column constraint",
            "An existing test depends on the current behavior",
            "The deployment environment has limited memory",
            "Another service polls this endpoint every 5 seconds",
        ]
        if complexity >= 5:
            return self._rng.sample(hidden_pool, min(2, len(hidden_pool)))
        return []
