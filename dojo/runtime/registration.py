"""Register the training candidate runtime with AAT's factory."""
from __future__ import annotations

from typing import Any, Dict, List


def register_candidate_runtime() -> None:
    """Register TrainingCandidateRuntime with AAT's runtime factory.

    After calling this, AAT will accept "training_candidate" as a
    runtime type in ExperimentConfigBuilder and agent configs.
    """
    from src.rl import register_runtime

    from dojo.runtime.candidate_runtime import TrainingCandidateRuntime

    def factory(config: Dict[str, Any], tools: List[Any]) -> TrainingCandidateRuntime:
        return TrainingCandidateRuntime(config, tools)

    register_runtime("training_candidate", factory)
