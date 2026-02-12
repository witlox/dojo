"""Trajectory buffer for collecting and sampling RL training data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Trajectory:
    """Single episode trajectory for RL training."""

    episode_id: str
    episode_type: str
    stage: int
    prompts: List[str]
    responses: List[str]
    rewards: List[float]
    log_probs: Optional[List[float]] = None
    values: Optional[List[float]] = None
    advantages: Optional[List[float]] = None
    returns: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.prompts)


class TrajectoryBuffer:
    """Collects episode trajectories and supports batched sampling for PPO.

    Handles:
    - Adding complete episode trajectories
    - Computing Generalized Advantage Estimation (GAE)
    - Batched sampling for PPO updates
    - Buffer management (clear after update)
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_buffer_size: int = 10000,
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.max_buffer_size = max_buffer_size
        self._trajectories: List[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        """Add a completed trajectory to the buffer."""
        self._trajectories.append(trajectory)

        # Evict oldest if over capacity
        while self.total_steps > self.max_buffer_size and len(self._trajectories) > 1:
            self._trajectories.pop(0)

    @property
    def total_steps(self) -> int:
        """Total number of prompt/response steps across all trajectories."""
        return sum(t.length for t in self._trajectories)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    def compute_advantages(self) -> None:
        """Compute GAE advantages for all trajectories in the buffer.

        Uses Generalized Advantage Estimation with parameters gamma and lam.
        If values are not provided, uses rewards directly as advantages.
        """
        for traj in self._trajectories:
            if traj.values is not None and len(traj.values) == traj.length:
                traj.advantages, traj.returns = self._gae(
                    traj.rewards, traj.values
                )
            else:
                # No value estimates — use simple reward-to-go
                traj.returns = self._reward_to_go(traj.rewards)
                traj.advantages = list(traj.returns)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[List[str], List[str], List[float], List[float]]:
        """Sample a batch of (prompts, responses, rewards, advantages).

        Args:
            batch_size: Number of prompt/response pairs to sample.

        Returns:
            Tuple of (prompts, responses, rewards, advantages).
        """
        all_prompts: List[str] = []
        all_responses: List[str] = []
        all_rewards: List[float] = []
        all_advantages: List[float] = []

        for traj in self._trajectories:
            for i in range(traj.length):
                all_prompts.append(traj.prompts[i])
                all_responses.append(traj.responses[i])
                all_rewards.append(traj.rewards[i])
                adv = traj.advantages[i] if traj.advantages else traj.rewards[i]
                all_advantages.append(adv)

        n = len(all_prompts)
        if n == 0:
            return [], [], [], []

        # Random sample (with replacement if batch_size > n)
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)

        return (
            [all_prompts[i] for i in indices],
            [all_responses[i] for i in indices],
            [all_rewards[i] for i in indices],
            [all_advantages[i] for i in indices],
        )

    def get_all(self) -> Tuple[List[str], List[str], List[float], List[float]]:
        """Get all data in the buffer (no sampling)."""
        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        advantages: List[float] = []

        for traj in self._trajectories:
            for i in range(traj.length):
                prompts.append(traj.prompts[i])
                responses.append(traj.responses[i])
                rewards.append(traj.rewards[i])
                adv = traj.advantages[i] if traj.advantages else traj.rewards[i]
                advantages.append(adv)

        return prompts, responses, rewards, advantages

    def clear(self) -> None:
        """Clear the buffer after a PPO update."""
        self._trajectories.clear()

    def _gae(
        self, rewards: List[float], values: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Per-step rewards.
            values: Per-step value estimates from value head.

        Returns:
            Tuple of (advantages, returns).
        """
        n = len(rewards)
        advantages = [0.0] * n
        returns = [0.0] * n

        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _reward_to_go(self, rewards: List[float]) -> List[float]:
        """Compute discounted reward-to-go."""
        n = len(rewards)
        rtg = [0.0] * n
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t] + self.gamma * running
            rtg[t] = running
        return rtg
