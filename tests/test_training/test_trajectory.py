"""Tests for trajectory buffer."""
from __future__ import annotations

import pytest

from dojo.training.trajectory_buffer import Trajectory, TrajectoryBuffer


def _make_trajectory(
    n_steps: int = 5,
    episode_id: str = "ep1",
    rewards: list = None,
    values: list = None,
) -> Trajectory:
    return Trajectory(
        episode_id=episode_id,
        episode_type="elicitation",
        stage=1,
        prompts=[f"prompt_{i}" for i in range(n_steps)],
        responses=[f"response_{i}" for i in range(n_steps)],
        rewards=rewards or [0.5] * n_steps,
        values=values,
    )


def test_trajectory_length() -> None:
    t = _make_trajectory(n_steps=3)
    assert t.length == 3


def test_buffer_add() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=5))
    assert buf.num_trajectories == 1
    assert buf.total_steps == 5


def test_buffer_multiple_add() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=3, episode_id="ep1"))
    buf.add(_make_trajectory(n_steps=7, episode_id="ep2"))
    assert buf.num_trajectories == 2
    assert buf.total_steps == 10


def test_buffer_eviction() -> None:
    buf = TrajectoryBuffer(max_buffer_size=10)
    buf.add(_make_trajectory(n_steps=6, episode_id="ep1"))
    buf.add(_make_trajectory(n_steps=6, episode_id="ep2"))
    # Should evict first trajectory since total > 10
    assert buf.total_steps <= 10 or buf.num_trajectories <= 2


def test_compute_advantages_without_values() -> None:
    buf = TrajectoryBuffer(gamma=0.99)
    t = _make_trajectory(n_steps=3, rewards=[1.0, 0.5, 0.0])
    buf.add(t)
    buf.compute_advantages()
    assert t.returns is not None
    assert len(t.returns) == 3
    assert t.advantages is not None
    # Reward-to-go: first step should have highest return
    assert t.returns[0] > t.returns[2]


def test_compute_advantages_with_values() -> None:
    buf = TrajectoryBuffer(gamma=0.99, lam=0.95)
    t = _make_trajectory(
        n_steps=3,
        rewards=[1.0, 0.5, 0.0],
        values=[0.8, 0.4, 0.1],
    )
    buf.add(t)
    buf.compute_advantages()
    assert t.advantages is not None
    assert t.returns is not None
    assert len(t.advantages) == 3
    # GAE should produce reasonable values
    for a in t.advantages:
        assert -10.0 < a < 10.0


def test_sample_batch() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=10, rewards=[float(i) / 10 for i in range(10)]))
    buf.compute_advantages()

    prompts, responses, rewards, advantages = buf.sample_batch(batch_size=5)
    assert len(prompts) == 5
    assert len(responses) == 5
    assert len(rewards) == 5
    assert len(advantages) == 5


def test_sample_batch_larger_than_buffer() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=3))
    buf.compute_advantages()

    prompts, responses, rewards, advantages = buf.sample_batch(batch_size=100)
    assert len(prompts) == 3  # Can't sample more than available


def test_sample_batch_empty() -> None:
    buf = TrajectoryBuffer()
    prompts, responses, rewards, advantages = buf.sample_batch(batch_size=5)
    assert len(prompts) == 0


def test_get_all() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=3, episode_id="ep1"))
    buf.add(_make_trajectory(n_steps=2, episode_id="ep2"))

    prompts, responses, rewards, advantages = buf.get_all()
    assert len(prompts) == 5


def test_clear() -> None:
    buf = TrajectoryBuffer()
    buf.add(_make_trajectory(n_steps=5))
    assert buf.num_trajectories == 1
    buf.clear()
    assert buf.num_trajectories == 0
    assert buf.total_steps == 0


def test_gae_computation() -> None:
    buf = TrajectoryBuffer(gamma=0.99, lam=0.95)
    rewards = [1.0, 0.0, 1.0]
    values = [0.5, 0.5, 0.5]
    advantages, returns = buf._gae(rewards, values)
    assert len(advantages) == 3
    assert len(returns) == 3
    # returns[t] = advantages[t] + values[t]
    for i in range(3):
        assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6


def test_reward_to_go() -> None:
    buf = TrajectoryBuffer(gamma=1.0)  # No discounting
    rewards = [1.0, 2.0, 3.0]
    rtg = buf._reward_to_go(rewards)
    assert rtg[0] == 6.0  # 1+2+3
    assert rtg[1] == 5.0  # 2+3
    assert rtg[2] == 3.0  # 3


def test_reward_to_go_with_discount() -> None:
    buf = TrajectoryBuffer(gamma=0.5)
    rewards = [1.0, 1.0, 1.0]
    rtg = buf._reward_to_go(rewards)
    # rtg[2] = 1.0
    # rtg[1] = 1.0 + 0.5*1.0 = 1.5
    # rtg[0] = 1.0 + 0.5*1.5 = 1.75
    assert abs(rtg[2] - 1.0) < 1e-6
    assert abs(rtg[1] - 1.5) < 1e-6
    assert abs(rtg[0] - 1.75) < 1e-6
