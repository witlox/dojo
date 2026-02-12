"""Training orchestrator — end-to-end training pipeline."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.env.aat_env import AATEnv, EpisodeConfig
from dojo.eval.runner import EvalConfig, EvalRunner
from dojo.reward.calibration import CalibrationMonitor
from dojo.reward.composite_reward import CompositeRewardCalculator
from dojo.reward.judge_evaluator import JudgeEvaluator
from dojo.training.curriculum import CurriculumManager
from dojo.training.ppo_trainer import DojoTrainer, TrainerConfig
from dojo.training.trajectory_buffer import Trajectory, TrajectoryBuffer

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the training orchestrator."""

    base_model: str = "deepseek-ai/deepseek-coder-v2-lite-instruct"
    output_dir: str = "/tmp/dojo-output"
    stages: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    judge_model: str = "claude-opus-4-6"
    judge_every_n: int = 50
    eval_quick_every_n: int = 100
    eval_full_every_n: int = 500
    ppo_update_every_n: int = 16
    checkpoint_every_n: int = 100
    workspace_root: str = "/tmp/dojo-episodes"
    mock_mode: bool = False


class TrainingOrchestrator:
    """Coordinates all components for the full training pipeline.

    Manages:
    - AATEnv for episode execution
    - CompositeRewardCalculator for reward computation
    - DojoTrainer for PPO updates
    - CurriculumManager for stage progression
    - EvalRunner for periodic evaluation
    - JudgeEvaluator for periodic behavioral scoring
    - CalibrationMonitor for reward drift detection
    - TrajectoryBuffer for episode data collection
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.curriculum = CurriculumManager(
            start_stage=self.config.stages[0],
            stages=self.config.stages,
        )
        self.reward_calc = CompositeRewardCalculator(stage=self.curriculum.current_stage)
        self.judge = JudgeEvaluator(
            model=self.config.judge_model,
            evaluate_every_n=self.config.judge_every_n,
        )
        self.calibration = CalibrationMonitor()
        self.buffer = TrajectoryBuffer()
        self.eval_runner = EvalRunner(config=EvalConfig())
        self.trainer: Optional[DojoTrainer] = None

        # Metrics tracking
        self._episode_rewards: List[float] = []
        self._behavioral_scores: List[float] = []
        self._training_log: List[Dict[str, Any]] = []

    async def run(self) -> Dict[str, Any]:
        """Run the full training pipeline across all stages.

        Returns:
            Summary dict with training results.
        """
        logger.info(
            "Starting training: stages=%s, model=%s",
            self.config.stages, self.config.base_model,
        )

        results: Dict[str, Any] = {}
        for stage in self.config.stages:
            stage_result = await self._run_stage(stage)
            results[f"stage_{stage}"] = stage_result

            if not self.curriculum.check_graduation():
                logger.warning("Stage %d graduation criteria not met", stage)

            if stage != self.config.stages[-1]:
                self.curriculum.advance_stage()

        # Final comprehensive evaluation
        final_eval = self.eval_runner.comprehensive_eval()
        results["final_eval"] = {
            "num_tasks": final_eval.num_tasks,
            "primary_mean": final_eval.primary.mean(),
            "mean_transfer": final_eval.mean_transfer,
        }

        results["total_episodes"] = self.curriculum.episode_count
        results["calibration"] = {
            "aligned_rate": self.calibration.report().aligned_rate,
            "performative_rate": self.calibration.report().performative_rate,
        }

        return results

    async def _run_stage(self, stage: int) -> Dict[str, Any]:
        """Run training for a single curriculum stage."""
        logger.info("Starting stage %d", stage)
        start_time = time.monotonic()

        self.reward_calc.set_stage(stage)
        self.judge.reset_counter()

        env = AATEnv(
            episode_config=EpisodeConfig(stage=stage),
            workspace_root=self.config.workspace_root,
        )

        episode_count = 0
        stage_rewards: List[float] = []

        while not self.curriculum.check_graduation():
            # Generate batch of episodes
            batch = self.curriculum.generate_batch(batch_size=self.config.ppo_update_every_n)

            for scenario in batch:
                # Run episode
                episode_result = await self._run_episode(env, scenario)
                episode_count += 1
                self.curriculum.record_episode()
                stage_rewards.append(episode_result.get("reward", 0.0))

                # Periodic evaluations
                if episode_count % self.config.eval_quick_every_n == 0:
                    eval_summary = self.eval_runner.quick_eval()
                    logger.info(
                        "Quick eval at ep %d: mean=%.3f",
                        episode_count, eval_summary.primary.mean(),
                    )

                # PPO update
                if episode_count % self.config.ppo_update_every_n == 0:
                    self._do_ppo_update()

            # Safety limit
            if episode_count > 10000:
                logger.warning("Stage %d: safety limit reached", stage)
                break

        # Stage completion
        duration = time.monotonic() - start_time
        logger.info("Stage %d complete: %d episodes in %.1fs", stage, episode_count, duration)

        # Save adapter
        adapter_path = None
        if self.trainer and self.trainer._initialized:
            adapter_path = str(self.trainer.save_adapter(f"lora-v{stage}"))

        return {
            "episodes": episode_count,
            "duration_seconds": duration,
            "mean_reward": sum(stage_rewards) / max(len(stage_rewards), 1),
            "adapter_path": adapter_path,
        }

    async def _run_episode(
        self, env: AATEnv, scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single training episode and collect data."""
        episode_type = scenario.get("episode_type", "elicitation")
        difficulty = scenario.get("difficulty", 0.5)

        obs, info = await env.reset_async(
            options={"episode_type": episode_type, "difficulty": difficulty}
        )

        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        total_reward = 0.0
        behavioral_score = 0.0

        # Phase-level episode loop
        while True:
            obs, reward, terminated, truncated, step_info = await env.step_async(0)
            total_reward += reward
            rewards.append(reward)
            behavioral_score = step_info.get("behavioral_score", 0.0)

            if terminated or truncated:
                break

        # Record calibration
        self.calibration.record(
            episode_id=f"ep-{self.curriculum.episode_count}",
            episode_type=episode_type,
            outcome_score=total_reward,
            heuristic_score=behavioral_score,
        )

        # Record metrics for graduation
        self.curriculum.record_metric(f"{episode_type}_quality", behavioral_score)
        self._episode_rewards.append(total_reward)
        self._behavioral_scores.append(behavioral_score)

        # Add to trajectory buffer
        if prompts or rewards:
            self.buffer.add(Trajectory(
                episode_id=f"ep-{self.curriculum.episode_count}",
                episode_type=episode_type,
                stage=self.curriculum.current_stage,
                prompts=prompts or [""],
                responses=responses or [""],
                rewards=rewards or [total_reward],
            ))

        return {
            "reward": total_reward,
            "behavioral_score": behavioral_score,
            "episode_type": episode_type,
        }

    def _do_ppo_update(self) -> None:
        """Perform a PPO training update from the trajectory buffer."""
        if self.buffer.total_steps == 0:
            return

        self.buffer.compute_advantages()
        prompts, responses, rewards, advantages = self.buffer.get_all()

        if not prompts:
            return

        # Log but don't actually train without GPU
        logger.info(
            "PPO update: %d steps, mean reward=%.3f",
            len(prompts),
            sum(rewards) / max(len(rewards), 1),
        )

        # If trainer is initialized, do actual update
        if self.trainer and self.trainer._initialized:
            result = self.trainer.train_step(prompts, responses, rewards)
            self._training_log.append({
                "step": self.trainer.step_count,
                "loss": result.loss,
                "num_samples": result.num_samples,
            })

        self.buffer.clear()

    async def run_single_episode(
        self,
        episode_type: str = "elicitation",
        stage: int = 1,
        difficulty: float = 0.5,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode for debugging.

        Returns:
            Episode result dict.
        """
        env = AATEnv(
            episode_config=EpisodeConfig(
                stage=stage,
                episode_type=episode_type,
                difficulty=difficulty,
            ),
            workspace_root=self.config.workspace_root,
        )

        result = await self._run_episode(env, {
            "episode_type": episode_type,
            "difficulty": difficulty,
        })

        if verbose:
            logger.info("Episode result: %s", result)

        return result
