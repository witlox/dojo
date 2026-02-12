"""CLI entry point for dojo."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dojo",
        description="Senior Meta-Cognitive Model Trainer",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run full training pipeline")
    train_parser.add_argument("--base-model", required=True, help="Base model path or HF name")
    train_parser.add_argument("--output", required=True, help="Output directory")
    train_parser.add_argument("--stages", default="1,2,3,4", help="Comma-separated stage numbers")
    train_parser.add_argument("--judge-model", default="claude-opus-4-6", help="Judge model")
    train_parser.add_argument("--resume", help="Resume from checkpoint directory")
    train_parser.add_argument("--mock", action="store_true", help="Use mock mode (no LLM)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", required=True, help="Model path")
    eval_parser.add_argument("--adapter", help="LoRA adapter path")
    eval_parser.add_argument("--tasks", help="Task bank directory")
    eval_parser.add_argument("--output", required=True, help="Output directory")
    eval_parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks")

    # Episode command
    episode_parser = subparsers.add_parser("episode", help="Run a single episode for debugging")
    episode_parser.add_argument("--type", default="elicitation", help="Episode type")
    episode_parser.add_argument("--stage", type=int, default=1, help="Curriculum stage")
    episode_parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty 0-1")
    episode_parser.add_argument("--model", help="Model path")
    episode_parser.add_argument("--mock", action="store_true", help="Use mock mode")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        return _cmd_train(args)
    elif args.command == "evaluate":
        return _cmd_evaluate(args)
    elif args.command == "episode":
        return _cmd_episode(args)
    else:
        parser.print_help()
        return 1


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the full training pipeline."""
    from dojo.orchestrator import OrchestratorConfig, TrainingOrchestrator

    stages = [int(s) for s in args.stages.split(",")]

    config = OrchestratorConfig(
        base_model=args.base_model,
        output_dir=args.output,
        stages=stages,
        judge_model=args.judge_model,
        mock_mode=getattr(args, "mock", False),
    )

    orchestrator = TrainingOrchestrator(config)

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(orchestrator.run())
    finally:
        loop.close()

    # Write results
    output_path = Path(args.output) / "training_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logging.info("Training complete. Results: %s", output_path)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run standalone evaluation."""
    from dojo.eval.runner import EvalConfig, EvalRunner

    config = EvalConfig(comprehensive_num_tasks=args.num_tasks)
    runner = EvalRunner(config=config)

    if args.tasks:
        runner.load_task_bank(args.tasks)

    summary = runner.comprehensive_eval()

    output_path = Path(args.output) / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "num_tasks": summary.num_tasks,
            "primary_metrics": summary.primary.to_dict(),
            "mean_transfer": summary.mean_transfer,
            "per_task_scores": summary.per_task_scores,
        }, f, indent=2)

    logging.info("Evaluation complete. Results: %s", output_path)
    logging.info("Primary metrics mean: %.3f", summary.primary.mean())
    return 0


def _cmd_episode(args: argparse.Namespace) -> int:
    """Run a single episode for debugging."""
    import os

    from dojo.orchestrator import OrchestratorConfig, TrainingOrchestrator

    if args.mock:
        os.environ["MOCK_LLM"] = "true"

    config = OrchestratorConfig(mock_mode=True)
    orchestrator = TrainingOrchestrator(config)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            orchestrator.run_single_episode(
                episode_type=args.type,
                stage=args.stage,
                difficulty=args.difficulty,
                verbose=True,
            )
        )
    finally:
        loop.close()

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
