# Dojo: Senior Meta-Cognitive Model Trainer

## Project Overview

Dojo trains a local LLM to exhibit senior-engineer behavioral patterns using reinforcement learning. It does NOT train coding skills — it trains the meta-cognitive layer: knowing what to ask, when to research, how to decompose ambiguity, when to stop and rethink, and how to adapt when things go wrong.

The training environment is [agile-agent-team](https://github.com/witlox/agile-agent-team) (AAT), a multi-agent software development simulation. AAT provides a public `src.rl` API with 28 exported symbols for RL integration. Dojo wraps this as a `gym.Env` and adds the reward attribution pipeline, training loop (PPO + LoRA), and evaluation harness.

**Status**: Ready for implementation. All design docs and specs are complete. AAT's RL API (Phases A-C) is fully implemented.

## Architecture

Four components:

1. **`AATEnv(gym.Env)`** — Gym wrapper around AAT's `EpisodeRunner`, `PhaseRunner`, `ObservationExtractor`, `ActionExecutor`, and `CheckpointManager`
2. **Reward Attribution Pipeline** — Combines AAT's `RewardCalculator` + `BehavioralScorer` (fast heuristic) with a Claude Opus judge evaluator (nuanced behavioral quality)
3. **Training Loop** — PPO with LoRA/QLoRA adapters across 4 curriculum stages, producing stage-specific checkpoints
4. **Evaluation Harness** — Solo deployment testing (model + human proxy) with 5-phase protocol and transfer scoring

See `docs/ARCHITECTURE.md` for full details.

## Development Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for training and local model inference)
- [agile-agent-team](https://github.com/witlox/agile-agent-team) cloned and installable
- Anthropic API key (for judge evaluator — Claude Opus)
- vLLM (for local model serving during training episodes)

### Environment Setup

```bash
git clone https://github.com/witlox/dojo
cd dojo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install agile-agent-team as a dependency (for src.rl imports)
pip install -e /path/to/agile-agent-team

# Verify AAT RL API is accessible
python -c "from src.rl import EpisodeRunner, BehavioralScorer; print('AAT RL API OK')"
```

### Configuration

```bash
# Anthropic API key for judge evaluator
export ANTHROPIC_API_KEY="sk-ant-..."

# vLLM endpoint for environment agents (if using local models)
export VLLM_ENDPOINT="http://localhost:8000"

# Training candidate model path
export CANDIDATE_MODEL="/path/to/base-model.gguf"
```

## Key Concepts

### Behavioral Taxonomy (30 Patterns)

Defined in `specs/BEHAVIORAL_TAXONOMY.md`. Ten categories:

| Category | Codes | Key Behaviors |
|---|---|---|
| Elicitation | B-01..B-04 | Ambiguity detection, scope probing, constraint discovery, sufficiency |
| Decomposition | B-05..B-08 | Risk-first ordering, dependency ID, granularity, trade-offs |
| Self-Monitoring | B-09..B-12 | Progress calibration, stuck detection, error diagnosis, knowing limits |
| Adaptation | B-13..B-15 | Severity triage, graceful replanning, post-incident learning |
| Research | B-16..B-17 | Search strategy, source evaluation |
| Orientation | B-18..B-20 | Codebase recon, convention respect, context acquisition |
| Communication | B-21..B-22 | Status transparency, help request quality |
| Knowledge Transfer | B-23..B-24 | Decision documentation, knowledge gap handoff |
| Scoping | B-25..B-26 | YAGNI application, MVP instinct |
| Meta-Learning | B-27..B-30 | Pattern recognition, mistake non-repetition, feedback integration, confidence calibration |

24 high-transfer, 6 medium-transfer, 1 low-transfer to solo deployment. See `specs/TRANSFER_ANALYSIS.md`.

### Training Curriculum (4 Stages)

| Stage | Environment | Focus | Episodes | Output |
|---|---|---|---|---|
| 1. Stable Team | Single team, no disruptions | Elicitation, decomposition, self-monitoring | ~5000 | LoRA v1 |
| 2. Disturbances | Single team + chaos | Triage, replanning, recovery | ~1200 | LoRA v2 |
| 3. Cross-Team | Multi-team + borrowing | Orientation, context-switching | ~700 | LoRA v3 |
| 4. Team Change | Attrition + onboarding | Knowledge transfer, compensation | ~700 | LoRA v4 |

Total: ~7600 episodes + ~100 full-sprint evaluations. See `docs/TRAINING_PIPELINE.md`.

### Episode Types (13 Total)

| ID | Type | Stage | Duration | Primary Behaviors |
|---|---|---|---|---|
| E-1.1 | Elicitation | 1 | ~2 min | B-01..B-04 |
| E-1.2 | Decomposition | 1 | ~3 min | B-05..B-08 |
| E-1.3 | Implementation | 1 | ~10 min | B-09..B-11, B-25..B-26 |
| E-1.4 | Self-Monitoring | 1 | ~3 min | B-09, B-10, B-12, B-30 |
| E-1.5 | Research | 1 | ~3 min | B-16, B-17, B-04 |
| E-2.1 | Triage | 2 | ~3 min | B-13, B-14, B-21 |
| E-2.2 | Recovery | 2 | ~5 min | B-10, B-11, B-14, B-15 |
| E-2.3 | Scope Change | 2 | ~3 min | B-02, B-13, B-14, B-25 |
| E-3.1 | Borrowing Arrival | 3 | ~5 min | B-18..B-20, B-22 |
| E-3.2 | Cross-Team Dep | 3 | ~3 min | B-06, B-03, B-21 |
| E-4.1 | Knowledge Handoff | 4 | ~5 min | B-23, B-24 |
| E-4.2 | Onboarding Support | 4 | ~5 min | B-20, B-07, B-22 |
| E-4.3 | Compensation | 4 | ~5 min | B-10, B-12, B-13, B-14 |

See `specs/TRAINING_EPISODES.md`.

### Reward Function

```python
composite_reward = (
    outcome_weight     * outcome_reward +      # From AAT RewardCalculator
    behavioral_weight  * behavioral_reward +    # From BehavioralScorer + Judge
    efficiency_penalty +                        # Penalizes unnecessary actions
    phase_bonus                                 # Sparse bonus for correct transitions
)
```

Weight schedule shifts from behavioral-heavy (Stage 1: 70/30) to outcome-heavy (Stage 4: 40/60). See `docs/REWARD_FUNCTION.md` and `specs/REWARD_SIGNALS.md`.

## Implementation Plan

### Phase 1: Core Infrastructure

Build the foundational components that everything else depends on.

#### 1.1 AAT Gym Wrapper (`src/env/aat_env.py`)

```python
# Key class: AATEnv(gym.Env)
# Wraps: EpisodeRunner, PhaseRunner, ObservationExtractor, ActionExecutor
# Provides: reset(), step(), render(), close()
# Action space: Built from ACTION_SPACE_SPEC
# Observation space: Built from ObservationExtractor output schema
```

Implementation steps:
- Import AAT's `src.rl` API
- Implement `gym.Env` interface wrapping `EpisodeRunner.run_episode()` and `PhaseRunner.run_phase()`
- Build action/observation spaces from `ACTION_SPACE_SPEC` and `ObservationExtractor`
- Register custom training candidate runtime via `register_runtime()`
- Handle model injection via `ExperimentConfigBuilder`
- Support both episode-level (full episodes) and phase-level (individual ceremonies) modes

#### 1.2 Training Candidate Runtime (`src/runtime/candidate_runtime.py`)

Custom AAT runtime that wraps the training candidate model:
- Implements AAT's runtime interface (same as `vllm_runtime` or `anthropic_runtime`)
- Forwards inference calls to the candidate model (local vLLM or direct)
- Intercepts all model inputs/outputs for decision tracing
- Supports LoRA adapter hot-swapping between episodes

#### 1.3 Backlog Generator (`src/data/backlog_generator.py`)

Generates diverse synthetic backlogs for training episodes:
- Varies domain (API, data pipeline, frontend, CLI, library, infra)
- Varies language (Python, TypeScript, Go, Rust)
- Varies ambiguity (0.1 fully specified to 0.9 one-line description)
- Varies codebase context (greenfield, small brownfield, large brownfield)
- Outputs AAT-compatible `backlog.yaml` files

### Phase 2: Reward Attribution Pipeline

#### 2.1 Judge Evaluator (`src/reward/judge_evaluator.py`)

Claude Opus-based behavioral quality scorer:
- Receives decision traces and behavioral rubric
- Scores each decision on taxonomy dimensions
- Returns score in [0, 1] with justification
- Runs periodically (not every episode — expensive)
- Calibrates against AAT's `BehavioralScorer` heuristic scores

#### 2.2 Composite Reward Calculator (`src/reward/composite_reward.py`)

Combines all reward signals:
- Wraps AAT's `RewardCalculator` for outcome signals
- Wraps AAT's `BehavioralScorer` for fast behavioral signals
- Adds judge evaluator scores (when available)
- Applies efficiency penalty and phase bonus
- Implements weight schedule per curriculum stage

#### 2.3 Cross-Validation Monitor (`src/reward/calibration.py`)

Detects reward function drift:
- Compares judge scores with outcome scores per batch
- Flags disagreements (high behavioral + bad outcome, or vice versa)
- Tracks reward distribution stability across training
- Alerts when distributions shift beyond thresholds

### Phase 3: Training Loop

#### 3.1 Trajectory Buffer (`src/training/trajectory_buffer.py`)

Collects and stores episode trajectories:
- Per-decision rewards from composite reward calculator
- Episode metadata (type, stage, difficulty, scenario config)
- Supports batched sampling for PPO updates
- Implements GAE (Generalized Advantage Estimation)

#### 3.2 PPO Trainer (`src/training/ppo_trainer.py`)

LoRA/QLoRA adapter training:
- PPO with clipped surrogate objective
- LoRA rank 16-64 on behavioral layers only (preserve base coding capability)
- Per-stage adapter checkpoints
- Learning rate schedule: 1e-5 to 5e-5 (conservative)
- Batch size: 16-32 episodes

#### 3.3 Curriculum Manager (`src/training/curriculum.py`)

Manages stage progression:
- Uses `ScenarioCatalog` to generate episode configs per stage
- Tracks graduation criteria per stage (see `docs/TRAINING_PIPELINE.md`)
- Advances to next stage when criteria met
- Saves LoRA adapters at stage completion
- Supports ablation (train subset of stages)

### Phase 4: Evaluation Harness

#### 4.1 Solo Evaluation Environment (`src/eval/solo_env.py`)

Standalone evaluation (no AAT dependency):
- 5-phase protocol: elicitation, research, planning, execution, verification
- Human proxy: scripted responses or fixed model
- Task bank: 50-100 tasks spanning ambiguity, domain, codebase, complexity

#### 4.2 Evaluation Metrics (`src/eval/metrics.py`)

Primary metrics: elicitation quality, sufficiency calibration, research effectiveness, decomposition accuracy, self-correction rate, adaptation quality, intent match.

Transfer score: `solo_metric / team_metric` per behavioral pattern. Target > 0.8.

#### 4.3 Evaluation Runner (`src/eval/runner.py`)

Evaluation schedule:
- Every 100 episodes: quick eval (10 tasks, primary metrics)
- Every 500 episodes: full eval (50 tasks, all metrics)
- Stage completion: comprehensive eval (100 tasks, transfer scores)
- Post-training: full benchmark, ablation studies, regression checks

### Phase 5: Integration and Pipeline

#### 5.1 Training Orchestrator (`src/orchestrator.py`)

End-to-end training pipeline:
- Initializes AAT env, reward pipeline, trainer, evaluator
- Runs curriculum from Stage 1 through Stage 4
- Handles checkpointing and fault recovery via `CheckpointManager`
- Parallelizes episodes across GPU instances
- Logs metrics (Prometheus, W&B, or similar)

#### 5.2 CLI Interface (`src/cli.py`)

```bash
# Full training pipeline
python -m src.cli train \
    --base-model /path/to/base-model \
    --aat-path /path/to/agile-agent-team \
    --output /path/to/output \
    --stages 1,2,3,4 \
    --judge-model claude-opus-4-6

# Resume training from checkpoint
python -m src.cli train \
    --resume /path/to/output/checkpoint-latest

# Evaluate a trained model
python -m src.cli evaluate \
    --model /path/to/trained-model \
    --adapter /path/to/lora-v4 \
    --tasks /path/to/task-bank \
    --output /path/to/eval-output

# Run a single episode for debugging
python -m src.cli episode \
    --type elicitation \
    --stage 1 \
    --difficulty medium \
    --model /path/to/candidate \
    --verbose
```

## Project Structure

```
dojo/
├── CLAUDE.md                           # This file
├── README.md                           # Project overview
├── LICENSE                             # MIT
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Tool config (black, ruff, mypy)
├── docs/
│   ├── ARCHITECTURE.md                 # System architecture
│   ├── DESIGN_RATIONALE.md             # Key decisions and reasoning
│   ├── TRAINING_PIPELINE.md            # End-to-end training flow
│   ├── REWARD_FUNCTION.md              # Reward attribution design
│   ├── EVALUATION_HARNESS.md           # Solo-deployment evaluation
│   └── INTEGRATION_WITH_AAT.md         # AAT RL API integration
├── specs/
│   ├── BEHAVIORAL_TAXONOMY.md          # 30 behavioral pattern specs
│   ├── TRAINING_EPISODES.md            # 13 episode types across 4 stages
│   ├── REWARD_SIGNALS.md               # Per-behavior reward signals
│   └── TRANSFER_ANALYSIS.md            # Team → solo transfer analysis
└── src/
    ├── __init__.py
    ├── cli.py                          # CLI entry point
    ├── orchestrator.py                 # Training orchestrator
    ├── env/
    │   ├── __init__.py
    │   └── aat_env.py                  # AATEnv(gym.Env) wrapper
    ├── runtime/
    │   ├── __init__.py
    │   └── candidate_runtime.py        # Training candidate AAT runtime
    ├── data/
    │   ├── __init__.py
    │   ├── backlog_generator.py        # Synthetic backlog generation
    │   └── task_bank/                  # Evaluation task bank (YAML)
    ├── reward/
    │   ├── __init__.py
    │   ├── judge_evaluator.py          # Claude Opus behavioral judge
    │   ├── composite_reward.py         # Combined reward calculator
    │   └── calibration.py              # Cross-validation monitor
    ├── training/
    │   ├── __init__.py
    │   ├── trajectory_buffer.py        # Trajectory collection + GAE
    │   ├── ppo_trainer.py              # PPO with LoRA/QLoRA
    │   └── curriculum.py               # Stage progression manager
    └── eval/
        ├── __init__.py
        ├── solo_env.py                 # Solo evaluation environment
        ├── metrics.py                  # Transfer scores and metrics
        └── runner.py                   # Evaluation scheduler
```

## AAT Integration Reference

### AAT's `src.rl` Exports (28 symbols)

| Symbol | Purpose | Dojo Component That Uses It |
|---|---|---|
| `EpisodeRunner` | Runs complete episodes | `AATEnv` |
| `EpisodeResult` | Episode outcome data | `AATEnv`, `composite_reward` |
| `PhaseRunner` | Runs individual phases | `AATEnv` (phase-level mode) |
| `PhaseResult` | Phase outcome data | `AATEnv` |
| `ScenarioCatalog` | Generates episode configs | `curriculum` |
| `ScenarioConfig` | Single episode config | `AATEnv` |
| `EPISODE_TYPES` | Supported episode types | `curriculum` |
| `ObservationExtractor` | Gym-compatible observations | `AATEnv` |
| `Observation` | Structured observation | `AATEnv` |
| `AgentObservation` | Per-agent observation | `AATEnv` |
| `RewardCalculator` | Outcome-based reward | `composite_reward` |
| `RewardSignal` | Individual reward signal | `composite_reward` |
| `RewardWeights` | Weight configuration | `composite_reward` |
| `BehavioralScorer` | Heuristic behavioral scoring | `composite_reward` |
| `BehavioralCode` | Single code definition | `judge_evaluator` |
| `BEHAVIORAL_CODES` | All 30 codes | `judge_evaluator`, `metrics` |
| `ActionExecutor` | Dispatches RL actions | `AATEnv` |
| `InjectDisturbance` | Chaos injection action | `AATEnv` (Stage 2+) |
| `SwapAgentRole` | Role swap action | `AATEnv` |
| `ModifyBacklog` | Backlog modification | `AATEnv` |
| `ModifyTeamComposition` | Attrition/backfill | `AATEnv` (Stage 4) |
| `AdjustSprintParams` | Sprint parameter tuning | `AATEnv` |
| `ACTION_SPACE_SPEC` | Gym space metadata | `AATEnv` |
| `CheckpointManager` | State serialization | `orchestrator` |
| `Checkpoint` | State snapshot | `orchestrator` |
| `ExperimentConfigBuilder` | Programmatic config | `AATEnv`, `curriculum` |
| `ExperimentConfig` | Validated config | `AATEnv` |
| `register_runtime` | Custom runtime registration | `candidate_runtime` |

### AAT CLI (for debugging/fallback)

```bash
# Run a sprint in mock mode (no LLM)
MOCK_LLM=true python -m src.orchestrator.main \
    --sprints 1 --output /tmp/test --db-url mock://

# Continue an experiment
python -m src.orchestrator.main --continue 2 --output /tmp/test --db-url mock://
```

## Compute Estimates

| Resource | Estimate | Notes |
|---|---|---|
| Training GPU-hours | ~200 | Depends on model size and parallelism |
| Judge evaluations | $200-500 | Claude Opus API costs |
| Evaluation harness | ~20 GPU-hours | Periodic full evaluations |
| Total episodes | ~7600 | Across all 4 stages |
| Per-episode time | 30-75s | Phase-level episodes |
| Per-stage wall-clock | ~12.5h | With 4x parallelism |

## Code Conventions

- Python 3.11+, async-first with asyncio
- Type hints everywhere (mypy strict)
- Black formatting, Ruff linting
- Pre-commit hooks enforced
- Tests: pytest + pytest-asyncio
- No emojis in code or docs
- Docstrings on public API only (not internal helpers)
- Keep it simple: no premature abstractions

## Testing Strategy

- Unit tests for each component (reward calculations, curriculum logic, backlog generation)
- Integration tests using AAT's mock mode (`MOCK_LLM=true`) for full episode runs
- Evaluation tests verifying metric computation against known-good traces
- No tests for the judge evaluator itself (LLM output is non-deterministic)

## Key Design Decisions

1. **Behavior over skills**: Train meta-cognitive patterns into weights; keep language/tool knowledge in RAG
2. **RL over distillation**: Let the model discover strategies through trial, not imitate a larger model
3. **Team training, solo deployment**: Multi-agent gym for richer signal; evaluate always in solo context
4. **Phase-level episodes**: 10-100x cheaper than full sprints; enable focused behavioral training
5. **Library integration**: Import AAT's `src.rl` directly instead of subprocess invocation
6. **Two-level behavioral scoring**: Fast heuristic (every episode) + nuanced judge (periodic calibration)

See `docs/DESIGN_RATIONALE.md` for full reasoning.

## Critical Files

| File | Importance |
|---|---|
| `specs/BEHAVIORAL_TAXONOMY.md` | The curriculum and rubric — defines what "senior behavior" means |
| `specs/TRAINING_EPISODES.md` | Episode specifications — defines what the model experiences |
| `specs/REWARD_SIGNALS.md` | Per-behavior reward signals — defines how we measure |
| `docs/INTEGRATION_WITH_AAT.md` | AAT API reference — defines how we connect |
| `docs/REWARD_FUNCTION.md` | Composite reward design — the core training signal |
| `docs/TRAINING_PIPELINE.md` | Curriculum stages and graduation criteria |
| `docs/EVALUATION_HARNESS.md` | How we know if it worked |

## Open Questions

1. Can behavioral LoRA adapters transfer across base models?
2. Risk of "performative seniority" — model performing behaviors without substance
3. Sufficient episode diversity for behavioral convergence
4. Optimal calibration frequency for judge evaluator vs. heuristic scorer
5. Whether Stage 3 (borrowing/orientation) is disproportionately valuable for solo deployment
