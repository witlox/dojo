# System Architecture

## Overview

The Senior Meta-Cognitive Model Trainer is a pipeline that uses a multi-agent software development simulation as a reinforcement learning environment to train behavioral patterns into a local LLM.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Training Orchestrator                         │
│  - Manages curriculum stages (1→4)                                   │
│  - Selects episode types and difficulty                               │
│  - Coordinates between components                                    │
│  - Tracks training progress and convergence                          │
└───────────┬──────────────────┬───────────────────┬───────────────────┘
            │                  │                   │
            ▼                  ▼                   ▼
┌───────────────────┐ ┌────────────────────┐ ┌─────────────────────────┐
│  Simulation Env   │ │ Reward Attribution │ │    Training Loop        │
│  (agile-agent-    │ │    Pipeline        │ │                         │
│   team wrapper)   │ │                    │ │  - Trajectory buffer    │
│                   │ │ - Decision tracer  │ │  - Advantage estimation │
│  - Episode API    │ │ - Outcome mapper   │ │  - Policy updates (PPO) │
│  - Model injection│ │ - Judge evaluator  │ │  - Checkpoint mgmt      │
│  - Artifact       │ │ - Composite reward │ │  - Behavioral adapter   │
│    extraction     │ │   calculation      │ │    (LoRA/QLoRA)         │
│                   │ │                    │ │                         │
└───────────────────┘ └────────────────────┘ └─────────────────────────┘
            │                  │                   │
            ▼                  ▼                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Evaluation Harness                             │
│  - Solo deployment simulation (model + human proxy)                  │
│  - Behavioral rubric scoring                                         │
│  - Transfer measurement (team training → solo performance)           │
│  - Regression testing across curriculum stages                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Component 1: Simulation Environment Wrapper

### Purpose

Wraps the agile-agent-team system to expose it as a `gym.Env` with episode-level and phase-level granularity. Uses AAT's public `src.rl` API directly.

### Interface

```python
import gymnasium as gym
from src.rl import (
    EpisodeRunner, EpisodeResult, ScenarioCatalog, ScenarioConfig,
    ObservationExtractor, Observation, ActionExecutor, ACTION_SPACE_SPEC,
    RewardCalculator, RewardWeights, BehavioralScorer,
    CheckpointManager, ExperimentConfigBuilder, PhaseRunner,
)

class AATEnv(gym.Env):
    """Gym wrapper around agile-agent-team's RL API."""

    def __init__(self, stage: int, episode_type: str, difficulty: str = "medium"):
        self.config = ExperimentConfigBuilder() \
            .with_stage(stage) \
            .with_episode_type(episode_type) \
            .build()
        self.runner = EpisodeRunner(self.config)
        self.phase_runner = PhaseRunner(self.config)
        self.catalog = ScenarioCatalog()
        self.obs_extractor = ObservationExtractor()
        self.reward_calc = RewardCalculator(RewardWeights.for_stage(stage))
        self.scorer = BehavioralScorer()
        self.action_exec = ActionExecutor()
        self.checkpoint_mgr = CheckpointManager()

        # Gym spaces from AAT's ACTION_SPACE_SPEC
        self.action_space = self._build_action_space(ACTION_SPACE_SPEC)
        self.observation_space = self._build_obs_space()

    def reset(self, seed=None, options=None):
        scenario = self.catalog.generate(self.episode_type, self.difficulty)
        self._current_scenario = scenario
        observation = self.obs_extractor.extract(self.runner)
        return observation, {"scenario": scenario}

    def step(self, action):
        result = self.action_exec.execute(action)
        observation = self.obs_extractor.extract(self.runner)
        reward = self.reward_calc.compute(result)
        score, behaviors = self.scorer.score(result.decision_traces)
        info = {
            "behavioral_score": score,
            "behaviors_detected": behaviors,
            "decision_traces": result.decision_traces,
            "phase_results": result.phase_results,
        }
        return observation, reward, result.terminated, result.truncated, info

    async def run_episode(self, episode_type: str, difficulty: str) -> EpisodeResult:
        """Run a complete episode using AAT's EpisodeRunner."""
        return await self.runner.run_episode(episode_type=episode_type, difficulty=difficulty)

    async def run_scenario(self, scenario: ScenarioConfig) -> EpisodeResult:
        """Run a pre-generated scenario."""
        return await self.runner.run_scenario(scenario)
```

### Model Injection

The wrapper injects the training candidate model into specific agent slots using AAT's `ExperimentConfigBuilder` and `register_runtime`:

```python
from src.rl import ExperimentConfigBuilder, register_runtime

# Register custom runtime for the training candidate
register_runtime("training_candidate", TrainingCandidateRuntime)

config = ExperimentConfigBuilder() \
    .with_runtime("training_candidate", model_path="/path/to/checkpoint-N") \
    .with_agent_override("ahmed_senior_dev_lead", runtime="training_candidate") \
    .build()
```

For behavioral training, the candidate model is placed in senior agent slots. The remaining agents run on fixed models (Claude Sonnet or vLLM) to provide a consistent environment.

### Episode Types

| Episode Type | Duration | Ceremony Phase | Primary Behaviors |
|---|---|---|---|
| `elicitation` | Short (~2 min) | Story refinement | Question quality, sufficiency detection |
| `decomposition` | Short (~3 min) | Technical planning | Task breakdown, dependency identification |
| `risk_assessment` | Short (~2 min) | Planning + checkpoint | Risk identification, mitigation planning |
| `implementation` | Medium (~10 min) | Pairing session | Self-monitoring, checkpoint decisions |
| `adaptation` | Medium (~5 min) | Disturbance response | Triage, replanning, communication |
| `orientation` | Medium (~5 min) | Borrowing arrival | Rapid context gathering, convention detection |
| `full_sprint` | Long (~20 min) | Full sprint lifecycle | Composed behaviors (evaluation only) |
| `multi_sprint` | Long (~60 min) | Multiple sprints | Learning curves, meta-learning (evaluation only) |

## Component 2: Reward Attribution Pipeline

### Purpose

Transforms raw sprint artifacts into per-decision reward vectors that connect reasoning choices to outcomes.

### Architecture

```
Sprint Artifacts                    Decision-Outcome Pairs
┌─────────────────┐                ┌──────────────────────────┐
│ pairing_log.json│──┐             │ decision_id: "q-003"     │
│ kanban.json     │  │  ┌───────┐  │ action: ask_question     │
│ retro.md        │──┼──│Tracer │──│ content: "What about..." │
│ test_results    │  │  └───┬───┘  │ outcome_reward: 0.73     │
│ meta_learnings  │──┘      │      │ behavioral_reward: 0.81  │
│ final_report    │         │      │ composite_reward: 0.78   │
└─────────────────┘         │      └──────────────────────────┘
                            │
                    ┌───────▼──────┐
                    │   Outcome    │
                    │   Mapper     │
                    │              │
                    │ Maps decision│
                    │ IDs to sprint│
                    │ outcomes     │
                    └───────┬──────┘
                            │
                    ┌───────▼──────┐
                    │    Judge     │
                    │  Evaluator   │
                    │              │
                    │ Large model  │
                    │ scores       │
                    │ behavioral   │
                    │ quality      │
                    └───────┬──────┘
                            │
                    ┌───────▼──────┐
                    │  Composite   │
                    │   Reward     │
                    │              │
                    │ Combines     │
                    │ outcome +    │
                    │ behavioral   │
                    │ signals      │
                    └──────────────┘
```

### Decision Tracer

AAT's decision tracing (Phase A of the RL API) logs every action the training model takes during an episode with a unique `decision_id`. The tracer links these forward to outcomes:

- Question asked during refinement → Did the answer change the implementation?
- Research query executed → Was the retrieved information used in the code?
- Decomposition choice → Did tasks parallelize effectively? Were dependencies real?
- Checkpoint decision (continue/pivot/escalate) → Did it lead to completion or rework?
- Risk identified → Did the predicted risk materialize?

### Outcome Mapper

Uses agile-agent-team artifacts to compute outcome scores:

| Artifact | Outcome Signal |
|---|---|
| `final_report.json` | Velocity, features completed |
| `kanban.json` | Stories accepted vs. rejected, cycle time |
| Test results | Pass rate, iteration count |
| `pairing_log.json` | Rework count, checkpoint outcomes |
| `retro.md` | Learnings captured (meta-learning quality) |
| Disturbance recovery | Recovery time, blast radius |

### Judge Evaluator

Two levels of behavioral evaluation are used:

1. **AAT's `BehavioralScorer`**: Fast heuristic detection using keyword matching and action ordering checks. Runs during training for every episode. Detects all 30 behavioral codes without LLM calls.
2. **Dojo's Judge Evaluator (Claude Opus)**: Nuanced behavioral quality scoring against the full taxonomy rubric. Runs periodically (every N episodes) for calibration and cross-validation.

The judge receives the full trace and scores each decision on dimensions defined in the taxonomy. The judge evaluation and outcome evaluation serve as cross-validation:
- High behavioral score + good outcome → Reinforce
- High behavioral score + bad outcome → Rubric needs updating
- Low behavioral score + good outcome → Lucky; don't reinforce strongly
- Low behavioral score + bad outcome → Clear negative signal

### Composite Reward

```python
composite_reward = (
    outcome_weight * outcome_reward +
    behavioral_weight * behavioral_reward +
    efficiency_penalty * unnecessary_actions +
    phase_completion_bonus
)
```

Weights shift across curriculum stages:
- Stage 1: Heavier behavioral weight (learning patterns)
- Stage 4: Heavier outcome weight (patterns must produce results)

## Component 3: Training Loop

### Purpose

Standard RL training infrastructure adapted for behavioral fine-tuning of language models.

### Approach

- **Algorithm**: PPO (Proximal Policy Optimization) or similar on-policy method
- **Model adaptation**: LoRA/QLoRA adapters on the base model — only behavioral layers are trained, preserving base coding capability
- **Trajectory buffer**: Collects episode trajectories with per-decision rewards
- **Advantage estimation**: GAE (Generalized Advantage Estimation) over phase-level episodes

### Training Flow

```
For each curriculum stage:
    For each epoch:
        1. Sample batch of episode configs via ScenarioCatalog
        2. Run episodes through AATEnv (wrapping EpisodeRunner/PhaseRunner)
        3. Extract trajectories from EpisodeResult.decision_traces
        4. Compute rewards via RewardCalculator + BehavioralScorer + Judge Evaluator
        5. Estimate advantages (GAE)
        6. Update LoRA adapters via PPO
        7. Evaluate on held-out solo scenarios (Evaluation Harness)
        8. If solo performance converges, advance to next stage
```

### Checkpoint Management

- Save LoRA adapters at each stage completion
- Maintain separate adapters per stage for ablation studies
- Final model merges all stage adapters

## Component 4: Evaluation Harness

### Purpose

Tests whether behavioral patterns learned in the team environment transfer to solo deployment — the actual use case.

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Task Bank  │────▶│ Trained Model│────▶│  Outcome        │
│             │     │              │     │  Evaluator       │
│ Vague tasks │◀────│ (asks Qs,    │     │                 │
│ Codebases   │     │  researches, │     │ - Code quality  │
│ Constraints │     │  plans,      │     │ - Intent match  │
│             │     │  executes)   │     │ - Efficiency    │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  Behavioral  │
                    │  Scorer      │
                    │              │
                    │ Rubric from  │
                    │ taxonomy     │
                    └──────────────┘
```

### Solo Evaluation Protocol

1. Present model with a vague task description and optional codebase
2. Model enters elicitation phase — measure question quality and sufficiency detection
3. Provide answers; model enters research phase — measure search strategy and source evaluation
4. Model enters planning phase — measure decomposition quality and risk identification
5. Model enters execution phase — measure self-correction and checkpoint behavior
6. Evaluate final output against intent and quality criteria

### Metrics

| Metric | What It Measures | Source |
|---|---|---|
| Elicitation quality | Questions specificity, coverage, prioritization | Judge evaluator |
| Sufficiency detection | Did model stop asking at the right time? | Judge + outcome |
| Decomposition quality | Task breakdown tractability, dependency accuracy | Outcome |
| Risk identification | Predicted vs. actual problems | Outcome |
| Self-correction rate | How often model caught and fixed own mistakes | Execution trace |
| Adaptation quality | Response to unexpected failures | Execution trace |
| Intent match | Did the output match what was asked? | Outcome evaluator |
| Efficiency | Actions taken vs. minimum necessary | Trace analysis |

### Transfer Score

The key metric is the *transfer score* — the ratio of solo evaluation performance to team training performance. If borrowing training improves solo codebase orientation scores, the transfer is positive and the training is valuable. If it doesn't, it's wasted compute.

## Data Flow

### Training Episode

```
1. Curriculum Manager selects episode config via ScenarioCatalog
   (type: elicitation, stage: 2, difficulty: medium)
        │
2. AATEnv.reset(config)
   - Configures agile-agent-team via ExperimentConfigBuilder
   - Injects training model via register_runtime()
   - Initializes sprint context
        │
3. Episode loop:
   - AATEnv provides observation via ObservationExtractor
   - Training model produces action
   - AATEnv.step(action) → next observation + done flag
   - AAT decision tracer logs (decision_id, action, context)
        │
4. Episode completes → EpisodeResult
   - BehavioralScorer scores decision traces (fast heuristic)
   - RewardCalculator computes outcome reward
   - Judge Evaluator scores behavioral quality (periodic)
        │
5. Trajectory + rewards → Training Loop buffer
        │
6. After N episodes: PPO update on LoRA adapters
        │
7. Periodic: Evaluation Harness measures solo performance
```

## Infrastructure Requirements

### Training Phase

- GPU for model inference during episodes (candidate model + environment agents)
- GPU for LoRA training updates
- Storage for trajectories and checkpoints
- Large model API access for judge evaluator (Claude Opus)

### Deployment Phase (Product)

- Local inference hardware for the fine-tuned model
- RAG pipeline for external knowledge retrieval
- No cloud dependency (fully local operation possible)

## Scalability

### Episode Parallelism

Multiple episodes can run concurrently using the agile-agent-team's async architecture:
- Different episode configs on different GPU instances via `EpisodeRunner`
- Trajectory collection is embarrassingly parallel
- Only the PPO update step requires synchronization
- `CheckpointManager` enables state save/restore for fault tolerance

### Curriculum Efficiency

Phase-level episodes via AAT's `PhaseRunner` are 10-100x cheaper than full sprints. The bulk of training uses short episodes (2-10 minutes) targeting individual ceremonies. Full sprint evaluations via `EpisodeRunner` run periodically (every N training steps) to measure composition of learned behaviors.
