# Integration with agile-agent-team

## Relationship

This project uses [agile-agent-team](https://github.com/witlox/agile-agent-team) (AAT) as a simulation environment. AAT provides a public `src.rl` package with a complete API surface designed specifically for this integration. Dojo wraps AAT's RL API as a `gym.Env`.

## AAT RL API Surface

AAT exports 28 symbols from `src.rl`:

```python
from src.rl import (
    # Episode harness
    EpisodeRunner,        # Runs complete episodes by type and difficulty
    EpisodeResult,        # Episode outcome: rewards, traces, behavioral scores

    # Scenario catalog
    ScenarioCatalog,      # Generates scenario configs for each episode type
    ScenarioConfig,       # Configuration for a single episode
    EPISODE_TYPES,        # Registry of supported episode types

    # Observation
    ObservationExtractor, # Extracts gym-compatible observations from sprint state
    Observation,          # Structured observation (team state, kanban, context)
    AgentObservation,     # Per-agent observation slice

    # Reward
    RewardCalculator,     # Computes composite reward from artifacts
    RewardSignal,         # Individual reward signal (source, value, weight)
    RewardWeights,        # Weight configuration per curriculum stage

    # Behavioral taxonomy
    BehavioralScorer,     # Scores decision traces against 30 behavioral codes
    BehavioralCode,       # Single behavioral code definition
    BEHAVIORAL_CODES,     # Registry of all 30 codes (B-01 through B-30)

    # Action space
    ActionExecutor,       # Dispatches RL actions to SprintManager APIs
    InjectDisturbance,    # Action: inject a disturbance mid-sprint
    SwapAgentRole,        # Action: swap an agent's role
    ModifyBacklog,        # Action: add/remove a story from backlog
    ModifyTeamComposition,# Action: agent departure or backfill
    AdjustSprintParams,   # Action: modify sprint duration or WIP limits
    ACTION_SPACE_SPEC,    # Metadata dict for gym.Space construction

    # Checkpointing
    CheckpointManager,    # Saves/restores mid-episode state
    Checkpoint,           # Serialized state snapshot

    # Config
    ExperimentConfigBuilder, # Programmatic experiment configuration
    ExperimentConfig,        # Validated experiment config

    # Phase runner
    PhaseRunner,          # Runs individual sprint phases (ceremonies)
    PhaseResult,          # Result of a single phase

    # Runtime
    register_runtime,     # Register custom model runtimes
)
```

## Integration Architecture

### Dojo's `AATEnv(gym.Env)` Wrapper

Dojo wraps AAT's RL API as a standard gym environment:

```python
import gymnasium as gym
from src.rl import (
    EpisodeRunner, ScenarioCatalog, ObservationExtractor,
    RewardCalculator, BehavioralScorer, ActionExecutor,
    CheckpointManager, ExperimentConfigBuilder,
)

class AATEnv(gym.Env):
    """Gym wrapper around agile-agent-team's RL API."""

    def __init__(self, stage: int, episode_type: str, difficulty: str):
        self.config = ExperimentConfigBuilder() \
            .with_stage(stage) \
            .with_episode_type(episode_type) \
            .build()
        self.runner = EpisodeRunner(self.config)
        self.catalog = ScenarioCatalog()
        self.obs_extractor = ObservationExtractor()
        self.reward_calc = RewardCalculator()
        self.scorer = BehavioralScorer()
        self.action_exec = ActionExecutor()
        self.checkpoint_mgr = CheckpointManager()

    def reset(self, seed=None):
        scenario = self.catalog.generate(self.episode_type, self.difficulty)
        # ... initialize episode
        return observation, info

    def step(self, action):
        result = self.action_exec.execute(action)
        observation = self.obs_extractor.extract()
        reward = self.reward_calc.compute(result)
        score, behaviors = self.scorer.score(result.decision_traces)
        return observation, reward, terminated, truncated, info
```

### 1. Model Injection via ExperimentConfigBuilder

AAT's `ExperimentConfigBuilder` provides programmatic configuration. The training pipeline injects the candidate model into specific agent slots:

```python
from src.rl import ExperimentConfigBuilder, register_runtime

# Register the training candidate as a custom runtime
register_runtime("training_candidate", TrainingCandidateRuntime)

config = ExperimentConfigBuilder() \
    .with_runtime("training_candidate", model_path="/path/to/checkpoint-N") \
    .with_agent_override("ahmed_senior_dev_lead", runtime="training_candidate") \
    .with_disturbances(enabled=False) \
    .with_teams(count=1) \
    .build()
```

The candidate model can be placed in different agent slots to train different behavioral contexts:
- **Senior dev slot**: For elicitation, decomposition, implementation behaviors
- **Dev lead slot**: For triage, escalation, facilitation behaviors
- **Borrowed agent slot**: For orientation and cross-team behaviors

All other agents run on fixed models (Claude Sonnet or vLLM) to provide a consistent environment.

### 2. Episode Control via EpisodeRunner

AAT's `EpisodeRunner` provides both full-episode and scenario-based execution:

```python
from src.rl import EpisodeRunner, ScenarioCatalog, ScenarioConfig

runner = EpisodeRunner(config)

# Run by episode type and difficulty
result = await runner.run_episode(episode_type="elicitation", difficulty="medium")

# Or run from a pre-generated scenario
catalog = ScenarioCatalog()
scenario = catalog.generate(episode_type="triage", difficulty="hard")
result = await runner.run_scenario(scenario)

# Result contains everything needed for training
print(result.reward)              # Composite reward signal
print(result.behavioral_score)    # BehavioralScorer output
print(result.behaviors_detected)  # List of detected behavioral codes
print(result.decision_traces)     # Full decision trace for attribution
print(result.phase_results)       # Per-phase breakdown
```

### 3. Phase-Level Control via PhaseRunner

For phase-level training episodes (not full sprints), AAT's `PhaseRunner` provides direct control over individual sprint ceremonies:

```python
from src.rl import PhaseRunner, PhaseResult

phase_runner = PhaseRunner(config)

# Run individual phases
refinement_result = await phase_runner.run_phase("story_refinement")
planning_result = await phase_runner.run_phase("technical_planning")
pairing_result = await phase_runner.run_phase("pairing_session")

# Extract phase-level observations and rewards between phases
```

This enables the phase-level training episodes described in `specs/TRAINING_EPISODES.md` — short (2-10 minute) episodes targeting specific behavioral patterns rather than expensive full-sprint episodes.

### 4. Action Space

AAT defines 5 RL action types that Dojo uses between sprint phases to control the training environment:

| Action | Parameters | Use Case |
|---|---|---|
| `InjectDisturbance` | disturbance_type, severity | Stage 2+ chaos injection |
| `SwapAgentRole` | agent_id, target_role, proficiency | Profile swapping scenarios |
| `ModifyBacklog` | add/remove story | Scope change episodes |
| `ModifyTeamComposition` | depart/backfill agent | Stage 4 attrition/onboarding |
| `AdjustSprintParams` | duration, wip_limits | Pressure variation |

`ACTION_SPACE_SPEC` provides metadata for constructing `gym.Space` objects (categorical, continuous, agent_ref, role_ref, dict types).

### 5. Observation Extraction

AAT's `ObservationExtractor` produces structured observations compatible with gym:

```python
from src.rl import ObservationExtractor, Observation, AgentObservation

extractor = ObservationExtractor()
obs: Observation = extractor.extract(sprint_manager)

# Observation includes:
# - Team state (agents, roles, current assignments)
# - Kanban state (cards, transitions, WIP)
# - Sprint context (day, phase, velocity history)
# - Per-agent observations (AgentObservation)
```

### 6. Reward Computation

AAT provides both outcome-based and behavioral reward computation:

```python
from src.rl import RewardCalculator, RewardWeights, BehavioralScorer

# Configure reward weights per curriculum stage
weights = RewardWeights(
    outcome_weight=0.3,    # Stage 1: behavioral-heavy
    behavioral_weight=0.7,
)

reward_calc = RewardCalculator(weights)
reward = reward_calc.compute(episode_result)

# BehavioralScorer uses heuristic detection (no LLM calls)
scorer = BehavioralScorer()
score, detected_codes = scorer.score(decision_traces)
# Returns (float, List[str]) — overall score and list of detected codes
```

The `BehavioralScorer` provides fast heuristic scoring during training. Dojo supplements this with the large-model judge evaluator (Claude Opus) for nuanced behavioral quality assessment — see `docs/REWARD_FUNCTION.md`.

### 7. Behavioral Taxonomy Integration

AAT implements all 30 behavioral codes from `specs/BEHAVIORAL_TAXONOMY.md`:

```python
from src.rl import BEHAVIORAL_CODES, BehavioralCode

# Access any behavioral code definition
b01 = BEHAVIORAL_CODES["B-01"]
print(b01.name)              # "Ambiguity Detection"
print(b01.stage)             # 1
print(b01.category)          # "elicitation"
print(b01.detection_heuristic)  # Function for keyword-based detection
```

The `BehavioralScorer` implements detection heuristics for all 30 codes using keyword matching against action content and context fields, plus action ordering checks (e.g., test-before-commit patterns).

### 8. Checkpointing for Curriculum Replay

AAT's `CheckpointManager` serializes mid-episode state for curriculum replay and debugging:

```python
from src.rl import CheckpointManager, Checkpoint

mgr = CheckpointManager(checkpoint_dir="/tmp/checkpoints")

# Save mid-episode state
checkpoint = mgr.save(sprint_manager, episode_id="ep-001", sprint_num=1, phase="pairing")

# Restore from checkpoint
mgr.restore(sprint_manager, checkpoint_path)

# List all checkpoints for an episode
checkpoints = mgr.list_checkpoints(episode_id="ep-001")
# Storage: {checkpoint_dir}/{episode_id}/s{sprint:02d}-{phase}.json
```

### 9. Artifact Extraction

AAT produces structured artifacts that the reward pipeline consumes:

| Artifact | Path | Contains |
|---|---|---|
| Sprint results | `<output>/sprint-NN/kanban.json` | Card states, transitions, metadata |
| Pairing logs | `<output>/sprint-NN/pairing_log.json` | Session details, dialogues, outcomes |
| Retrospective | `<output>/sprint-NN/retro.md` | Keep/Drop/Puzzle learnings |
| Final report | `<output>/final_report.json` | Velocity, coverage, features |
| Generated code | `/tmp/agent-workspace/sprint-NN/*/` | Source, tests, git history |
| Meta-learnings | `team_config/07_meta/meta_learnings.jsonl` | Agent learnings across sprints |
| Decision traces | `<output>/sprint-NN/decision_traces.json` | Per-decision attribution data |

### 10. Multi-Team and Borrowing Episodes

AAT's multi-team mode is configured via `ExperimentConfigBuilder`:

```python
config = ExperimentConfigBuilder() \
    .with_teams(count=2) \
    .with_coordination(enabled=True, max_borrows=2) \
    .with_agent_override("training_candidate", team="team-beta") \
    .build()
```

For borrowing episodes, the training candidate starts in one team and gets borrowed to another. The `EpisodeRunner` handles the borrowing lifecycle, and the reward pipeline measures orientation speed and contribution quality in the receiving team.

### 11. Disturbance Configuration

AAT's disturbance engine is configured per curriculum stage:

```python
# Stage 1: No disturbances
config = ExperimentConfigBuilder() \
    .with_disturbances(enabled=False) \
    .build()

# Stage 2: Full disturbances
config = ExperimentConfigBuilder() \
    .with_disturbances(
        enabled=True,
        frequencies={
            "dependency_breaks": 0.166,
            "production_incident": 0.125,
            "flaky_test": 0.25,
            "scope_creep": 0.20,
            "junior_misunderstanding": 0.33,
            "architectural_debt_surfaces": 0.166,
            "merge_conflict": 0.30,
        }
    ).build()
```

The `InjectDisturbance` action also allows Dojo to inject disturbances dynamically during episodes.

### 12. Backlog Generation

Each episode needs a backlog. The training pipeline generates backlogs and passes them via config:

```python
config = ExperimentConfigBuilder() \
    .with_backlog(backlog_path="/tmp/episode-backlog.yaml") \
    .build()
```

Backlogs are generated by sampling from a curated task bank (see `specs/TRAINING_EPISODES.md`), varying ambiguity levels, domains, languages, and codebase scale.

## AAT Features Used Per Training Stage

| AAT Feature | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| EpisodeRunner | ✅ Primary | ✅ Primary | ✅ Primary | ✅ Primary |
| PhaseRunner | ✅ Phase episodes | ✅ Phase episodes | ✅ Phase episodes | ✅ Phase episodes |
| BehavioralScorer | ✅ All 30 codes | ✅ All 30 codes | ✅ All 30 codes | ✅ All 30 codes |
| RewardCalculator | ✅ Outcome signals | ✅ Outcome signals | ✅ Outcome signals | ✅ Outcome signals |
| CheckpointManager | ✅ State save/restore | ✅ State save/restore | ✅ State save/restore | ✅ State save/restore |
| ActionExecutor | ❌ | ✅ Disturbances | ✅ Composition | ✅ Attrition |
| ScenarioCatalog | ✅ Generates configs | ✅ Generates configs | ✅ Generates configs | ✅ Generates configs |
| Story refinement | ✅ Elicitation eps | ✅ | ✅ | ✅ |
| Technical planning | ✅ Decomposition eps | ✅ | ✅ | ✅ |
| Pairing sessions | ✅ Implementation eps | ✅ | ✅ | ✅ |
| Disturbances | ❌ | ✅ Primary | ✅ | ✅ |
| Multi-team mode | ❌ | ❌ | ✅ Primary | ✅ |
| Agent borrowing | ❌ | ❌ | ✅ Primary | ✅ |
| Turnover/attrition | ❌ | ❌ | ❌ | ✅ Primary |
| Decision tracing | ✅ Attribution | ✅ Attribution | ✅ Attribution | ✅ Attribution |

## CLI Fallback

While library-level integration via `src.rl` is the primary approach, AAT can also be invoked via CLI for isolation or debugging:

```bash
# Single sprint via CLI
python -m src.orchestrator.main \
  --config /tmp/episode-config.yaml \
  --backlog /tmp/episode-backlog.yaml \
  --sprints 1 \
  --output /tmp/episode-output \
  --db-url mock://

# Continue from previous run
python -m src.orchestrator.main \
  --continue 2 \
  --output /tmp/episode-output \
  --db-url mock://
```

## What Dojo Builds on Top of AAT

AAT provides the simulation environment and low-level RL primitives. Dojo adds:

| Dojo Component | What It Does | AAT Components Used |
|---|---|---|
| `AATEnv(gym.Env)` | Gym-compatible wrapper | EpisodeRunner, ObservationExtractor, ActionExecutor |
| Curriculum Manager | Selects episode types, manages stage progression | ScenarioCatalog, ExperimentConfigBuilder |
| Judge Evaluator | Large-model behavioral quality scoring | Decision traces from EpisodeResult |
| Composite Reward | Combines outcome + behavioral + judge + efficiency | RewardCalculator, BehavioralScorer |
| Training Loop (PPO) | LoRA/QLoRA adapter updates | Trajectory data from episodes |
| Evaluation Harness | Solo deployment testing | Trained model (standalone, no AAT) |
| Backlog Generator | Synthetic diverse backlogs | Backlog format from AAT |
| Checkpoint Orchestrator | Cross-episode state management | CheckpointManager |
