# Integration with agile-agent-team

## Relationship

This project uses [agile-agent-team](https://github.com/witlox/agile-agent-team) (AAT) as a simulation environment. AAT is not modified — this project wraps around it through its existing configuration and runtime interfaces.

## Integration Points

### 1. Model Injection via Runtime Config

AAT already supports per-agent runtime and model assignment. The training pipeline injects the candidate model by generating a modified `config.yaml`:

```yaml
# Training candidate replaces a senior agent slot
models:
  agents:
    ahmed_senior_dev_lead:
      runtime: "local_vllm"
      model: "/path/to/training-candidate-checkpoint-N"
      tools: ["filesystem", "git", "bash"]
      temperature: 0.7
      max_tokens: 3072
    
    # All other agents remain fixed (consistent environment)
    alex_senior_networking:
      runtime: "anthropic"
      model: "claude-sonnet-4-5"
      # ...
```

The candidate model can be placed in different agent slots to train different behavioral contexts:
- **Senior dev slot**: For elicitation, decomposition, implementation behaviors
- **Dev lead slot**: For triage, escalation, facilitation behaviors
- **Borrowed agent slot**: For orientation and cross-team behaviors

### 2. Episode Control via CLI

AAT's CLI provides episode control:

```bash
# Single sprint episode
python -m src.orchestrator.main \
  --config /tmp/episode-config.yaml \
  --backlog /tmp/episode-backlog.yaml \
  --sprints 1 \
  --output /tmp/episode-output \
  --db-url mock://

# Multi-sprint episode (for evaluation)
python -m src.orchestrator.main \
  --sprints 5 \
  --output /tmp/eval-output \
  --db-url mock://
```

The training pipeline generates per-episode config and backlog files, invokes AAT, and collects outputs.

### 3. Artifact Extraction

AAT produces structured artifacts that the reward pipeline consumes:

| Artifact | Path | Contains |
|---|---|---|
| Sprint results | `<output>/sprint-NN/kanban.json` | Card states, transitions, metadata |
| Pairing logs | `<output>/sprint-NN/pairing_log.json` | Session details, dialogues, outcomes |
| Retrospective | `<output>/sprint-NN/retro.md` | Keep/Drop/Puzzle learnings |
| Final report | `<output>/final_report.json` | Velocity, coverage, features |
| Generated code | `/tmp/agent-workspace/sprint-NN/*/` | Source, tests, git history |
| Meta-learnings | `team_config/07_meta/meta_learnings.jsonl` | Agent learnings across sprints |

### 4. Phase-Level Episode Extraction

For phase-level training (not full sprints), the pipeline needs to intercept AAT at ceremony boundaries. Two approaches:

**Approach A: Subprocess with artifact parsing (simpler, recommended initially)**
- Run AAT as a subprocess for a full sprint
- Parse artifacts to extract phase-level traces
- Attribute rewards to decisions within each phase
- Limitation: cannot stop mid-sprint

**Approach B: Library integration (more powerful, future)**
- Import AAT's `SprintManager`, `StoryRefinementSession`, `TechnicalPlanningSession`, etc. as Python modules
- Call individual ceremony methods directly
- Full control over episode boundaries
- Requires AAT to be installable as a package

### 5. Multi-Team and Borrowing Episodes

AAT's multi-team mode is accessed via the `teams:` config section:

```yaml
teams:
  - id: "team-alpha"
    agents: [ahmed_senior_dev_lead, alex_senior_networking, ...]
  - id: "team-beta"
    agents: [training_candidate_agent, marcus_mid_backend, ...]

coordination:
  enabled: true
  max_borrows_per_sprint: 2
  coordinators: [staff_engineer_01, enablement_lead_01]
```

For borrowing episodes, the training candidate starts in one team and gets borrowed to another. The reward pipeline measures orientation speed and contribution quality in the receiving team.

### 6. Disturbance Configuration

AAT's disturbance engine is controlled via config:

```yaml
# Stage 1: No disturbances
disturbances:
  enabled: false

# Stage 2: Full disturbances
disturbances:
  enabled: true
  frequencies:
    dependency_breaks: 0.166
    production_incident: 0.125
    flaky_test: 0.25
    scope_creep: 0.20
    junior_misunderstanding: 0.33
    architectural_debt_surfaces: 0.166
    merge_conflict: 0.30
```

The training pipeline varies disturbance frequencies across episodes for curriculum diversity.

### 7. Backlog Generation

Each episode needs a backlog. The training pipeline generates backlogs by:
- Sampling from a curated task bank (see `specs/TRAINING_EPISODES.md`)
- Varying ambiguity levels (fully specified → intentionally vague)
- Varying domains (CRUD, data pipeline, frontend, library, distributed system)
- Varying languages (Python, TypeScript, Go, Rust — using AAT's multi-language support)

```yaml
# Example generated backlog for an elicitation episode
product:
  name: "Training Episode Task"
  languages: [python]

stories:
  - id: US-TRAIN-001
    title: "Add caching to the API"  # Intentionally vague
    description: "We need caching for better performance"
    acceptance_criteria: []  # Intentionally empty — model must ask
    story_points: 5
    priority: 1
```

## AAT Features Used Per Training Stage

| AAT Feature | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Single team sprint | ✅ Primary | ✅ Primary | ✅ Evaluation | ✅ Evaluation |
| Story refinement | ✅ Elicitation episodes | ✅ | ✅ | ✅ |
| Technical planning | ✅ Decomposition episodes | ✅ | ✅ | ✅ |
| Pairing sessions | ✅ Implementation episodes | ✅ | ✅ | ✅ |
| Daily standups | ✅ Self-monitoring | ✅ | ✅ | ✅ |
| Sprint review | ✅ Outcome signal | ✅ | ✅ | ✅ |
| Retrospective | ✅ Meta-learning signal | ✅ | ✅ | ✅ |
| Disturbances | ❌ | ✅ Primary | ✅ | ✅ |
| Profile swapping | ❌ | ✅ Constrained | ✅ | ✅ |
| Multi-team mode | ❌ | ❌ | ✅ Primary | ✅ |
| Agent borrowing | ❌ | ❌ | ✅ Primary | ✅ |
| Coordination loop | ❌ | ❌ | ✅ | ✅ |
| Specialist consultant | ❌ | ✅ | ✅ | ✅ |
| Turnover/attrition | ❌ | ❌ | ❌ | ✅ Primary |
| Brownfield mode | ✅ Some episodes | ✅ | ✅ | ✅ |
| Remote git | Optional | Optional | Optional | Optional |

## AAT Modifications NOT Required

The following are intentionally handled by the wrapper, not by modifying AAT:

- **Decision tracing**: The wrapper intercepts model inputs/outputs at the runtime level
- **Reward computation**: Computed from artifacts after episodes, not during
- **Episode control**: Achieved via config generation and CLI invocation
- **Curriculum management**: Entirely external to AAT

## Potential AAT Enhancements (Optional, Future)

If the project matures, these AAT enhancements would improve integration:

1. **Installable as a Python package** (`pip install agile-agent-team`) — enables library-level integration instead of subprocess
2. **Event hooks** — callbacks at ceremony boundaries for phase-level episode control
3. **Decision ID injection** — AAT assigns unique IDs to each agent decision for attribution
4. **Structured trace export** — beyond artifacts, export full agent reasoning traces in a standard format
5. **Attrition implementation** — currently described but not fully implemented; needed for Stage 4

These are enhancements, not requirements. The project can proceed with subprocess-level integration.
