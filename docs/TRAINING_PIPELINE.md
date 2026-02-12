# Training Pipeline

## End-to-End Flow

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐
│  Stage 1    │───▶│   Stage 2    │───▶│  Stage 3   │───▶│  Stage 4   │
│  Stable     │    │  Disturbance │    │  Cross-    │    │  Team      │
│  Team       │    │              │    │  Team      │    │  Change    │
└─────────────┘    └──────────────┘    └────────────┘    └────────────┘
      │                  │                   │                  │
      ▼                  ▼                   ▼                  ▼
  LoRA v1            LoRA v2             LoRA v3           LoRA v4
  (foundation)       (resilience)        (orientation)     (adaptation)
```

## Stage 1: Stable Team Foundation

### Environment Configuration

- Single team, 11 agents
- No disturbances (`disturbances.enabled: false`)
- No profile swapping (`profile_swapping.mode: none`)
- Diverse backlogs (CRUD APIs, data pipelines, frontend apps, libraries)

### Episode Types

**Elicitation episodes** (primary):
- Input: Vague user story from backlog
- Model role: Senior developer in story refinement session
- Actions: Ask clarifying questions, probe scope, identify missing criteria
- Reward: Did questions change implementation? PO acceptance rate? Rework count?
- Target: 500-1000 episodes

**Decomposition episodes**:
- Input: Refined story with acceptance criteria
- Model role: Senior in technical planning (Phase 2, no PO)
- Actions: Break into tasks, identify dependencies, estimate complexity, assign approach
- Reward: Task parallelization effectiveness, dependency accuracy, estimation accuracy
- Target: 500-1000 episodes

**Implementation episodes**:
- Input: Task assignment in pairing session
- Model role: Driver or navigator in pair
- Actions: Write code, run tests, checkpoint dialogue, self-correct on failures
- Reward: Tests passing, iteration count (fewer is better), checkpoint decision quality
- Target: 1000-2000 episodes

**Self-monitoring episodes**:
- Input: Mid-implementation state (partially complete, possibly stuck)
- Model role: Agent at 50% checkpoint
- Actions: Continue current approach, pivot to alternative, escalate to lead, request research
- Reward: Did the choice lead to completion? Was rework needed?
- Target: 500-1000 episodes

### Graduation Criteria

- Solo evaluation: elicitation quality score > 0.7 (judge evaluator)
- Solo evaluation: decomposition quality > 0.65
- Solo evaluation: self-correction rate > 0.5 (catches own mistakes at least half the time)
- Sprint-level evaluation: at least 2/3 stories pass QA in a full sprint

## Stage 2: Disturbance Resilience

### Environment Configuration

- Single team, 11 agents
- All disturbances enabled (realistic frequencies from agile-agent-team defaults)
- Constrained profile swapping
- Blast radius controls active

### Episode Types

**Triage episodes** (primary):
- Input: Mid-sprint disturbance injection (production incident, dependency break, flaky test)
- Model role: Senior developer receiving disturbance
- Actions: Assess severity, estimate blast radius, decide response (fix immediately, defer, escalate)
- Reward: Velocity impact within bounds? Recovery time? Quality maintained?
- Target: 500-1000 episodes

**Replanning episodes**:
- Input: Scope creep or architectural debt surfacing mid-sprint
- Model role: Agent who must adjust plan
- Actions: Re-estimate remaining work, re-prioritize tasks, communicate to team, adjust approach
- Reward: Sprint still delivers committed stories? New scope handled appropriately?
- Target: 300-500 episodes

**Recovery episodes**:
- Input: Post-disturbance state (some work lost, tests failing, blocked cards)
- Model role: Agent leading recovery
- Actions: Diagnose root cause, plan fix, implement fix, verify recovery
- Reward: Time to green (tests passing), minimal collateral damage
- Target: 300-500 episodes

### Graduation Criteria

- Disturbance episodes: triage accuracy > 0.7 (correct severity assessment)
- Sprint-level evaluation: velocity drop < 30% during disturbance sprints
- Solo evaluation: adaptation quality score > 0.65 (responds well to unexpected failures)

## Stage 3: Cross-Team Orientation

### Environment Configuration

- Multi-team mode (2-3 teams)
- Coordination enabled with agent borrowing
- Cross-team dependencies in backlog
- Mixed backlogs (different domains per team)

### Episode Types

**Orientation episodes** (primary):
- Input: Agent borrowed into unfamiliar team mid-sprint
- Model role: The borrowed agent
- Actions: Read existing code, ask team about conventions, understand current sprint state, identify how to contribute
- Reward: Time to first productive contribution, convention adherence, receiving team velocity impact
- Target: 300-500 episodes

**Context-switching episodes**:
- Input: Agent must switch from one task domain to another (e.g., after borrowing return)
- Model role: Agent returning home or switching tasks
- Actions: Re-orient to home team state, resume previous work, apply learnings from away
- Reward: Productivity recovery time, knowledge brought back
- Target: 200-300 episodes

**Dependency identification episodes**:
- Input: Story that has implicit cross-team dependencies
- Model role: Senior developer during planning
- Actions: Identify that dependency exists, flag to coordinator, propose resolution approach
- Reward: Were real dependencies caught? Were false dependencies avoided?
- Target: 200-300 episodes

### Graduation Criteria

- Orientation episodes: productive contribution within first 3 actions (vs. 6+ for untrained)
- Solo evaluation: codebase orientation score > 0.7 (given unfamiliar repo, can navigate effectively)
- Sprint-level evaluation: borrowed agent improves receiving team velocity (net positive)

## Stage 4: Team Composition Change

### Environment Configuration

- Single or multi-team
- Attrition enabled (agent departure mid-experiment)
- New hire onboarding (fresh agent joining)
- Full disturbances + borrowing

### Episode Types

**Knowledge handoff episodes**:
- Input: Agent is departing the team in N sprints
- Model role: The departing senior
- Actions: Document decisions, ensure pairing transfers knowledge, prioritize handoff tasks
- Reward: Team velocity after departure (smaller dip = better handoff)
- Target: 200-300 episodes

**Onboarding episodes**:
- Input: New agent joins the team
- Model role: Existing senior welcoming new hire
- Actions: Orient new member, adjust pairing schedule, identify knowledge gaps, prioritize learning
- Reward: New member time to productivity, team velocity recovery
- Target: 200-300 episodes

**Compensation episodes**:
- Input: Key team member just left, knowledge gap exists
- Model role: Remaining senior who must cover
- Actions: Identify knowledge gaps, determine what can be recovered from artifacts vs. needs rediscovery, adjust sprint commitment
- Reward: Team maintains delivery despite loss, knowledge recovery rate
- Target: 200-300 episodes

### Graduation Criteria

- Solo evaluation: all Stage 1-3 metrics maintained (no regression)
- Handoff quality: team velocity dip < 20% after departure when model was departing agent
- Solo evaluation: when given a codebase with no documentation, can reconstruct intent and architecture

## Backlog Diversity

Training quality depends heavily on backlog diversity. The following dimensions should be varied across episodes:

| Dimension | Variations |
|---|---|
| Domain | CRUD APIs, data pipelines, ML services, frontend SPAs, CLI tools, libraries |
| Language | Python, TypeScript, Go, Rust (via AAT multi-language support) |
| Complexity | 1-point stories through 8-point stories |
| Ambiguity | Fully specified → intentionally vague |
| Scale | Greenfield small app → brownfield large codebase |
| Technical debt | Clean codebase → legacy with known issues |

The `examples/` directory in agile-agent-team provides starting points, but the training pipeline should generate additional synthetic backlogs for coverage.

## Hyperparameters (Initial Estimates)

| Parameter | Value | Rationale |
|---|---|---|
| Episodes per stage | 2000-5000 | Enough for behavioral convergence |
| Batch size | 16-32 episodes | Balance between variance and compute |
| PPO clip | 0.2 | Standard for language model RL |
| LoRA rank | 16-64 | Sufficient for behavioral adaptation |
| Learning rate | 1e-5 to 5e-5 | Conservative to preserve base capabilities |
| Evaluation frequency | Every 100 episodes | Catch regressions early |
| Judge evaluation | Every 50 episodes | More expensive, sample |

These are starting points. Expect significant tuning.

## Compute Estimates

### Per Episode (Phase-Level)

- Training model inference: 5-20 calls × ~1s each = 5-20s
- Environment agent inference: 10-50 calls × ~1s each = 10-50s
- Judge evaluation: 1 call × ~5s = 5s
- Total: ~30-75s per phase-level episode

### Per Stage

- ~3000 episodes × ~60s average = ~50 hours of wall-clock time
- With 4x parallelism: ~12.5 hours per stage
- Total training: ~50 hours across all 4 stages

### Total Pipeline

- Training: ~200 GPU-hours (estimate, depends on model size and parallelism)
- Judge evaluations: ~$200-500 in API costs (Claude Opus for behavioral evaluation)
- Evaluation harness: ~20 GPU-hours for periodic full evaluations

These are rough estimates. Actual costs depend on model size, episode complexity, and convergence speed.
