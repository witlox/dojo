# Evaluation Harness

## Purpose

The evaluation harness tests whether behavioral patterns learned in the multi-agent team environment transfer to solo deployment — the actual product use case. The training environment is complex (11 agents, sprints, ceremonies). The deployment is simple (one model, one human developer, one task).

## Solo Evaluation Protocol

### Setup

The trained model is placed in a solo context:
- No team, no pairing partner, no PO, no lead dev
- One human developer (simulated by a fixed model or scripted responses)
- One task (varying complexity and ambiguity)
- One codebase (optional, for brownfield scenarios)
- Access to retrieval tools (web search, documentation lookup)

### Evaluation Flow

```
Phase 1: ELICITATION
  Input: Vague task description
  Model should: Ask clarifying questions
  Human provides: Answers (scripted or from proxy model)
  Measure: Question quality, coverage, sufficiency detection
  End: Model signals readiness to proceed

Phase 2: RESEARCH
  Input: Context from Phase 1
  Model should: Search for relevant information, evaluate sources
  Tools available: Web search, documentation lookup, codebase search
  Measure: Search strategy quality, source relevance, information integration
  End: Model signals sufficient knowledge gathered

Phase 3: PLANNING
  Input: Context from Phases 1-2
  Model should: Decompose task, identify risks, select approach
  Measure: Plan tractability, risk identification, trade-off reasoning
  End: Model produces implementation plan

Phase 4: EXECUTION
  Input: Plan from Phase 3
  Model should: Implement iteratively, self-correct, checkpoint
  Tools available: File operations, test execution, git
  Measure: First-attempt quality, self-correction, checkpoint decisions
  End: Implementation complete and tested

Phase 5: VERIFICATION
  Input: Completed implementation
  Model should: Verify against original intent, identify gaps
  Measure: Intent match, completeness, quality
  End: Model signals done or identifies remaining work
```

### Task Bank

Tasks should span multiple dimensions:

| Dimension | Easy | Medium | Hard |
|---|---|---|---|
| Ambiguity | Fully specified with examples | Missing some constraints | One-line description |
| Domain | CRUD API endpoint | Data pipeline with transforms | Distributed system component |
| Codebase | Greenfield (no existing code) | Small existing project | Large legacy codebase |
| Complexity | Single function | Multi-module feature | Cross-cutting concern |
| Risk | Low (well-understood pattern) | Medium (some unknowns) | High (novel problem domain) |

Target: 50-100 evaluation tasks across these dimensions.

### Human Proxy

For automated evaluation, the human side of the conversation is handled by:
- **Scripted responses**: For elicitation, predetermined answers to expected question categories
- **Proxy model**: A fixed large model (not the training target) that role-plays a developer with specific knowledge and constraints
- **Recorded human sessions**: Real developer interactions captured and replayed

The proxy must be consistent across evaluations to enable fair comparison.

## Metrics

### Primary Metrics (Transfer Indicators)

| Metric | Definition | Target | Source |
|---|---|---|---|
| Elicitation quality | Judge score on question specificity, coverage, ordering | > 0.70 | Judge evaluator |
| Sufficiency calibration | Did model stop asking at the right time? (not too early, not too late) | Error < 0.15 | Judge + outcome |
| Research effectiveness | Fraction of retrieved information that was actually used | > 0.60 | Trace analysis |
| Decomposition accuracy | Did the plan's assumptions hold during execution? | > 0.65 | Outcome |
| Self-correction rate | Fraction of own mistakes caught before completion | > 0.50 | Execution trace |
| Adaptation quality | Response quality when execution doesn't go as planned | > 0.60 | Judge evaluator |
| Intent match | Final output matches what was originally requested | > 0.75 | Outcome evaluator |

### Secondary Metrics (Behavioral Health)

| Metric | Definition | Concern If |
|---|---|---|
| Question count calibration | Questions asked vs. task ambiguity level | High questions for clear tasks, low for ambiguous |
| Research depth calibration | Searches performed vs. knowledge gap size | Excessive searching for known patterns |
| Plan-to-execution drift | How much execution diverged from plan | Large drift without documented reason |
| Action efficiency | Total actions vs. minimum necessary | > 2x minimum consistently |
| Escalation appropriateness | When model says "I need human help" | Never escalates, or escalates on trivial issues |

### Transfer Score

The headline metric: how well do team-trained behaviors transfer to solo context?

```python
transfer_score = solo_metric / team_metric
```

Computed per behavioral pattern. A transfer score > 0.8 indicates good transfer. A score < 0.5 indicates the team training didn't help for that behavior in solo context.

## Evaluation Schedule

### During Training

- **Every 100 episodes**: Quick evaluation (10 tasks, primary metrics only)
- **Every 500 episodes**: Full evaluation (50 tasks, all metrics)
- **Each stage completion**: Comprehensive evaluation (100 tasks, transfer score calculation)

### Post-Training

- **Full benchmark**: All 100+ tasks, all metrics, transfer scores per behavior
- **Ablation studies**: Compare Stage 1 only → Stage 1+2 → Stage 1+2+3 → Full to measure incremental value of each curriculum stage
- **Regression checks**: Verify base coding capability hasn't degraded

## Comparison Baselines

| Baseline | Description | Purpose |
|---|---|---|
| Base model (no training) | Same model without behavioral fine-tuning | Measures training effect |
| Distilled model | Model trained via knowledge distillation from large model | Compares RL vs. distillation |
| Large model (Opus/Sonnet) | Claude Opus on the same tasks | Ceiling performance |
| Prompt-only | Base model with detailed senior behavior prompt | Measures fine-tuning vs. prompting |
| AAT heuristic baseline | AAT's `BehavioralScorer` heuristic scores on untrained model | Measures training effect on detectable behaviors |

The key comparison is **trained model vs. prompt-only baseline**. If the fine-tuned model significantly outperforms a carefully prompted base model, the behavioral RL training has value beyond what prompting can achieve.

AAT's `BehavioralScorer` provides an additional automated baseline: comparing the number and quality of detected behavioral codes before and after training gives a fast, objective measure of behavioral improvement.

## Anti-Gaming Detection

The evaluation must detect models that have learned to *perform* senior behaviors without substance:

1. **Question quality audit**: Are questions actually specific to the task, or are they generic templates?
2. **Research utilization**: Did the model actually use what it searched for, or did it search performatively?
3. **Plan-execution coherence**: Does the implementation follow the plan, or were plan and execution independent?
4. **Escalation validity**: When the model says "I need help with X," is X actually beyond its capability?

Each evaluation episode is tagged with a "performative risk score" by the judge evaluator. High performative risk flags episodes for manual review.
