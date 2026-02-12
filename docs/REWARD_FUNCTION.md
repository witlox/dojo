# Reward Function Design

## Principles

1. **Reward upstream behavior, not just downstream outcomes.** Code quality is easy to measure but is a lagging indicator. The reasoning that led to good code is the leading indicator we want to reinforce.
2. **Penalize both insufficient AND excessive behavior.** A model that asks 20 questions for a trivial task is as dysfunctional as one that asks zero questions for an ambiguous task. Calibration matters.
3. **Two independent evaluation channels cross-validate.** Outcome-based reward (from sprint metrics) and behavioral reward (from judge evaluator) must agree. Disagreement signals a rubric problem, not a model problem.
4. **Reward shifts across curriculum stages.** Early stages weight behavioral adherence; late stages weight outcomes. The model must first learn *how* to think, then demonstrate that the thinking *produces results*.

## Reward Components

### 1. Outcome Reward (from Sprint Artifacts)

Computed from agile-agent-team's measurable outputs:

```python
outcome_reward = weighted_sum(
    story_accepted      * 0.30,  # PO accepted the story in sprint review
    tests_passing       * 0.20,  # All tests pass on first or second iteration
    low_iteration_count * 0.15,  # Fewer test-fix iterations = better first attempt
    velocity_maintained * 0.15,  # Sprint velocity within expected range
    low_rework          * 0.10,  # Card didn't bounce back from review to in_progress
    coverage_threshold  * 0.10,  # Met coverage targets
)
```

Each component is normalized to [0, 1].

### 2. Behavioral Reward (from Judge Evaluator)

A large model (Claude Opus) scores the reasoning trace against the behavioral taxonomy. The judge receives:
- The task context (story, codebase state, team state)
- The model's full action sequence with reasoning
- The behavioral rubric for the relevant patterns

```python
behavioral_reward = judge_score(
    trace=model_reasoning_trace,
    rubric=taxonomy_patterns_for_episode_type,
    context=task_and_environment_state
)
```

The judge returns a score in [0, 1] with justification. Justifications are logged for reward function debugging.

### 3. Efficiency Penalty

Penalizes unnecessary actions — the "performative seniority" problem where the model learns to go through motions without substance:

```python
efficiency_penalty = -penalty_weight * max(0, actions_taken - baseline_actions) / baseline_actions
```

`baseline_actions` is the median action count for successful episodes of the same type and difficulty. Models that solve problems in fewer actions than the baseline receive a small bonus.

### 4. Phase Completion Bonus

Sparse bonus for completing a phase transition correctly:

```python
phase_bonus = {
    "elicitation_to_planning": 0.1,    # Correctly decided to stop asking and start planning
    "planning_to_execution":   0.1,    # Produced viable plan before coding
    "checkpoint_continue":     0.05,   # Correctly continued at checkpoint
    "checkpoint_pivot":        0.15,   # Correctly pivoted at checkpoint (harder to learn)
    "checkpoint_escalate":     0.10,   # Correctly escalated (requires recognizing limits)
}
```

## Composite Reward Formula

```python
composite_reward = (
    outcome_weight     * outcome_reward +
    behavioral_weight  * behavioral_reward +
    efficiency_penalty +
    phase_bonus
)
```

### Weight Schedule Across Stages

| Stage | Outcome Weight | Behavioral Weight | Rationale |
|---|---|---|---|
| 1 (Stable) | 0.3 | 0.7 | Learn patterns first, outcomes follow |
| 2 (Disturbance) | 0.4 | 0.6 | Outcomes matter more under pressure |
| 3 (Cross-team) | 0.5 | 0.5 | Balanced — both matter in complex environments |
| 4 (Team change) | 0.6 | 0.4 | Must produce results in challenging conditions |

## Per-Phase Reward Details

### Elicitation Phase

**Positive signals:**
- Questions that changed the implementation (traced via decision attribution)
- Questions covering multiple dimensions (scope, constraints, success criteria, edge cases)
- Questions ordered by impact (most important first)
- Stopping when information is sufficient (sufficiency detection)

**Negative signals:**
- Generic questions ("Can you tell me more?") when specific ones were possible
- Asking about things already stated in the story description
- Asking too many questions for a well-specified task
- Not asking enough questions for an ambiguous task
- Questions that didn't influence any downstream decision

**Measurement:**
```python
elicitation_reward = weighted_sum(
    question_specificity_score  * 0.25,  # Judge evaluates
    question_coverage_score    * 0.25,  # Judge evaluates
    information_utilization    * 0.25,  # Traced: were answers used?
    sufficiency_calibration    * 0.25,  # Stopped at right time?
)
```

### Decomposition Phase

**Positive signals:**
- Tasks are independently implementable (low coupling)
- Dependencies identified were real (validated by execution)
- Task granularity appropriate (not too coarse, not too fine)
- Riskiest task identified first
- Approach justified with trade-off analysis

**Negative signals:**
- Tasks with circular dependencies
- Missed dependencies that caused blocks during execution
- Over-decomposition (10 tasks for a 3-point story)
- Under-decomposition (1 monolithic task)
- No risk assessment

**Measurement:**
```python
decomposition_reward = weighted_sum(
    task_independence          * 0.20,  # Execution: did tasks parallelize?
    dependency_accuracy        * 0.25,  # Execution: were predicted deps real?
    granularity_calibration    * 0.20,  # Judge evaluates against story size
    risk_identification        * 0.20,  # Did predicted risks materialize?
    justification_quality      * 0.15,  # Judge evaluates trade-off reasoning
)
```

### Implementation Phase

**Positive signals:**
- Tests pass on first attempt
- Clean checkpoint decisions (correct continue/pivot/escalate)
- Self-correction without external prompting
- Code matches the plan from decomposition phase
- Appropriate test coverage

**Negative signals:**
- Tests fail repeatedly (> 2 iterations)
- Diverging from plan without documented reason
- Ignoring failing tests
- Not running tests before committing
- Checkpoint decisions that led to rework

**Measurement:**
```python
implementation_reward = weighted_sum(
    first_attempt_success     * 0.25,  # Tests pass on iteration 1
    checkpoint_quality        * 0.25,  # Correct decisions at each checkpoint
    self_correction_rate      * 0.20,  # Caught and fixed own mistakes
    plan_adherence            * 0.15,  # Followed or deliberately adjusted plan
    test_coverage             * 0.15,  # Met coverage thresholds
)
```

### Adaptation Phase (Disturbance Response)

**Positive signals:**
- Correct severity assessment within first 2 actions
- Blast radius correctly estimated
- Appropriate response (fix now vs. defer vs. escalate)
- Recovery without breaking other things
- Learning captured (meta-learning entry quality)

**Negative signals:**
- Panic response (dropping everything for low-severity issue)
- Ignoring high-severity issue
- Fix that introduces new problems
- No post-incident learning captured
- Excessive time spent on diagnosis vs. fix

**Measurement:**
```python
adaptation_reward = weighted_sum(
    severity_accuracy          * 0.25,  # Correct triage
    response_appropriateness   * 0.25,  # Right action for severity level
    recovery_efficiency        * 0.20,  # Time to resolution
    collateral_damage          * 0.15,  # Didn't break other things
    learning_captured          * 0.15,  # Quality of post-incident insight
)
```

### Orientation Phase (Cross-Team)

**Positive signals:**
- Reads existing code before proposing changes
- Asks about team conventions early
- First contribution respects existing patterns
- Identifies how to help based on team's actual needs (not assumptions)
- Communicates status to both home and receiving team

**Negative signals:**
- Immediately proposing changes without understanding context
- Imposing home team's conventions on receiving team
- Working on low-priority items while high-priority items are blocked
- Not communicating with either team
- Taking too long to orient (diminishing returns)

**Measurement:**
```python
orientation_reward = weighted_sum(
    context_gathering_quality  * 0.25,  # Judge evaluates
    convention_adherence       * 0.25,  # Code follows receiving team patterns
    contribution_relevance     * 0.25,  # Worked on what team actually needed
    time_to_productivity       * 0.15,  # Actions before first useful contribution
    communication_quality      * 0.10,  # Kept both teams informed
)
```

## Reward Calibration

### Cross-Validation Protocol

Every N episodes, compare judge evaluator scores with outcome scores:

```python
for episode in calibration_batch:
    behavioral = judge.evaluate(episode.trace)
    outcome = compute_outcome_reward(episode.artifacts)
    
    if behavioral > 0.7 and outcome < 0.3:
        flag("Judge may be rewarding performative behavior")
    if behavioral < 0.3 and outcome > 0.7:
        flag("Judge may be missing effective but unconventional strategies")
```

Flagged episodes are reviewed and the judge rubric is updated.

### Baseline Establishment

Before training begins, run 100+ episodes with a strong model (Claude Opus) to establish reward baselines per episode type. These baselines define:
- Expected action counts (for efficiency penalty calibration)
- Expected reward ranges (for normalization)
- Outcome distributions (for detecting training collapse)

### Reward Shaping Anti-Patterns to Avoid

1. **Don't reward question count.** Reward question *quality* and *impact*. Otherwise the model learns to ask many mediocre questions.
2. **Don't reward plan length.** Reward plan *tractability* and *accuracy*. Otherwise the model produces elaborate but useless plans.
3. **Don't reward speed alone.** A model that rushes through elicitation to get to coding faster will score well on efficiency but poorly on outcomes. The composite reward must balance these.
4. **Don't penalize all pivots.** Pivoting at a checkpoint because new information invalidates the approach is *good* senior behavior. Only penalize pivots that indicate poor initial planning.
