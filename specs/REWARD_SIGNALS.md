# Reward Signals

## Overview

This document maps each behavioral pattern from the taxonomy to its concrete, measurable reward signals. Each signal is tagged with its source (outcome-based from AAT artifacts, or behavioral from judge evaluator).

## Signal Sources

| Source | Type | Latency | Reliability |
|---|---|---|---|
| AAT sprint artifacts | Outcome | End of episode | High (objective) |
| AAT test results | Outcome | During episode | High (binary) |
| AAT kanban transitions | Outcome | End of episode | High (objective) |
| Judge evaluator (Opus) | Behavioral | Post-episode | Medium (subjective) |
| Decision trace analysis | Behavioral | Post-episode | Medium (heuristic) |

## Per-Behavior Reward Signals

### B-01: Ambiguity Detection

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Question specificity | Judge | Score 0-1 on each question's specificity to task context | 0.25 |
| Gap coverage | Trace | Dimensions addressed / total dimensions needing address | 0.25 |
| Implementation change rate | Outcome | Questions whose answers changed implementation / total Qs | 0.30 |
| Non-redundancy | Trace | 1 - (questions about already-stated info / total Qs) | 0.20 |

### B-02: Scope Boundary Probing

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Explicit scope question asked | Trace | Binary: did model ask "what's NOT in scope?" or equivalent | 0.30 |
| Scope match | Outcome | Final output scope matches intended scope (judge compares) | 0.40 |
| Gold-plating penalty | Outcome | -penalty for features built beyond acceptance criteria | 0.30 |

### B-03: Constraint Discovery

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Constraints discovered early | Trace+Outcome | Constraints asked about before execution / constraints encountered during | 0.40 |
| Constraint categories covered | Trace | Categories addressed (performance, integration, security, deployment) / relevant categories | 0.30 |
| False constraint penalty | Trace | -penalty for asking about constraints obvious from context | 0.30 |

### B-04: Sufficiency Detection

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Question count calibration | Trace+Judge | 1 - abs(actual_questions - optimal_questions) / optimal_questions | 0.40 |
| Explicit transition signal | Trace | Binary: did model explicitly signal readiness to proceed? | 0.20 |
| Remaining assumptions stated | Trace | Binary: did model list assumptions it will validate during execution? | 0.20 |
| Outcome quality | Outcome | Story accepted by PO without major rework | 0.20 |

### B-05: Risk-First Ordering

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Risky task position | Outcome | 1 - (position of riskiest task / total tasks) | 0.35 |
| Risk materialization prediction | Outcome | Predicted risks that materialized / total predicted risks | 0.35 |
| Spike/prototype for uncertainty | Trace | Binary: did model build prototype for uncertain component? | 0.30 |

### B-06: Dependency Identification

| Signal | Source | Computation | Weight |
|---|---|---|---|
| True positive rate | Outcome | Real deps identified / real deps encountered | 0.40 |
| False positive rate | Outcome | -penalty for declared deps that weren't real | 0.30 |
| Dependency typing accuracy | Outcome | Correct hard/soft classification / total deps | 0.30 |

### B-07: Appropriate Granularity

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Tasks-per-point ratio | Trace | 1 if tasks/story_point ∈ [0.5, 2.0], decaying outside | 0.40 |
| Task completability | Outcome | Tasks completed in one session / total tasks | 0.30 |
| Task testability | Judge | Each task has clear done state (judge evaluates) | 0.30 |

### B-08: Trade-Off Articulation

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Approaches considered | Trace | min(approaches_enumerated, 3) / 3 | 0.25 |
| Comparison quality | Judge | Score on trade-off reasoning (benefits, drawbacks, conditions) | 0.25 |
| Selection justified | Judge | Binary: did model explain why it chose the selected approach? | 0.25 |
| Approach survived | Outcome | Selected approach worked without major pivot | 0.25 |

### B-09: Progress Calibration

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Checkpoint accuracy | Outcome | Checkpoint assessment matched actual state (judge compares) | 0.40 |
| Decision correctness | Outcome | Continue/pivot/escalate decision led to good outcome | 0.40 |
| Assessment honesty | Judge | Score on candor vs. optimism bias | 0.20 |

### B-10: Stuck Detection

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Detection speed | Trace | 1 / iterations_before_recognizing_stuck (capped) | 0.30 |
| Diagnosis accuracy | Outcome | Correct categorization (knowledge gap / wrong approach / blocker) | 0.35 |
| Resolution effectiveness | Outcome | Action after detection resolved the issue | 0.35 |

### B-11: Error Diagnosis Quality

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Root cause identified | Outcome | Fix addressed root cause (not symptoms) | 0.40 |
| Hypothesis formed before fix | Trace | Binary: did model state hypothesis before changing code? | 0.25 |
| Single-change discipline | Trace | Made one targeted change per iteration (not shotgun) | 0.20 |
| Error message read | Trace | Binary: did model reference specific error details? | 0.15 |

### B-12: Knowing Your Limits

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Self-assessment accuracy | Outcome | Stated confidence correlated with actual correctness | 0.40 |
| Appropriate help-seeking | Outcome | Sought help when actually needed, didn't when not needed | 0.35 |
| Gap categorization | Trace | Correctly identified researchable vs. expert-required gaps | 0.25 |

### B-13: Severity Triage

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Severity accuracy | Outcome | Assessed severity matched actual impact | 0.35 |
| Blast radius accuracy | Outcome | Assessed scope matched actual scope | 0.30 |
| Response match | Judge | Response type (fix/defer/escalate) appropriate for severity | 0.35 |

### B-14: Graceful Replanning

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Plan adjustment proportionality | Judge | Adjustment size matched disruption severity | 0.30 |
| Preserved valid components | Trace | Kept what still works, changed only what needed changing | 0.25 |
| Communication of change | Trace | Binary: communicated what changed and why | 0.20 |
| Outcome after replan | Outcome | Sprint delivery after replanning | 0.25 |

### B-15: Post-Incident Learning

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Root cause captured | Judge | Learning addresses root cause, not symptoms | 0.35 |
| Prevention identified | Judge | Learning includes actionable prevention | 0.30 |
| Non-recurrence | Outcome | Same issue type doesn't recur in subsequent episodes | 0.35 |

### B-16 & B-17: Research Behaviors

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Query specificity | Judge | Search queries specific to actual need (not generic) | 0.20 |
| Source authority | Judge | Used authoritative sources (docs > blogs > forums) | 0.20 |
| Recency check | Trace | Binary: verified information is current | 0.15 |
| Information utilization | Outcome | Retrieved info actually used in implementation | 0.25 |
| Search efficiency | Trace | 1 / searches_needed (capped) | 0.20 |

### B-18, B-19, B-20: Orientation Behaviors

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Read-before-write ratio | Trace | Read actions before first write / total early actions | 0.20 |
| Convention violations | Outcome | 1 - violations in first contribution / total code lines | 0.25 |
| Time to productivity | Trace | 1 / actions_before_first_useful_contribution (capped) | 0.25 |
| Context questions quality | Judge | Questions targeted at critical unknowns | 0.15 |
| Contribution relevance | Outcome | Worked on team's actual priority (not self-selected) | 0.15 |

### B-25 & B-26: Scoping Behaviors

| Signal | Source | Computation | Weight |
|---|---|---|---|
| Acceptance criteria coverage | Outcome | Code addresses all AC / total AC | 0.35 |
| Excess code penalty | Outcome | -penalty for code beyond AC scope | 0.30 |
| Core-first timing | Trace | Core functionality completed in first half of episode | 0.35 |

### B-27, B-28, B-29, B-30: Meta-Learning Behaviors

These are measured across multiple episodes rather than within a single episode:

| Signal | Source | Computation |
|---|---|---|
| Pattern reuse when appropriate | Cross-episode | Similar problems solved with proven approaches |
| Mistake non-repetition | Cross-episode | Same error pattern doesn't recur across episodes |
| Behavioral change after feedback | Cross-episode | Measurable change in subsequent episode after negative reward |
| Confidence-accuracy correlation | Cross-episode | Stated confidence predicts actual correctness (r > 0.6) |

## Reward Aggregation

### Per-Episode Composite

```
composite = outcome_weight * Σ(outcome_signals) + behavioral_weight * Σ(judge_signals) + efficiency_penalty + phase_bonus
```

### Cross-Episode (Meta-Learning)

```
meta_reward = pattern_reuse_score + non_repetition_score + feedback_integration_score + calibration_score
```

Applied as a bonus/penalty to episode rewards based on rolling window of last N episodes.
