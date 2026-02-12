# Training Episodes

## Episode Structure

Each training episode is a bounded interaction between the candidate model and the simulation environment, designed to be short enough for efficient RL training while capturing meaningful behavioral signal.

### Common Format

```
1. CONTEXT INJECTION — Task, environment state, tools, behavioral expectation
2. ACTION LOOP — Observe → act → environment responds → tracer logs
3. OUTCOME COLLECTION — Artifacts, outcome eval, judge eval, composite reward
```

## Stage 1: Stable Team Foundation

### E-1.1: Elicitation Episode (~2 min, 5-15 actions)

**Setup**: Single user story (ambiguity 0.3-0.9), PO available for questions.

**Actions**: `ask_question`, `state_assumption`, `signal_ready`, `request_example`

**Reward**: Questions that changed implementation, dimension coverage, sufficiency calibration, judge score on quality.

**Trains**: B-01 (Ambiguity Detection), B-02 (Scope Probing), B-03 (Constraint Discovery), B-04 (Sufficiency Detection)

### E-1.2: Decomposition Episode (~3 min, 8-20 actions)

**Setup**: Refined story with criteria, optional brownfield codebase context. No PO.

**Actions**: `create_task`, `add_dependency`, `identify_risk`, `propose_approach`, `compare_approaches`, `signal_plan_complete`

**Reward**: Task independence (parallelization), dependency accuracy, risk identification, granularity calibration, approach survival.

**Trains**: B-05 (Risk-First Ordering), B-06 (Dependency ID), B-07 (Granularity), B-08 (Trade-Off Articulation)

### E-1.3: Implementation Episode (~10 min, 20-50 actions)

**Setup**: Task from decomposition, workspace with git repo, BDD feature file, tools available (filesystem, git, bash, test_runner).

**Actions**: `read_file`, `write_file`, `edit_file`, `run_tests`, `bash`, `git_add`, `git_commit`, `checkpoint_continue`, `checkpoint_pivot`, `checkpoint_escalate`

**Reward**: Tests passing, iteration count (fewer = better), checkpoint decision quality, self-correction rate, code quality (judge).

**Trains**: B-09 (Progress Calibration), B-10 (Stuck Detection), B-11 (Error Diagnosis), B-25 (YAGNI), B-26 (MVP Instinct)

### E-1.4: Self-Monitoring Episode (~3 min, 5-15 actions)

**Setup**: Mid-implementation state (~50% complete), some tests passing/failing, deliberately planted subtle issue, checkpoint triggered.

**Actions**: `assess_progress`, `identify_issue`, `checkpoint_continue`, `checkpoint_pivot`, `checkpoint_escalate`, `research`, `reread_requirements`

**Reward**: Detected planted issue? Correct checkpoint decision? Assessment matched reality? Time efficiency.

**Trains**: B-09 (Progress Calibration), B-10 (Stuck Detection), B-12 (Knowing Limits), B-30 (Confidence Calibration)

### E-1.5: Research Episode (~3 min, 5-15 actions)

**Setup**: Technical question requiring external information, web search and docs available, known correct answer for reward computation.

**Actions**: `search`, `fetch_doc`, `search_code`, `synthesize`, `assess_confidence`, `signal_sufficient`

**Reward**: Found correct information? Query quality? Source authority? Information utilization? Efficiency.

**Trains**: B-16 (Search Strategy), B-17 (Source Evaluation), B-04 (Sufficiency Detection)

## Stage 2: Disturbance Resilience

### E-2.1: Triage Episode (~3 min, 5-15 actions)

**Setup**: Mid-sprint, disturbance injected (incident, dependency break, flaky test, scope creep), current sprint state visible.

**Actions**: `assess_severity`, `assess_blast_radius`, `respond_fix_now`, `respond_defer`, `respond_escalate`, `communicate`, `investigate`

**Reward**: Severity accuracy, blast radius accuracy, response appropriateness, recovery time, velocity impact.

**Trains**: B-13 (Severity Triage), B-14 (Graceful Replanning), B-21 (Status Transparency)

### E-2.2: Recovery Episode (~5 min, 10-25 actions)

**Setup**: Post-disturbance state (failing tests, blocked card, lost work), time pressure from sprint deadline.

**Actions**: All implementation actions plus `diagnose_root_cause`, `verify_hypothesis`, `replan_sprint`, `capture_learning`

**Reward**: Time to green, root cause identified, collateral damage avoided, sprint delivery maintained, learning quality.

**Trains**: B-10 (Stuck Detection), B-11 (Error Diagnosis), B-14 (Graceful Replanning), B-15 (Post-Incident Learning)

### E-2.3: Scope Change Episode (~3 min, 5-15 actions)

**Setup**: Mid-sprint scope creep (new unplanned card), sprint near capacity, PO available for negotiation.

**Actions**: `assess_new_scope`, `negotiate_tradeoff`, `accept_scope`, `defer_scope`, `ask_po`

**Reward**: Sprint commitment protected? Tradeoff analysis sound? Sprint outcome successful? PO satisfied?

**Trains**: B-02 (Scope Probing), B-13 (Severity Triage), B-14 (Graceful Replanning), B-25 (YAGNI)

## Stage 3: Cross-Team Orientation

### E-3.1: Borrowing Arrival Episode (~5 min, 10-25 actions)

**Setup**: Model "borrowed" to unfamiliar team, different codebase and conventions, receiving team has specific need.

**Actions**: `read_file`, `search_code`, `ask_team`, `read_docs`, `identify_conventions`, `propose_contribution`, `start_work`, plus implementation actions

**Reward**: Time to first productive contribution, convention adherence, contribution relevance, question quality, receiving team velocity impact.

**Trains**: B-18 (Codebase Reconnaissance), B-19 (Convention Respect), B-20 (Context Acquisition), B-22 (Help Request Quality)

### E-3.2: Cross-Team Dependency Episode (~3 min, 5-15 actions)

**Setup**: Story with implicit cross-team dependency (not explicitly stated), cross-team communication available.

**Actions**: `analyze_story`, `search_architecture`, `flag_dependency`, `propose_resolution`, `communicate_cross_team`, `proceed_without`

**Reward**: Real dependencies identified, false dependencies avoided, resolution quality, communication timeliness.

**Trains**: B-06 (Dependency ID), B-03 (Constraint Discovery), B-21 (Status Transparency)

## Stage 4: Team Composition Change

### E-4.1: Knowledge Handoff Episode (~5 min, 10-20 actions)

**Setup**: Model is "departing" team, has unique knowledge about specific areas, team has gaps.

**Actions**: `identify_knowledge`, `document_decision`, `schedule_pairing`, `write_documentation`, `prioritize_handoff`, `verify_transfer`

**Reward**: Team velocity after departure, knowledge areas transferred, prioritization quality, documentation quality.

**Trains**: B-23 (Decision Documentation), B-24 (Knowledge Gap Handoff)

### E-4.2: Onboarding Support Episode (~5 min, 10-20 actions)

**Setup**: New agent joined team, model is existing senior helping onboard.

**Actions**: `assess_skills`, `identify_gaps`, `suggest_pairing`, `provide_context`, `assign_starter_task`, `adjust_sprint`

**Reward**: New member time to productivity, team velocity recovery, task assignment appropriateness, gap identification accuracy.

**Trains**: B-20 (Context Acquisition), B-07 (Appropriate Granularity), B-22 (Help Request Quality)

### E-4.3: Compensation Episode (~5 min, 10-20 actions)

**Setup**: Key member departed, knowledge area uncovered, sprint in progress with commitments.

**Actions**: `assess_impact`, `search_artifacts`, `recover_knowledge`, `adjust_commitment`, `request_specialist`, `redistribute_work`

**Reward**: Team delivery maintained, knowledge recovery rate, commitment adjustment appropriate, no quality regression.

**Trains**: B-10 (Stuck Detection), B-12 (Knowing Limits), B-13 (Severity Triage), B-14 (Graceful Replanning)

## Episode Generation & Diversity

Each episode type requires varied inputs. The pipeline generates by combining:

| Dimension | Variations |
|---|---|
| Domain | API, data pipeline, frontend, CLI, library, infrastructure |
| Language | Python, TypeScript, Go, Rust |
| Ambiguity | 0.1 (fully specified) to 0.9 (one-line description) |
| Codebase | Greenfield, small brownfield, large brownfield |
| Sprint context | Early (day 1-2), mid (day 3), late (day 4-5) |
| Disturbance | All 7 AAT types, varying severity |

## Target Episode Counts

| Stage | Types | Per Type | Total |
|---|---|---|---|
| 1 | 5 (E-1.1 to E-1.5) | 500-2000 | ~5000 |
| 2 | 3 (E-2.1 to E-2.3) | 300-500 | ~1200 |
| 3 | 2 (E-3.1 to E-3.2) | 200-500 | ~700 |
| 4 | 3 (E-4.1 to E-4.3) | 200-300 | ~700 |
| **Total** | **13 types** | | **~7600** |

Plus ~100 full-sprint evaluation episodes across training.
