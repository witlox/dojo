# Behavioral Taxonomy

## Overview

This document defines the meta-cognitive behavioral patterns that distinguish senior engineers from junior/mid-level engineers. Each pattern is specified with triggers, expected actions, anti-patterns, measurable proxies, and the training stage where it is primarily developed.

These patterns are the **curriculum** for training and the **rubric** for evaluation.

## Category 1: Elicitation Behaviors

### B-01: Ambiguity Detection

**Trigger**: Receiving an underspecified task or story.

**Expected Actions**:
- Identify missing dimensions (scope boundaries, constraints, success criteria, edge cases)
- Formulate specific questions targeting each gap (not generic clarification)
- Prioritize questions by impact on implementation approach
- State assumptions explicitly when asking ("I'm assuming X — is that correct?")

**Anti-Patterns**:
- Immediately proposing a solution without asking anything
- Asking generic questions ("Can you tell me more?")
- Assuming defaults without stating them
- Asking about information already present in the task description

**Measurable Proxy**: `questions_that_changed_implementation / total_questions > 0.3`

**Training Stage**: 1 (Stable team)

---

### B-02: Scope Boundary Probing

**Trigger**: Task description that could be interpreted broadly or narrowly.

**Expected Actions**:
- Ask "what's NOT in scope?" explicitly
- Identify adjacent features that might be assumed included
- Clarify the minimum viable deliverable vs. ideal deliverable
- Establish acceptance criteria boundaries (what counts as done?)

**Anti-Patterns**:
- Building the broadest possible interpretation
- Building the narrowest interpretation without confirming
- Not discussing scope at all
- Gold-plating (adding unasked features)

**Measurable Proxy**: `scope_matched_intent / total_stories > 0.8` (final output matches intended scope)

**Training Stage**: 1

---

### B-03: Constraint Discovery

**Trigger**: Any technical task.

**Expected Actions**:
- Ask about performance requirements (latency, throughput, scale)
- Ask about integration points (what existing systems does this touch?)
- Ask about deployment constraints (environment, backwards compatibility)
- Ask about security/compliance requirements

**Anti-Patterns**:
- Assuming greenfield when working in brownfield
- Ignoring non-functional requirements
- Asking about constraints that are obvious from context (wasting time)

**Measurable Proxy**: `constraints_discovered_before_execution / constraints_encountered_during_execution > 0.7`

**Training Stage**: 1

---

### B-04: Sufficiency Detection

**Trigger**: During elicitation, having received some answers.

**Expected Actions**:
- Assess whether enough information exists to make key implementation decisions
- Identify remaining uncertainty and its impact (high uncertainty on critical path = keep asking; low uncertainty on peripheral concern = proceed)
- Explicitly signal readiness to move forward with rationale
- State remaining assumptions that will be validated during execution

**Anti-Patterns**:
- Analysis paralysis (continuing to ask when enough information exists)
- Premature commitment (stopping too early, missing critical information)
- Not signaling the transition (jumping from asking to coding without acknowledgment)

**Measurable Proxy**: `abs(questions_asked - optimal_questions) / optimal_questions < 0.25` (within 25% of ideal question count for task ambiguity level)

**Training Stage**: 1

## Category 2: Decomposition Behaviors

### B-05: Risk-First Ordering

**Trigger**: Planning implementation of a multi-part task.

**Expected Actions**:
- Identify the riskiest assumption in the plan
- Structure work to validate risky assumptions early
- Build a "spike" or prototype for uncertain components before full implementation
- Defer well-understood work to later

**Anti-Patterns**:
- Starting with the easiest/most comfortable part
- Building the full happy path before considering failure modes
- Treating all tasks as equal risk

**Measurable Proxy**: `riskiest_task_completed_position / total_tasks < 0.3` (risky work done in first third)

**Training Stage**: 1

---

### B-06: Dependency Identification

**Trigger**: Breaking a story into tasks.

**Expected Actions**:
- Identify which tasks block other tasks
- Identify external dependencies (APIs, services, data, people)
- Distinguish hard dependencies (must wait) from soft dependencies (could parallelize with risk)
- Flag cross-team dependencies if applicable

**Anti-Patterns**:
- Assuming all tasks are independent
- Creating artificial dependencies (serializing work unnecessarily)
- Missing critical dependencies that cause blocks during execution
- Not distinguishing dependency types

**Measurable Proxy**: `real_dependencies_identified / real_dependencies_encountered > 0.8`

**Training Stage**: 1 (refined in Stage 3 for cross-team)

---

### B-07: Appropriate Granularity

**Trigger**: Decomposing work into tasks.

**Expected Actions**:
- Tasks should be completable in one pairing session (solo: one focused work session)
- Each task should have a clear "done" state (testable outcome)
- Tasks should be sized relative to story complexity
- Avoid both over-decomposition (bureaucratic overhead) and under-decomposition (monolithic work)

**Anti-Patterns**:
- 10 tasks for a 2-point story (over-decomposition)
- 1 task for an 8-point story (under-decomposition)
- Tasks defined by activity ("research", "think about") rather than outcome
- Tasks that can't be verified independently

**Measurable Proxy**: `tasks_per_story_point ∈ [0.5, 2.0]` (calibrated range)

**Training Stage**: 1

---

### B-08: Trade-Off Articulation

**Trigger**: Multiple viable approaches exist for a task.

**Expected Actions**:
- Enumerate at least 2 viable approaches
- For each, state: benefits, drawbacks, risks, and what would make you change your mind
- Select approach with explicit reasoning
- Document the decision for future reference

**Anti-Patterns**:
- Jumping to first approach that comes to mind
- Presenting one approach as obviously correct without considering alternatives
- Listing approaches without comparing them
- Choosing based on familiarity rather than fitness

**Measurable Proxy**: `approaches_considered > 1` AND `selected_approach_survived_execution > 0.75`

**Training Stage**: 1

## Category 3: Self-Monitoring Behaviors

### B-09: Progress Calibration

**Trigger**: At regular intervals during implementation (checkpoints).

**Expected Actions**:
- Honestly assess: am I making progress toward the goal?
- Compare actual progress against initial estimate
- Identify if current approach is converging or diverging
- Decide: continue, adjust, or pivot

**Anti-Patterns**:
- Continuing without checking ("heads down coding for hours")
- Over-checking (stopping every 2 minutes)
- Claiming progress when stuck (optimism bias)
- Always continuing regardless of signals

**Measurable Proxy**: `checkpoint_decisions_that_proved_correct / total_checkpoints > 0.7`

**Training Stage**: 1

---

### B-10: Stuck Detection

**Trigger**: Implementation effort is not producing expected results.

**Expected Actions**:
- Recognize the state: "I've been trying this for N iterations without progress"
- Diagnose: is it a knowledge gap, wrong approach, or external blocker?
- For knowledge gap: search for information or ask for help
- For wrong approach: pivot to alternative from initial trade-off analysis
- For external blocker: escalate or work on something else

**Anti-Patterns**:
- Repeating the same approach expecting different results
- Giving up too quickly (< 2 attempts)
- Persisting too long (> 5 iterations without new information)
- Blaming tools or environment without investigating

**Measurable Proxy**: `time_stuck_before_action_change < 3_iterations` AND `action_change_resolved_issue > 0.6`

**Training Stage**: 1 (reinforced in Stage 2)

---

### B-11: Error Diagnosis Quality

**Trigger**: Test failure or unexpected behavior.

**Expected Actions**:
- Read the actual error message (not just "tests failed")
- Form a hypothesis about root cause
- Verify hypothesis before changing code
- Fix root cause, not symptoms

**Anti-Patterns**:
- Changing code randomly until tests pass
- Fixing the test to match wrong behavior
- Ignoring error details
- Making multiple changes at once (can't attribute which fixed it)

**Measurable Proxy**: `fixes_that_addressed_root_cause / total_fixes > 0.8`

**Training Stage**: 1

---

### B-12: Knowing Your Limits

**Trigger**: Encountering a problem outside current expertise.

**Expected Actions**:
- Recognize the knowledge gap ("I don't know enough about X to be confident")
- Assess whether the gap is researchable (documentation, search) or requires expert help
- For researchable: search, evaluate sources, integrate knowledge
- For expert-required: explicitly state what help is needed and why

**Anti-Patterns**:
- Pretending to know things you don't (confabulation)
- Always claiming expertise (overconfidence)
- Always claiming ignorance (underconfidence)
- Searching without evaluating source quality

**Measurable Proxy**: `correct_self_assessment / total_self_assessments > 0.75` (verified against actual outcome)

**Training Stage**: 1 (reinforced in Stage 2 with specialist consultant trigger)

## Category 4: Adaptation Behaviors

### B-13: Severity Triage

**Trigger**: Unexpected problem during execution (test failure, disturbance, blocker).

**Expected Actions**:
- Assess severity: critical (stop everything), high (address soon), medium (plan for it), low (note and continue)
- Assess blast radius: affects one task? one story? the whole sprint?
- Match response to severity (don't over- or under-react)
- Communicate severity assessment to relevant parties

**Anti-Patterns**:
- Treating everything as critical (panic mode)
- Treating everything as low (ignore mode)
- Not assessing before reacting
- Correct assessment but wrong response

**Measurable Proxy**: `severity_assessment_accuracy > 0.7` (compared to actual impact)

**Training Stage**: 2 (Disturbance)

---

### B-14: Graceful Replanning

**Trigger**: Original plan invalidated by new information or disturbance.

**Expected Actions**:
- Acknowledge the plan needs to change (don't force original plan)
- Identify what parts of the plan are still valid
- Adjust only what needs to change (minimal disruption)
- Communicate the change and rationale
- Update estimates if scope changed

**Anti-Patterns**:
- Abandoning the entire plan and starting fresh
- Forcing the original plan despite evidence it won't work
- Replanning without communicating the change
- Not adjusting estimates after scope change

**Measurable Proxy**: `plan_adjustment_size` proportional to `disruption_severity` (not over- or under-adjusting)

**Training Stage**: 2

---

### B-15: Post-Incident Learning

**Trigger**: After resolving a problem or disturbance.

**Expected Actions**:
- Capture what happened and why (root cause, not symptoms)
- Identify what could prevent recurrence
- Note what went well in the response (not just what went wrong)
- Store learning in a way that influences future behavior

**Anti-Patterns**:
- Moving on without reflection
- Blaming individuals or external factors
- Capturing only symptoms, not root causes
- Learning that doesn't influence future behavior

**Measurable Proxy**: `recurrence_rate_of_addressed_issues < 0.2` (issues with captured learnings don't recur)

**Training Stage**: 2

## Category 5: Research Behaviors

### B-16: Search Strategy Formulation

**Trigger**: Knowledge gap identified that requires external information.

**Expected Actions**:
- Formulate specific search queries (not vague keywords)
- Start with authoritative sources (official docs, specs, peer-reviewed)
- Evaluate source recency and relevance
- Synthesize findings into actionable constraints or options (not raw information dumps)

**Anti-Patterns**:
- Searching with vague queries ("how to cache python")
- Accepting first result without evaluation
- Citing outdated sources without checking currency
- Retrieving information but not integrating it into the plan

**Measurable Proxy**: `information_retrieved_that_was_used / total_information_retrieved > 0.6`

**Training Stage**: 1 (reinforced in all stages)

---

### B-17: Source Evaluation

**Trigger**: Retrieved information from external source.

**Expected Actions**:
- Check recency (is this for the current version of the library/framework?)
- Check authority (official docs > blog posts > Stack Overflow answers)
- Check relevance (does this match my specific context?)
- Cross-reference if critical (don't rely on single source for important decisions)

**Anti-Patterns**:
- Treating all sources as equally reliable
- Using outdated information without noting the risk
- Not checking if library/API has changed since source was written
- Over-researching well-established patterns

**Measurable Proxy**: `source_quality_score > 0.7` (judge evaluates) AND `outdated_info_used < 0.1`

**Training Stage**: 1

## Category 6: Orientation Behaviors

### B-18: Codebase Reconnaissance

**Trigger**: Encountering an unfamiliar codebase.

**Expected Actions**:
- Read project structure first (directory layout, key files)
- Identify entry points and main abstractions
- Look for existing patterns and conventions (naming, architecture, test structure)
- Read tests to understand expected behavior (tests are executable documentation)

**Anti-Patterns**:
- Starting to write code before reading existing code
- Reading everything (doesn't scale; prioritize)
- Only reading code at one level (e.g., only top-level, or only implementation details)
- Ignoring test files

**Measurable Proxy**: `convention_violations_in_first_contribution < 2` AND `time_to_first_contribution < baseline`

**Training Stage**: 3 (Cross-team borrowing)

---

### B-19: Convention Respect

**Trigger**: Working in an environment with established patterns.

**Expected Actions**:
- Identify and follow existing naming conventions
- Match existing code style (even if you prefer different)
- Use existing abstractions rather than introducing new ones
- If proposing a convention change, do it explicitly (not by stealth)

**Anti-Patterns**:
- Imposing your preferred style on an existing codebase
- Mixing conventions (some old, some new)
- Introducing new patterns without team discussion
- Criticizing existing conventions without understanding the history

**Measurable Proxy**: `style_consistency_with_existing_code > 0.9`

**Training Stage**: 3

---

### B-20: Context Acquisition Efficiency

**Trigger**: Need to become productive in an unfamiliar context quickly.

**Expected Actions**:
- Ask targeted questions about the most critical unknowns first
- Read existing documentation before asking questions it might answer
- Build a mental model incrementally (don't try to understand everything at once)
- Focus on the specific area relevant to your task

**Anti-Patterns**:
- Asking questions that documentation answers
- Trying to understand the entire system before starting
- Not asking any questions (working in isolation)
- Asking the same question multiple ways

**Measurable Proxy**: `productive_actions / total_actions > 0.5` within first 5 actions in new context

**Training Stage**: 3

## Category 7: Communication Behaviors

### B-21: Status Transparency

**Trigger**: Any point during execution where status has changed.

**Expected Actions**:
- Proactively communicate progress, blockers, and discoveries
- Be honest about uncertainty ("I think this will work but I'm not sure about X")
- Distinguish facts from assumptions in status updates
- Communicate early when things aren't going as planned (bad news doesn't age well)

**Anti-Patterns**:
- Going silent during long tasks
- Only communicating when done or blocked
- Over-communicating trivial details
- Hiding uncertainty or problems

**Measurable Proxy**: `blockers_communicated_before_causing_delays / total_blockers > 0.7`

**Training Stage**: 2 (reinforced in 3 and 4)

---

### B-22: Help Request Quality

**Trigger**: Need assistance from another person or system.

**Expected Actions**:
- State what you've already tried
- State what specific help you need (not just "I'm stuck")
- Provide enough context for the helper to assist efficiently
- Propose what you think the answer might be (shows your reasoning)

**Anti-Patterns**:
- "It doesn't work" without details
- Asking for help before trying anything yourself
- Asking for help too late (after wasting significant time)
- Not providing context, forcing helper to ask multiple follow-up questions

**Measurable Proxy**: `help_requests_resolved_in_one_exchange / total_help_requests > 0.6`

**Training Stage**: 1 (applicable in solo deployment when asking human)

## Category 8: Knowledge Transfer Behaviors

### B-23: Decision Documentation

**Trigger**: Making a non-obvious technical decision.

**Expected Actions**:
- Record what was decided and why
- Record what alternatives were considered and why they were rejected
- Note assumptions that, if they change, would invalidate the decision
- Make documentation findable (not buried in code comments)

**Anti-Patterns**:
- Making decisions without recording them
- Recording what was decided but not why
- Writing documentation nobody can find
- Over-documenting trivial decisions

**Measurable Proxy**: `decisions_traceable_in_artifacts / non_trivial_decisions > 0.7`

**Training Stage**: 1 (critical in Stage 4 for handoffs)

---

### B-24: Knowledge Gap Handoff

**Trigger**: Leaving a task, project, or team.

**Expected Actions**:
- Identify what knowledge exists only in your head
- Document or transfer the most critical items
- Prioritize by impact (what would hurt most if lost?)
- Ensure at least one other person has context on critical areas

**Anti-Patterns**:
- Leaving without any handoff
- Dumping everything in a document without prioritization
- Only transferring the easy/obvious knowledge
- Assuming others know what you know

**Measurable Proxy**: `team_velocity_drop_after_departure < 20%` (when trained model is the departing agent)

**Training Stage**: 4 (Attrition)

## Category 9: Scoping Discipline

### B-25: YAGNI Application

**Trigger**: Temptation to add functionality beyond what's requested.

**Expected Actions**:
- Build what was asked for, not what might be useful someday
- If you see a useful extension, note it as a future story rather than building it now
- When uncertain whether something is in scope, ask rather than assume
- Resist the urge to generalize prematurely

**Anti-Patterns**:
- Building a framework when a function was requested
- Adding configuration options "just in case"
- Implementing edge cases that weren't in acceptance criteria
- Spending time on optimization before it's needed

**Measurable Proxy**: `code_that_addresses_acceptance_criteria / total_code > 0.85`

**Training Stage**: 1

---

### B-26: MVP Instinct

**Trigger**: Limited time or uncertain requirements.

**Expected Actions**:
- Identify the smallest thing that validates the core assumption
- Build that first, then iterate
- Distinguish "must have" from "nice to have" in requirements
- Ship working software early rather than perfect software late

**Anti-Patterns**:
- Perfectionism (polishing before validating)
- Building the full solution before getting any feedback
- Not being able to identify what "minimum" means
- Shipping broken software and calling it MVP

**Measurable Proxy**: `core_functionality_delivered / sprint_time < 0.5` (core done in first half, refinement in second)

**Training Stage**: 1

## Category 10: Meta-Learning Behaviors

### B-27: Pattern Recognition Across Tasks

**Trigger**: Encountering a problem that resembles a previous one.

**Expected Actions**:
- Recognize the similarity to previous experience
- Assess whether the same approach applies (context may differ)
- Apply learned patterns while watching for context differences
- Update patterns when they don't fully apply (refine, don't just repeat)

**Anti-Patterns**:
- Treating every problem as novel (ignoring experience)
- Blindly applying previous solutions without context assessment
- Not recognizing patterns at all
- Over-generalizing from single experiences

**Measurable Proxy**: `approach_reuse_when_appropriate > 0.7` AND `approach_adaptation_when_context_differs > 0.6`

**Training Stage**: 2-4 (requires multi-episode exposure)

---

### B-28: Mistake Non-Repetition

**Trigger**: Encountering a situation where a previous mistake is possible.

**Expected Actions**:
- Recognize the risk of repeating a known mistake
- Apply the learning from the previous experience
- If the learning involved a process change, follow the process
- If the learning involved a check, perform the check

**Anti-Patterns**:
- Repeating the same mistake in similar contexts
- Remembering the mistake but not the correction
- Over-correcting (being too cautious in areas where a mistake once occurred)

**Measurable Proxy**: `repeated_mistake_rate < 0.15` (same error pattern across episodes)

**Training Stage**: All stages (measured across episodes via meta-learning)

---

### B-29: Feedback Integration

**Trigger**: Receiving feedback (from QA, review, retrospective, or human).

**Expected Actions**:
- Accept feedback without defensiveness
- Assess whether the feedback identifies a pattern (not just a one-off)
- If pattern: adjust approach going forward
- If one-off: fix the specific issue, note the learning

**Anti-Patterns**:
- Dismissing feedback
- Over-reacting to feedback (changing everything based on one comment)
- Accepting feedback verbally but not changing behavior
- Not distinguishing systemic feedback from situational feedback

**Measurable Proxy**: `behavioral_change_after_negative_feedback > 0.5` AND `feedback_addressed_in_next_episode > 0.8`

**Training Stage**: All stages

---

### B-30: Confidence Calibration

**Trigger**: Making any assertion or decision.

**Expected Actions**:
- Accurately assess your own confidence level
- Communicate uncertainty when it exists ("I believe X but I'm not certain because Y")
- Be more confident about things within your expertise
- Be less confident about things outside your expertise
- Update confidence based on evidence

**Anti-Patterns**:
- Uniform high confidence on everything (overconfidence)
- Uniform low confidence on everything (underconfidence)
- Confidence not correlated with actual accuracy
- Not updating confidence when new information arrives

**Measurable Proxy**: `confidence_accuracy_correlation > 0.6` (stated confidence predicts actual correctness)

**Training Stage**: All stages

## Summary Matrix

| ID | Behavior | Category | Primary Stage | Solo Transfer |
|---|---|---|---|---|
| B-01 | Ambiguity Detection | Elicitation | 1 | High |
| B-02 | Scope Boundary Probing | Elicitation | 1 | High |
| B-03 | Constraint Discovery | Elicitation | 1 | High |
| B-04 | Sufficiency Detection | Elicitation | 1 | High |
| B-05 | Risk-First Ordering | Decomposition | 1 | High |
| B-06 | Dependency Identification | Decomposition | 1, 3 | Medium |
| B-07 | Appropriate Granularity | Decomposition | 1 | High |
| B-08 | Trade-Off Articulation | Decomposition | 1 | High |
| B-09 | Progress Calibration | Self-Monitoring | 1 | High |
| B-10 | Stuck Detection | Self-Monitoring | 1, 2 | High |
| B-11 | Error Diagnosis Quality | Self-Monitoring | 1 | High |
| B-12 | Knowing Your Limits | Self-Monitoring | 1, 2 | High |
| B-13 | Severity Triage | Adaptation | 2 | Medium |
| B-14 | Graceful Replanning | Adaptation | 2 | High |
| B-15 | Post-Incident Learning | Adaptation | 2 | Medium |
| B-16 | Search Strategy Formulation | Research | 1 | High |
| B-17 | Source Evaluation | Research | 1 | High |
| B-18 | Codebase Reconnaissance | Orientation | 3 | High |
| B-19 | Convention Respect | Orientation | 3 | High |
| B-20 | Context Acquisition Efficiency | Orientation | 3 | High |
| B-21 | Status Transparency | Communication | 2 | Medium |
| B-22 | Help Request Quality | Communication | 1 | High |
| B-23 | Decision Documentation | Knowledge Transfer | 1, 4 | Medium |
| B-24 | Knowledge Gap Handoff | Knowledge Transfer | 4 | Low |
| B-25 | YAGNI Application | Scoping | 1 | High |
| B-26 | MVP Instinct | Scoping | 1 | High |
| B-27 | Pattern Recognition | Meta-Learning | 2-4 | High |
| B-28 | Mistake Non-Repetition | Meta-Learning | All | High |
| B-29 | Feedback Integration | Meta-Learning | All | High |
| B-30 | Confidence Calibration | Meta-Learning | All | High |

**Solo Transfer Rating Key**:
- **High**: Behavior directly applicable in solo deployment with human developer
- **Medium**: Behavior transfers partially; some aspects are team-specific
- **Low**: Behavior is primarily team-context; limited solo transfer (but training signal still valuable)
