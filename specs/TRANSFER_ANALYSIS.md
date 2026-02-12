# Transfer Analysis

## Purpose

This document analyzes which behavioral patterns trained in the multi-agent team environment transfer to solo deployment (the product use case: one model, one human developer, one task).

## Transfer Framework

A behavior transfers well to solo deployment when:
1. The core cognitive pattern is the same (even if the social context differs)
2. The trigger conditions exist in solo context
3. The action space in solo context supports the behavior
4. The reward signal (what makes it effective) is similar

## Full Transfer Analysis

### High Transfer (directly applicable solo)

| Behavior | Team Context | Solo Context | Why It Transfers |
|---|---|---|---|
| B-01: Ambiguity Detection | Ask PO/team | Ask human developer | Same skill: recognizing gaps in specifications |
| B-02: Scope Boundary Probing | Clarify with PO | Clarify with developer | Same skill: preventing scope mismatch |
| B-03: Constraint Discovery | Discuss in planning | Ask developer upfront | Same skill: uncovering hidden requirements |
| B-04: Sufficiency Detection | Know when to stop asking team | Know when to stop asking developer | Same skill: calibrating information needs |
| B-05: Risk-First Ordering | Plan sprint tasks | Plan implementation steps | Same skill: tackling uncertainty first |
| B-07: Appropriate Granularity | Break stories into tasks | Break task into steps | Same skill: right-sizing work units |
| B-08: Trade-Off Articulation | Design dialogue with pair | Explain options to developer | Same skill: structured decision-making |
| B-09: Progress Calibration | Pairing checkpoints | Self-checkpoints during execution | Same skill: honest self-assessment |
| B-10: Stuck Detection | Recognize during pairing | Recognize during solo execution | Same skill: knowing when to change approach |
| B-11: Error Diagnosis Quality | Debug in pair | Debug solo | Same skill: systematic error analysis |
| B-12: Knowing Your Limits | Trigger specialist consultant | Tell developer "I'm not confident about X" | Same skill: accurate self-assessment |
| B-14: Graceful Replanning | Adjust sprint plan | Adjust implementation approach | Same skill: adapting to new information |
| B-16: Search Strategy | Research during sprint | Research during solo work | Same skill: effective information retrieval |
| B-17: Source Evaluation | Evaluate sources for team | Evaluate sources for implementation | Same skill: critical source assessment |
| B-18: Codebase Reconnaissance | Orient in borrowed team's code | Orient in developer's codebase | Same skill: rapid codebase understanding |
| B-19: Convention Respect | Follow receiving team's style | Follow existing codebase style | Same skill: style adaptation |
| B-22: Help Request Quality | Ask team for help | Ask developer for help | Same skill: structured help requests |
| B-25: YAGNI Application | Resist scope creep in sprint | Resist over-engineering in task | Same skill: building only what's needed |
| B-26: MVP Instinct | Deliver working increment | Deliver working solution first | Same skill: prioritizing completeness over perfection |
| B-27: Pattern Recognition | Across sprints | Across tasks/sessions | Same skill: applying past experience |
| B-28: Mistake Non-Repetition | Across sprints (meta-learning) | Across tasks/sessions | Same skill: learning from errors |
| B-29: Feedback Integration | From retros and reviews | From developer feedback | Same skill: incorporating feedback |
| B-30: Confidence Calibration | In all team contexts | In all solo contexts | Same skill: accurate self-assessment |

### Medium Transfer (partially applicable, context adaptation needed)

| Behavior | Team Context | Solo Context | What Transfers | What Doesn't |
|---|---|---|---|---|
| B-06: Dependency ID | Cross-task, cross-team deps | Dependencies within own implementation | Recognizing integration points | Multi-team coordination |
| B-13: Severity Triage | Team-wide impact assessment | Impact on current task/codebase | Severity assessment skill | Blast radius across team |
| B-15: Post-Incident Learning | Team retro, meta-learning | Self-reflection, notes for developer | Root cause analysis | Team process changes |
| B-20: Context Acquisition | Asking teammates efficiently | Reading code and docs efficiently | Information prioritization | Social information gathering |
| B-21: Status Transparency | Update team and stakeholders | Update developer on progress | Proactive communication | Multi-stakeholder management |
| B-23: Decision Documentation | Team-accessible records | Code comments, developer-facing notes | Recording rationale | Team discovery/access patterns |

### Low Transfer (primarily team-context, but valuable training signal)

| Behavior | Why Low Transfer | What Training Value Remains |
|---|---|---|
| B-24: Knowledge Gap Handoff | No one to hand off to solo | Develops B-23 (documentation quality) as a side effect |

## Solo Deployment Behavioral Profile

Based on the transfer analysis, the deployed model should exhibit this behavioral profile:

### Phase 1: Receiving a Task

1. **Read the task description carefully** (B-01)
2. **Identify what's missing** — scope boundaries, constraints, success criteria (B-01, B-02, B-03)
3. **Ask specific, prioritized questions** to the human developer (B-01, B-22)
4. **State assumptions explicitly** (B-04)
5. **Signal readiness to proceed** with rationale (B-04)

### Phase 2: Research and Planning

6. **Search for relevant information** using specific queries (B-16)
7. **Evaluate source quality** — recency, authority, relevance (B-17)
8. **Decompose the task** into tractable steps (B-05, B-07)
9. **Identify the riskiest assumption** and plan to validate it first (B-05)
10. **Consider alternative approaches** and select with reasoning (B-08)

### Phase 3: Execution

11. **Orient in the codebase** — read before writing, follow conventions (B-18, B-19)
12. **Build the minimum viable solution first** (B-26, B-25)
13. **Self-checkpoint at regular intervals** — am I on track? (B-09)
14. **When tests fail, diagnose systematically** — read errors, form hypothesis, verify (B-11)
15. **Recognize when stuck** and change approach (B-10)
16. **Know when to ask for help** vs. research vs. try harder (B-12)

### Phase 4: Adaptation

17. **When something unexpected happens, triage** before reacting (B-13)
18. **Adjust plan gracefully** — preserve what works, change what doesn't (B-14)
19. **Communicate status proactively** — especially problems (B-21)
20. **Capture what you learned** for future reference (B-15, B-23)

### Across Tasks

21. **Apply patterns from previous tasks** when appropriate (B-27)
22. **Don't repeat previous mistakes** (B-28)
23. **Integrate feedback** from the developer (B-29)
24. **Calibrate confidence** — be honest about uncertainty (B-30)

## Implications for Training

### Focus Training Compute on High-Transfer Behaviors

The 24 high-transfer behaviors should receive the majority of training episodes. The 6 medium-transfer behaviors should receive moderate episodes. B-24 (low transfer) is trained only as a side effect of Stage 4 episodes.

### Evaluation Must Test Solo Context

Even though training happens in team context, evaluation must always use the solo protocol (see `docs/EVALUATION_HARNESS.md`). The transfer score per behavior tracks whether team training improves solo performance.

### Stage Value for Solo Deployment

| Stage | Primary Solo Contribution |
|---|---|
| 1 (Stable) | Core behavioral foundation — elicitation, decomposition, self-monitoring, research |
| 2 (Disturbance) | Adaptation and resilience — triage, replanning, recovery |
| 3 (Cross-team) | Orientation and context-switching — codebase navigation, convention respect |
| 4 (Team change) | Documentation quality, knowledge management (indirect) |

**Expected diminishing returns**: Stage 1 contributes most to solo performance. Stage 4 contributes least directly. But Stage 3 (borrowing/orientation) may be disproportionately valuable for the common solo scenario of "here's an unfamiliar codebase, add a feature."

### Ablation Study Design

To measure the incremental value of each stage for solo deployment:

1. Train Stage 1 only → evaluate solo → baseline transfer score
2. Train Stage 1+2 → evaluate solo → measure improvement from disturbance training
3. Train Stage 1+2+3 → evaluate solo → measure improvement from orientation training
4. Train Stage 1+2+3+4 → evaluate solo → measure improvement from team change training

If Stage N doesn't improve solo scores, it can be dropped from the curriculum to save compute.
