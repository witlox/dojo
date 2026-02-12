# Design Rationale

This document captures the key design decisions and the reasoning behind them.

## Origin

This project emerged from a conversation exploring whether AI coding agents could be elevated from junior/mid-level code generators to senior-level problem solvers. The core observation was that current agents are impressive at code production but weak at the *upstream discipline* that defines senior engineers: knowing what to build before touching code, knowing when to stop and ask, knowing which problems to solve vs. defer, and knowing how to decompose ambiguity.

## Decision 1: Behavior Over Skills

### The Problem

Current coding agents produce competent code but exhibit junior-level problem-solving behavior. They jump straight to implementation without adequate analysis, fail to ask clarifying questions, don't assess risk before committing to an approach, and struggle to adapt when their initial approach fails.

### The Decision

Train meta-cognitive *behavioral patterns* into model weights, while keeping language/tool knowledge external via retrieval. The model learns *how to think about problems*, not *how to write code*.

### The Reasoning

The behaviors that define senior engineers are relatively stable over time — how a senior approaches an unfamiliar problem hasn't fundamentally changed in 20 years, even though the tools have changed completely. Meanwhile language/tool knowledge has a half-life of perhaps 18 months. This separation aligns the update frequency with the volatility of each knowledge type.

Additionally, behavioral patterns are more *compressible* than knowledge. There are perhaps 30-40 core meta-cognitive patterns a senior engineer uses. A small model can plausibly learn to select and apply these patterns even if it can't contain vast technical knowledge.

## Decision 2: RL Over Distillation

### The Problem

The obvious approach to creating a "senior model" is knowledge distillation — run a large model (Claude Opus) exhibiting senior behavior, collect its outputs, and fine-tune a small model to imitate them.

### The Decision

Use reinforcement learning with the agile-agent-team simulation as the environment, rather than supervised distillation from a large model.

### The Reasoning

Distillation produces a lossy compression of the source model. The distilled model approximates the large model's behavior on in-distribution examples but degrades on novel situations — which is precisely where senior judgment matters most. The ceiling is the source model, and you're not creating new capability, only compressing existing capability.

RL with a simulation environment lets the model *discover* effective strategies through trial and feedback. This mirrors how human seniors actually develop judgment — not by imitating other seniors, but by repeatedly experiencing consequences of their decisions. The model learns *which meta-cognitive strategies lead to successful outcomes* rather than *what a large model would say*.

The agile-agent-team simulation is uniquely suited as this environment because it produces structured reasoning traces with measurable outcomes (velocity, test coverage, QA acceptance, disturbance recovery time) — the reward signal needed for RL.

## Decision 3: Team Training, Solo Deployment

### The Problem

The training environment (agile-agent-team) is a multi-agent system. The deployment target is a single model interacting with one human developer. Team-specific behaviors (coordination, delegation, cross-team communication) don't transfer to solo deployment.

### The Decision

Use team dynamics during training as a mechanism for producing richer behavioral signal, but evaluate always against solo-deployment scenarios.

### The Reasoning

Team environments create higher-variance training scenarios that force more robust behavioral patterns. A model trained only on stable, predictable tasks may learn brittle strategies. A model that has been "borrowed" into unfamiliar teams repeatedly has learned a *general* orientation strategy, not one tuned to a specific codebase.

Specific team behaviors that transfer to solo deployment:
- **Elicitation** (asking PO → asking human developer)
- **Self-monitoring** (pairing checkpoints → internal progress evaluation)
- **Risk-first decomposition** (sprint planning → task planning)
- **Knowing your limits** (specialist consultant trigger → knowing when to search/ask for help)
- **Scoping discipline** (PO acceptance feedback → building the right thing)
- **Rapid orientation** (borrowing into unfamiliar team → approaching unfamiliar codebase)

Behaviors that don't transfer (coordination, mentorship, delegation) are still valuable during training because they create scenarios that develop the transferable skills above.

## Decision 4: Curriculum via Increasing Complexity

### The Decision

Structure training in four stages of increasing environmental complexity: stable team → disturbances → cross-team → team composition changes.

### The Reasoning

This mirrors actual career progression — junior engineers master individual contribution, mid-levels handle disruption, seniors handle cross-boundary work, staff/principal engineers handle organizational change. Each stage builds on behavioral patterns from the previous one.

Practically, this also helps with training stability. RL in complex environments is notoriously unstable. Starting with simple, predictable episodes and gradually increasing complexity provides curriculum learning that helps the model develop foundational patterns before encountering high-variance scenarios.

## Decision 5: Phase-Level Training Episodes

### The Problem

Full sprint episodes (60 minutes, ~244 LLM calls) are too expensive for RL training, which typically requires thousands of episodes.

### The Decision

Decompose training into phase-level episodes (elicitation, decomposition, research, execution, adaptation) rather than full sprints. Use full sprints only for periodic evaluation.

### The Reasoning

The agile-agent-team's ceremony structure (refinement → technical planning → daily standup → pairing → review) provides natural phase boundaries. Each phase has well-defined inputs, outputs, and measurable outcomes, making them tractable as RL episodes.

Phase-level training also allows focused development of specific behavioral patterns. The elicitation behavior can be trained in hundreds of short episodes before the model ever attempts a full sprint, ensuring foundational patterns are solid before composition.

## Decision 6: Behavioral Evaluator Using Large Model as Judge

### The Problem

Behavioral reward is noisier than code-level reward. "Tests pass" is binary. "Asked good questions during refinement" is subjective and context-dependent.

### The Decision

Use a large model (Claude Opus) as a behavioral evaluator — not as the training target, but as the judge of behavioral quality.

### The Reasoning

This follows the RLAIF (RL from AI Feedback) pattern but applied to engineering meta-cognition rather than safety. The large model evaluates reasoning traces against a behavioral rubric (the taxonomy defined in `specs/BEHAVIORAL_TAXONOMY.md`).

The agile-agent-team environment provides a calibration mechanism: the behavioral evaluation (from the judge model) and the outcome evaluation (from sprint metrics) serve as two independent channels. A behavior that the judge rates highly but leads to poor sprint outcomes indicates the rubric needs updating. The two channels cross-validate.

## Decision 7: Separate Project, External Pipeline

### The Decision

Build this as a separate project that wraps around agile-agent-team, rather than extending agile-agent-team directly.

### The Reasoning

The agile-agent-team is a research simulation for studying team dynamics. This project has a fundamentally different purpose — training a model. The concerns are different (episode structure, reward functions, training loops vs. sprint ceremonies, team culture, organizational dynamics).

Integration points are well-defined: this project uses agile-agent-team as a callable environment, injects candidate models into agent slots, runs episodes, and extracts outcome data. The agile-agent-team system's existing model-swapping capability (different models per seniority level) already supports this.

## Open Questions

### Can behavioral patterns transfer across base models?

If we fine-tune a 14B model and a better base model releases, does the behavioral adapter transfer? Unknown. This matters for long-term viability.

### Risk of "performative seniority"

A model that learns to *perform* senior behaviors (asking questions because it's rewarded) rather than *using* them (asking because it needs answers). The reward function must penalize unnecessary elicitation as well as insufficient elicitation.

### Non-stationarity of "good behavior"

What counts as senior behavior evolves. The behavioral taxonomy will need versioning and periodic updating.

### Sufficient episode diversity

The quality of training depends on the diversity of backlogs and scenarios in the simulation. Narrow task distributions will produce narrow behavioral patterns.
