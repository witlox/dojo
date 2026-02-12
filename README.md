# Dojo: Senior Meta-Cognitive Model Trainer

## Vision

Train a local LLM to exhibit senior-engineer **behavioral patterns** — not coding skills (models are already good at those), but the meta-cognitive layer that distinguishes seniors from juniors: knowing what to ask, when to research, how to decompose ambiguity, when to stop and rethink, and how to adapt when things go wrong.

## Core Insight

Senior engineers aren't defined by their ability to write code. They're defined by **what they do before writing code** and **how they respond when things don't go as planned**. Current coding agents skip straight to implementation. This project trains the upstream discipline.

## Architecture

The system has three major components:

1. **Training Environment** — The [agile-agent-team](https://github.com/witlox/agile-agent-team) multi-agent simulation, adapted to serve as a reinforcement learning gym (eg. **Dojo**). It produces structured behavioral traces with measurable outcomes across diverse scenarios (single-team, multi-team, disturbances, agent borrowing, attrition/onboarding).

2. **Reward Attribution Pipeline** — Consumes sprint artifacts and produces per-decision reward vectors. Connects reasoning traces (questions asked, research performed, decomposition choices) to downstream outcomes (tests passing, QA acceptance, velocity, recovery time).

3. **Training Loop** — RL-based training that updates model weights based on reward signals. The model learns *which meta-cognitive strategies lead to successful outcomes* through repeated exposure to consequence — the same mechanism by which human seniors develop judgment.

## Key Design Decisions

- **Behavior, not skills**: The fine-tuned weights encode meta-cognitive patterns (elicitation, decomposition, self-monitoring, adaptation). Language/tool knowledge comes from external retrieval at inference time.
- **Team training, solo deployment**: The multi-agent environment is the gym. The product is a single model that interacts with one human developer — but whose behavioral patterns were forged in complex team dynamics.
- **RL over distillation**: We're not copying a large model's behavior. We're letting a small model discover effective strategies through trial and feedback, producing genuinely learned judgment rather than compressed imitation.

## Separation of Concerns

| Layer | Source | Update Frequency |
|-------|--------|-----------------|
| Meta-cognitive patterns | Fine-tuned into weights | Retrained periodically |
| Problem-solving strategies | Fine-tuned into weights | Retrained periodically |
| Self-monitoring / adaptation | Fine-tuned into weights | Retrained periodically |
| Language syntax, APIs, libraries | External retrieval (RAG) | Always current |
| Tool documentation, best practices | External retrieval (RAG) | Always current |
| Domain-specific patterns | External retrieval (RAG) | Always current |

## Training Curriculum

| Stage | Environment | Behaviors Trained | Prerequisites |
|-------|-------------|-------------------|---------------|
| 1. Stable team | Single team, no disruptions | Elicitation, decomposition, self-monitoring, checkpoint dialogue | None |
| 2. Disturbances | Single team + chaos injection | Triage, replanning, communication under pressure, recovery | Stage 1 |
| 3. Cross-team | Multi-team + agent borrowing | Rapid orientation, context-switching, convention respect | Stage 2 |
| 4. Team change | Attrition + new hire scenarios | Knowledge transfer, onboarding, compensating for knowledge loss | Stage 3 |

## Project Structure

```
senior-model-trainer/
├── README.md                           # This file
├── docs/
│   ├── DESIGN_RATIONALE.md             # Why this approach, key decisions
│   ├── ARCHITECTURE.md                 # System architecture and components
│   ├── TRAINING_PIPELINE.md            # End-to-end training pipeline
│   ├── REWARD_FUNCTION.md              # Reward attribution design
│   ├── EVALUATION_HARNESS.md           # Solo-deployment evaluation
│   ├── INTEGRATION_WITH_AAT.md         # How this connects to agile-agent-team
│   └── DIALOGUE_REFERENCE.md           # Original design conversation
└── specs/
    ├── BEHAVIORAL_TAXONOMY.md          # 30 behavioral pattern specifications
    ├── TRAINING_EPISODES.md            # Episode structure per curriculum stage
    ├── REWARD_SIGNALS.md               # Measurable reward signals per behavior
    └── TRANSFER_ANALYSIS.md            # Which team behaviors transfer to solo
```

## Relationship to agile-agent-team

This is a **separate project** that uses agile-agent-team as one component (the simulation environment). Integration points are documented in `docs/INTEGRATION_WITH_AAT.md`. The agile-agent-team system is not modified — this project wraps around it.

## Status

**Phase: Design & Specification**

This repository contains the design documents, behavioral specifications, and architectural plans. Implementation has not yet started.

## License

MIT
