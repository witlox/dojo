# Dojo: Senior Meta-Cognitive Model Trainer

## Vision

Train a local LLM to exhibit senior-engineer **behavioral patterns** — not coding skills (models are already good at those), but the meta-cognitive layer that distinguishes seniors from juniors: knowing what to ask, when to research, how to decompose ambiguity, when to stop and rethink, and how to adapt when things go wrong.

## Core Insight

Senior engineers aren't defined by their ability to write code. They're defined by **what they do before writing code** and **how they respond when things don't go as planned**. Current coding agents skip straight to implementation. This project trains the upstream discipline.

## Architecture

The system has four major components:

1. **Simulation Environment** (`dojo/env/`) — The [agile-agent-team](https://github.com/witlox/agile-agent-team) multi-agent simulation, used as a reinforcement learning gym via its public `src.rl` API. AAT exposes an `EpisodeRunner`, `PhaseRunner`, `ScenarioCatalog`, `ObservationExtractor`, `ActionExecutor`, and `CheckpointManager` — providing episode-level and phase-level control over sprints, behavioral trace extraction, and mid-episode state serialization. Dojo wraps these as a `gym.Env` with `Discrete(6)` action space and structured `Dict` observation space.

2. **Reward Attribution Pipeline** (`dojo/reward/`) — Consumes sprint artifacts and produces per-decision reward vectors. Uses AAT's `RewardCalculator` and `BehavioralScorer` for outcome-based and heuristic behavioral signals, supplemented by a large-model judge evaluator (Claude Opus) for nuanced behavioral quality assessment. Includes a calibration monitor that detects reward signal drift.

3. **Training Loop** (`dojo/training/`) — RL-based training (PPO with LoRA/QLoRA adapters via HuggingFace TRL) that updates model weights based on composite reward signals. Default base model: `deepseek-coder-v2-lite-instruct`. The model learns *which meta-cognitive strategies lead to successful outcomes* through repeated exposure to consequence — the same mechanism by which human seniors develop judgment.

4. **Evaluation Harness** (`dojo/eval/`) — Tests whether behavioral patterns learned in the team environment transfer to solo deployment. Uses a 5-phase protocol (elicitation → research → planning → execution → verification) with transfer scoring per behavioral pattern. Three evaluation cadences: quick (every 100 episodes), full (every 500), comprehensive (at stage completion).

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

## Setup

```bash
git clone https://github.com/witlox/dojo
cd dojo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install agile-agent-team for src.rl imports
pip install -e /path/to/agile-agent-team
```

## Relationship to agile-agent-team

This is a **separate project** that uses agile-agent-team as one component (the simulation environment). AAT now provides a public `src.rl` package with 28 exported symbols specifically designed for this integration — including `EpisodeRunner`, `PhaseRunner`, `ScenarioCatalog`, `ActionExecutor`, `BehavioralScorer`, `CheckpointManager`, and `RewardCalculator`. Integration points are documented in `docs/INTEGRATION_WITH_AAT.md`.

## License

MIT
