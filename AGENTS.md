# Multi-Agent Dataset Generation

This document describes the multi-agent orchestration system for generating high-quality synthetic datasets. Each "mix" uses multiple LLM agents working together, where each agent can be configured with a different LLM provider and model.

## Overview

The system supports 4 orchestration methods ("mixes"), each inspired by different concepts:

| Mix | Name | Inspiration | Agents | Best For |
|-----|------|-------------|--------|----------|
| **Swarm** | Ensemble Cascade | Bee/Ant colonies | 4 | General purpose, diverse examples |
| **Evolution** | Swarm Collective Convergence | Genetic algorithms | 3 | Iterative refinement |
| **Cosmic** | Evolutionary Agents Fusion | Big Bang expansion | 4 | Creative variations from seeds |
| **Quantum** | Lattice Network Sync | Quantum physics | 5 | High-quality through selection |

---

## Ensemble Cascade (Swarm)

*Inspired by bee and ant colony behavior - agents compete and collaborate.*

### Agents

| Agent | Role | LLM Task |
|-------|------|----------|
| **Scout** | Explorer | Generates initial examples using domain-specific prompts |
| **Gatherer** | Critic | Evaluates quality, identifies issues and strengths |
| **Mutator** | Variator | Creates variations (rephrase, formalize, simplify, extend) |
| **Selector** | Curator | Checks diversity, removes duplicates, selects best |

### Flow

```
Scout → Generate examples
    ↓
Gatherer → Evaluate quality (keep score ≥ threshold)
    ↓
Mutator → Create variations of good examples
    ↓
Selector → Diversity check, select top N
    ↓
Final Dataset
```

### Use Case
Best for generating diverse datasets where you need many variations and quality filtering.

---

## Swarm Collective Convergence (Evolution)

*Inspired by genetic algorithms - survival of the fittest examples.*

### Agents

| Agent | Role | LLM Task |
|-------|------|----------|
| **Scout** | Seed generator | Creates initial population |
| **Mutagen** | Mutator | Applies mutations (complexity, edge cases, perspective) |
| **Crossover** | Combiner | Combines best traits from two parent examples |

### Flow

```
Scout → Generate initial population
    ↓
┌─────────────────────────────┐
│  Evolution Loop (N generations)  │
│                             │
│  Mutagen → Apply mutations  │
│      ↓                      │
│  Crossover → Combine pairs  │
│      ↓                      │
│  Evaluate fitness           │
│      ↓                      │
│  Select elite survivors     │
└─────────────────────────────┘
    ↓
Final Dataset (fittest examples)
```

### Use Case
Best when you want examples to improve over iterations, combining best traits.

---

## Evolutionary Agents Fusion (Cosmic)

*Inspired by Big Bang - rapid expansion from seed examples.*

### Agents

| Agent | Role | LLM Task |
|-------|------|----------|
| **Scout** | Seed creator | Generates initial seed example |
| **Exploder** | Expander | Creates many variations in different directions |
| **Cooler** | Stabilizer | Refines and stabilizes "hot" variations |
| **Synthesizer** | Polisher | Final quality polish on stable examples |

### Flow

```
Scout → Generate seed example
    ↓
Exploder → Expand into N variations
           (formal, casual, technical, simple, emotional, etc.)
    ↓
Cooler → Refine unstable variations
    ↓
Synthesizer → Final polish
    ↓
Final Dataset
```

### Variation Directions
- More formal tone
- More casual tone
- Add technical details
- Simplify for beginners
- Add emotional context
- Make more concise
- Expand with examples
- Change scenario context

### Use Case
Best for creative expansion - start with one good example and explode into many variations.

---

## Lattice Network Sync (Quantum)

*Inspired by quantum physics - superposition and wave function collapse.*

### Concept

In quantum physics, a particle exists in **superposition** (all possible states simultaneously) until **measurement** causes collapse into a single state. We apply this to dataset generation:

1. **Superposition**: Generate multiple approaches to the same topic
2. **Measurement**: Evaluate and compare all variants
3. **Collapse**: Select the best variant
4. **Interaction**: Optionally combine best parts from other variants

### Agents

| Agent | Role | LLM Task |
|-------|------|----------|
| **Gauge** | Field generator | Creates N different approaches per example (superposition) |
| **Fermion** | Materializer | Generates Q&A pairs for each approach |
| **Higgs** | Selector | Evaluates variants, selects best (collapse) |
| **Yukawa** | Enhancer | Combines improvements from other variants |
| **Potential** | Formatter | Final output formatting (no LLM) |

### Flow

```
GAUGE: Superposition
├── Approach 1: Technical angle, expert user
├── Approach 2: Practical angle, beginner user  
└── Approach 3: Conceptual angle, curious user
    ↓
FERMION: Materialization
├── Q&A for Approach 1
├── Q&A for Approach 2
└── Q&A for Approach 3
    ↓
HIGGS: Measurement (Collapse)
└── Compare all 3, select best → Approach 2 wins
    ↓
YUKAWA: Interaction
└── Apply suggested improvements using parts from other variants
    ↓
POTENTIAL: Final Collapse
└── Format as standard chat example
```

### Cost Optimization

Each step uses **batch prompts** - one LLM call handles all variants for an example:

| Step | LLM Calls per Example |
|------|----------------------|
| Gauge | 1 (generates 3 approaches) |
| Fermion | 1 (generates 3 Q&A pairs) |
| Higgs | 1 (compares 3 variants) |
| Yukawa | 0-1 (only if improvements needed) |
| Potential | 0 (formatting only) |
| **Total** | **~4 calls per example** |

### Use Case
Best for high-quality examples where you want the system to explore multiple approaches and select the best one.

---

## Multi-Model Configuration

Each agent can use a **different LLM provider and model**. This enables:

- **Diverse perspectives**: Different models have different strengths
- **Cost optimization**: Use cheaper models for simple tasks
- **Quality maximization**: Use best models for critical decisions

### Example Configuration

```
Swarm Mix:
├── Scout: OpenAI GPT-4 (creative generation)
├── Gatherer: Anthropic Claude (critical evaluation)
├── Mutator: OpenAI GPT-4o-mini (fast variations)
└── Selector: Anthropic Claude (quality judgment)

Quantum Mix:
├── Gauge: OpenAI GPT-4 (diverse approaches)
├── Fermion: Ollama Llama3 (Q&A generation)
├── Higgs: Anthropic Claude (best at comparison)
├── Yukawa: OpenAI GPT-4o-mini (quick improvements)
└── Potential: (no LLM needed)
```

### UI Configuration

In the Generator UI:
1. Select "Advanced" generation mode
2. Choose your mix (Swarm, Evolution, Cosmic, Quantum)
3. For each agent role, select:
   - Provider (Ollama, OpenAI, Anthropic)
   - Model (depends on provider)

---

## LLM Budget Management

Each generation task has a budget to prevent runaway costs:

```
Budget = requested_count × agents × multiplier
```

- **Swarm**: `count × 4 agents × 2 = 8× budget`
- **Evolution**: `count × 3 agents × 1.5 = 4.5× budget`
- **Cosmic**: `count × 4 agents × 2 = 8× budget`
- **Quantum**: `count × 4 agents × 1.5 = 6× budget`

When budget is exhausted, agents gracefully degrade (use cached results or skip optional steps).

---

## Comparison

| Feature | Swarm | Evolution | Cosmic | Quantum |
|---------|-------|-----------|--------|---------|
| LLM calls/example | ~4-6 | ~3-4 | ~4-5 | ~4 |
| Diversity | High | Medium | Very High | Medium |
| Quality control | Critic-based | Fitness-based | Refinement | Selection-based |
| Best for | Volume | Iteration | Creativity | Quality |
| Unique strength | Mutations | Crossover | Expansion | Superposition |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Generator Component                                 │   │
│  │  - Select domain/subdomain                          │   │
│  │  - Choose mix (Swarm/Evolution/Cosmic/Quantum)      │   │
│  │  - Configure LLM per agent role                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  /api/generator/start                               │   │
│  │  - Receives agent_models config                     │   │
│  │  - Creates orchestrator with role_models            │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Orchestrator (Swarm/Evolution/Cosmic/Quantum)      │   │
│  │  - Manages agent lifecycle                          │   │
│  │  - Passes role-specific LLM config to each agent    │   │
│  │  - Coordinates data flow between agents             │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Agents (Scout, Gatherer, Mutator, etc.)            │   │
│  │  - Each has own LLM provider/model                  │   │
│  │  - Executes specific role                           │   │
│  │  - Emits events for monitoring                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LLM Providers (Ollama, OpenAI, Anthropic)          │   │
│  │  - Unified interface                                │   │
│  │  - Retry logic                                      │   │
│  │  - Budget tracking                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `backend/agents/base_agent.py` | Base agent class, event bus, budget management |
| `backend/agents/swarm.py` | Swarm orchestrator and agents |
| `backend/agents/evolution.py` | Evolution orchestrator and agents |
| `backend/agents/cosmic.py` | Cosmic orchestrator and agents |
| `backend/agents/quantum.py` | Quantum orchestrator and agents |
| `backend/main.py` | API endpoint, role_models handling |
| `frontend/src/components/Generator.js` | UI for agent model configuration |
| `frontend/src/components/AgentMonitor.js` | Real-time agent activity display |
