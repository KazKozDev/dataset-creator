# LLM Dataset Creator

Synthetic dataset generator for fine-tuning LLMs. Supports multi-agent orchestration, quality control, and multiple export formats.

## Who Is This For

- **AI/ML Teams** building custom chatbots and assistants
- **Enterprises** fine-tuning LLMs on domain-specific data (support, legal, medical, sales)
- **Startups** creating training data without manual annotation
- **Researchers** generating synthetic datasets for experiments

## Use Cases

- Customer support chatbots
- Internal knowledge assistants
- Domain-specific Q&A systems
- Sales and onboarding bots
- Educational tutors

## Features

- **Generation**: 12+ domains (support, medical, legal, education, sales, etc.)
- **Multi-Agent Orchestration**: Ensemble Cascade, Swarm Collective Convergence, Evolutionary Agents Fusion, Lattice Network Sync
- **Quality Control**: Toxicity detection, PII filtering, deduplication, diversity analysis
- **Export Formats**: HuggingFace, OpenAI, Alpaca, LangChain
- **Versioning**: Dataset diff, merge, version history
- **Collaboration**: Users, roles, permissions, review workflows

## Tech Stack

- **Backend**: FastAPI, SQLite, Python 3.9+
- **Frontend**: React 18, Chakra UI
- **LLM Providers**: Ollama (local), OpenAI, Anthropic, Google, Mistral

## Quick Start

**Docker:**
```bash
cp .env.example .env
docker-compose up --build
```

**macOS:** Double-click `restart_app.command` — kills old processes, starts backend + frontend, opens browser.

**Manual:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd frontend && npm install

# Run
cd backend && uvicorn main:app --reload --port 8000
cd frontend && npm start
```

Open http://localhost:3000

## Configuration

Set API keys in `.env` (optional — for cloud providers):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

For local inference, install [Ollama](https://ollama.ai) and pull any model. The app auto-detects your installed models — choose what fits your hardware.

## Domains

support, medical, legal, education, business, technical, sales, financial, research, coaching, creative, meetings, ecommerce

## Multi-Agent Modes

| Mode | Agents | Best For |
|------|--------|----------|
| Ensemble Cascade | Scout, Gatherer, Mutator, Selector | Volume + diversity |
| Swarm Collective Convergence | Scout, Mutagen, Crossover | Iterative refinement |
| Evolutionary Agents Fusion | Scout, Exploder, Cooler, Synthesizer | Creative expansion |
| Lattice Network Sync | Gauge, Fermion, Higgs, Yukawa | Quality through selection |

See [AGENTS.md](AGENTS.md) for details.

## Project Structure

```
backend/
  main.py           # FastAPI app
  generator.py      # Dataset generation
  quality.py        # Quality control
  domains.py        # Domain definitions
  llm_providers.py  # LLM integrations
  agents/           # Multi-agent orchestration
  exporters/        # Export formats
  quality_advanced/ # Toxicity, dedup, diversity
  versioning/       # Diff, merge
  collaboration/    # Users, permissions
frontend/
  src/components/   # React components
```

## API

Base URL: `http://localhost:8000`

- `POST /api/generator/start` - Start generation task
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}` - Get dataset
- `POST /api/export/{id}` - Export dataset
- `GET /api/quality/{id}` - Quality report

Full API docs: http://localhost:8000/docs

## License

MIT
