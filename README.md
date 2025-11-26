# LLM Dataset Creator

Synthetic dataset generator for fine-tuning LLMs. Supports multi-agent orchestration, quality control, and multiple export formats.

## Who Is This For

- **AI/ML Teams** building chatbots and assistants
- **Enterprises** creating training data for internal tools
- **Startups** generating datasets without manual annotation
- **Researchers** running synthetic data experiments

## Features

- **12+ Domains**: support, medical, legal, education, business, technical, sales, financial, research, coaching, creative, meetings, ecommerce
- **Multi-Agent Orchestration**: 4 generation methods for different use cases
- **Quality Control**: Toxicity detection, PII filtering, deduplication, diversity analysis
- **Export Formats**: HuggingFace, OpenAI, Alpaca, LangChain
- **Versioning**: Dataset diff, merge, version history

## Quick Start

**Docker:**
```bash
cp .env.example .env
docker-compose up --build
```

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

**macOS:** Double-click `restart_app.command` — kills old processes, starts backend + frontend, opens browser.

## Configuration

Set API keys in `.env` (optional — for cloud providers):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

For local inference, install [Ollama](https://ollama.ai) and pull any model. The app auto-detects your installed models — choose what fits your hardware.

## Multi-Agent Modes

| Mode | Agents | Best For |
|------|--------|----------|
| Ensemble Cascade | Scout, Gatherer, Mutator, Selector | Volume + diversity |
| Swarm Collective Convergence | Scout, Mutagen, Crossover | Iterative refinement |
| Evolutionary Agents Fusion | Scout, Exploder, Cooler, Synthesizer | Creative expansion |
| Lattice Network Sync | Gauge, Fermion, Higgs, Yukawa | Quality through selection |

See [AGENTS.md](AGENTS.md) for details.

## API

Base URL: `http://localhost:8000`

- `POST /api/generator/start` - Start generation task
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}` - Get dataset
- `POST /api/export/{id}` - Export dataset
- `GET /api/quality/{id}` - Quality report

Full API docs: http://localhost:8000/docs

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
