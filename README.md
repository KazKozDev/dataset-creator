
<div align="center">
  <img width="330" height="" alt="sdf" src="https://github.com/user-attachments/assets/97944bef-5254-423c-9aa7-10909b49eb69" />
  <p>
   <br> <h3>Quality datasets. In just a few clicks.</h3><br>
   
  </p>
</div>

Synthetic dataset generator for fine-tuning LLMs. Supports multi-agent orchestration, quality control, and multiple export formats.

This tool is designed for AI/ML teams building chatbots and assistants, enterprises creating training data for internal tools, startups generating datasets without manual annotation, and researchers running synthetic data experiments. Whether you're scaling conversational AI, accelerating model development, or exploring new approaches to data generation, this framework provides the infrastructure to produce high-quality synthetic datasets efficiently.

## Features

- **12+ Domains**: support, medical, legal, education, business, technical, sales, financial, research, coaching, creative, meetings, ecommerce
- **Multi-Agent Orchestration**: 4 generation methods for different use cases
- **Quality Control**: Toxicity detection, PII filtering, deduplication, diversity analysis
- **Export Formats**: HuggingFace, OpenAI, Alpaca, LangChain
- **Versioning**: Dataset diff, merge, version history

<img width="1379" height="947" alt="Screenshot 2025-11-26 at 20 06 22" src="https://github.com/user-attachments/assets/6bf845d5-b3c0-45af-a119-f1abdfc8b390" />

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

Multi-agent orchestration system for generating high-quality synthetic datasets.

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
