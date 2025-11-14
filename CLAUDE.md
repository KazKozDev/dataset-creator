# CLAUDE.md - AI Assistant Guide for Dataset Creator

## Project Overview

**Dataset Creator (SynthGen)** is a synthetic data generation platform for creating and managing training datasets for LLM fine-tuning. It leverages foundation models (Ollama, OpenAI, Anthropic) to generate domain-specific examples through a web interface.

**Version**: 1.1.1
**License**: MIT
**Tech Stack**: Python/FastAPI backend + React/Chakra UI frontend + PostgreSQL/SQLite database

---

## Architecture

### High-Level Structure

```
┌─────────────────┐
│  React Frontend │ ← Chakra UI, React Query, Axios
│   (Port 3000)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Backend│ ← Python, Async, Pydantic
│   (Port 8000)   │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬────────────┐
    ▼         ▼          ▼            ▼
┌────────┐ ┌─────┐  ┌────────┐  ┌──────────┐
│SQLite  │ │JSONL│  │LLM     │  │Background│
│Database│ │Files│  │Providers│ │Tasks     │
└────────┘ └─────┘  └────────┘  └──────────┘
```

### Key Design Principles

1. **Async-First**: All I/O operations use async/await
2. **Modular**: Separation of concerns (generator, quality, database, providers)
3. **Background Jobs**: Long-running tasks don't block API responses
4. **Type Safety**: Pydantic models for validation
5. **Provider Agnostic**: Unified interface for multiple LLM providers

---

## Directory Structure

```
dataset-creator/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # API routes & application entry point
│   ├── generator.py           # Dataset generation logic
│   ├── database.py            # SQLite database operations
│   ├── domains.py             # 12 domain configurations
│   ├── quality.py             # Quality control service
│   ├── llm_providers.py       # LLM provider integrations (Ollama/OpenAI/Anthropic)
│   ├── utils.py               # Utilities (JSONL I/O, format conversion, monitoring)
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Backend container
│   └── docker-compose.yml     # Backend-only compose
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.js            # Main app component & routing
│   │   ├── theme.js          # Chakra UI theme configuration
│   │   ├── components/       # UI components
│   │   │   ├── Dashboard.js  # Metrics overview
│   │   │   ├── Generator.js  # Dataset generation wizard
│   │   │   ├── QualityControl.js  # Quality checking UI
│   │   │   ├── Datasets.js   # Dataset list & management
│   │   │   ├── DatasetDetail.js  # Individual dataset view/edit
│   │   │   ├── Tasks.js      # Job monitoring
│   │   │   ├── Settings.js   # System configuration
│   │   │   └── common/       # Reusable components
│   │   │       ├── Sidebar.js
│   │   │       ├── DomainCard.js
│   │   │       └── ExampleViewer.js
│   │   └── services/
│   │       └── api.js        # Axios API client
│   ├── package.json          # Node dependencies
│   └── Dockerfile            # Frontend container (Node → Nginx)
├── docker/
│   └── docker-compose.yml    # Full stack orchestration
├── data/
│   └── datasets/             # Generated JSONL files
├── requirements.txt          # Root Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Key Backend Modules

### 1. main.py - API Entry Point

**Purpose**: FastAPI application with all API routes

**Key Features**:
- 60+ REST endpoints under `/api/`
- CORS middleware for cross-origin requests
- Auto-import existing datasets on startup
- Background task management
- Lifespan context for initialization/cleanup

**Route Groups**:
```python
/api/domains              # Domain & subdomain management
/api/providers            # LLM provider configuration
/api/generator            # Dataset generation
/api/quality              # Quality control
/api/datasets             # Dataset CRUD operations
/api/tasks                # Task management
/api/settings             # System settings
/api/utils                # Utility endpoints (merge, sample, convert)
```

**Important Patterns**:
- All routes return JSON with consistent error handling
- Long operations use `background_tasks.add_task()`
- Status tracking via database job tables
- File paths in `backend/data/datasets/`

### 2. generator.py - Dataset Generation

**Class**: `DatasetGenerator`

**Purpose**: Generate synthetic training data using LLMs

**Key Methods**:
```python
generate_scenario_parameters(domain, subdomain, num_examples)
  → Creates weighted random scenarios based on domain config

get_prompt_in_language(domain, subdomain, scenario_params, language)
  → Builds LLM prompts with domain-specific context

generate_example(prompt, attempt=1, max_attempts=3)
  → Single example generation with retry logic

generate_examples_batch(dataset_id, num_examples, domain, subdomain, format_type, language)
  → Async batch generation with progress tracking
```

**Generation Flow**:
1. Create generation job in database
2. Load domain configuration from `domains.py`
3. For each example:
   - Generate scenario parameters (weighted random)
   - Build prompt with domain context
   - Call LLM provider
   - Parse JSON response
   - Validate format (chat vs instruction)
   - Save to database & JSONL file
4. Update job status and progress

**Formats Supported**:
- **Chat**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **Instruction**: `{"instruction": "...", "output": "..."}`

### 3. database.py - Data Persistence

**Database**: SQLite (`llm_dataset_creator.db`)

**Tables**:
```sql
datasets
  - id, name, description, domain, subdomain, format
  - example_count, created_at, updated_at, file_path
  - metadata (JSON: language, quality_score, etc.)

generation_jobs
  - id, dataset_id, status, parameters (JSON)
  - started_at, completed_at
  - examples_generated, examples_requested, errors (JSON)

quality_jobs
  - id, dataset_id, status, parameters (JSON)
  - examples_processed, examples_kept, examples_fixed, examples_removed
  - avg_quality_score, result_file_path, errors (JSON)

examples
  - id, dataset_id, example_index, content (JSON)
  - quality_score, status, metadata (JSON)
  - created_at, updated_at
```

**Key Functions**:
- `get_datasets(domain=None, format_type=None)` - Filter datasets
- `get_dataset_examples(dataset_id, limit=100, offset=0)` - Paginated examples
- `update_example(example_id, content, quality_score)` - Edit & auto-save to JSONL
- `delete_dataset(dataset_id)` - Soft delete + remove file

**Important**: All example updates automatically sync to JSONL files via `save_dataset_to_jsonl()`

### 4. domains.py - Domain Configurations

**Purpose**: Define 12 specialized domains with subdomains and parameters

**Structure**:
```python
DOMAINS = {
    "Domain Name": {
        "subdomains": ["Subdomain1", "Subdomain2", ...],
        "specific_params": {
            "param_name": {
                "values": ["value1", "value2", ...],
                "weights": [0.3, 0.7, ...]
            }
        }
    }
}

COMMON_PARAMS = {
    "conversation_types": {...},
    "complexity_levels": {...},
    "question_structures": {...},
    "emotional_tones": {...},
    "communication_styles": {...}
}
```

**12 Domains**:
1. Customer Support (tech support, account, returns, billing, complaints)
2. Medical Information (consultations, clinical, diagnostics, patient ed, research)
3. Legal Documentation (contracts, litigation, compliance, IP, real estate)
4. Educational Content (instructional, tutoring, assessments, curriculum, workshops)
5. Business Communication (internal, external, reports, proposals, negotiations)
6. Technical Documentation (API docs, manuals, specs, troubleshooting, architecture)
7. Sales & Negotiation (pitches, objections, discovery, closing, upselling)
8. Financial Analysis (investment, market, risk, budgeting, audits)
9. Research Summaries (literature, methodology, findings, meta-analysis, proposals)
10. Coaching & Mentoring (career, skills, performance, leadership, life)
11. Creative Writing (narrative, dialogue, descriptive, character, worldbuilding)
12. Meeting Summaries (executive, team, project, client, board)

**How to Add a Domain**:
1. Add entry to `DOMAINS` dict with subdomains
2. Define `specific_params` with weighted values
3. Update `generate_scenario_parameters()` in `generator.py`
4. Add domain-specific prompt logic in `get_prompt_in_language()`

### 5. quality.py - Quality Control

**Class**: `QualityController`

**Purpose**: Validate and improve dataset quality using LLM evaluation

**Key Methods**:
```python
validate_example_structure(example, format_type)
  → Check required fields (messages, instruction/output)

evaluate_example_quality(example, format_type)
  → LLM-based scoring (0-10) on 5 criteria

improve_example(example, format_type, evaluation)
  → LLM-based auto-fix for low-quality examples

process_dataset_quality(dataset_id, batch_size, min_quality_score, auto_fix, auto_remove)
  → Async batch processing with status tracking
```

**Quality Criteria**:
1. Content quality (natural, coherent, informative)
2. Instruction-response alignment
3. Factual correctness
4. Formatting and structure
5. Training utility

**Process Flow**:
1. Create quality job in database
2. Load dataset examples
3. For each example:
   - Validate structure
   - Evaluate quality (LLM call)
   - If score < threshold:
     - Auto-fix: Call LLM to improve
     - Auto-remove: Mark as removed
   - Save results
4. Create improved dataset file (`*_improved.jsonl`)
5. Update job status with metrics

### 6. llm_providers.py - LLM Integration

**Abstract Base**: `LLMProvider`

**Interface**:
```python
generate_text(prompt: str, max_tokens: int) -> str
is_available() -> bool
get_models() -> List[str]
```

**Providers**:

1. **OllamaProvider** (Default)
   - Local LLM server (http://localhost:11434)
   - No API key required
   - Models: gemma3:27b, llama3, mistral, etc.
   - JSON extraction utilities

2. **OpenAIProvider**
   - API key required ($)
   - Models: gpt-4, gpt-4-turbo, gpt-3.5-turbo
   - Chat completions endpoint

3. **AnthropicProvider**
   - API key required ($)
   - Models: claude-3-7-sonnet-20250219, claude-3-opus
   - Messages endpoint

**Factory Function**:
```python
create_provider(provider_type: str, config: dict) -> LLMProvider
```

**How to Add a Provider**:
1. Create class inheriting from `LLMProvider`
2. Implement `generate_text()`, `is_available()`, `get_models()`
3. Add to `create_provider()` factory
4. Update `get_provider_types()` list
5. Add UI configuration in `Settings.js`

### 7. utils.py - Utility Functions

**Key Functions**:

**File Operations**:
- `scan_for_datasets(data_dir)` - Auto-discover JSONL files
- `load_jsonl(file_path)` - Load JSONL as list of dicts
- `save_jsonl(file_path, data)` - Save list of dicts as JSONL

**Format Detection & Conversion**:
- `detect_format_type(example)` - Detect "chat" or "instruction"
- `convert_format(example, target_format)` - Convert between formats

**Dataset Manipulation**:
- `merge_datasets(dataset_ids)` - Combine multiple datasets
- `sample_dataset(dataset_id, sample_size)` - Random sampling
- `export_to_csv(dataset_id)` - Export for analysis

**System Monitoring**:
- `get_disk_usage()` - Disk space metrics
- `get_cpu_usage()` - CPU utilization
- `get_memory_usage()` - RAM metrics

---

## Key Frontend Components

### Technology Stack

- **React 18.2** - UI framework
- **Chakra UI 2.8** - Component library & theming
- **TanStack React Query 5.28** - Server state management
- **React Router DOM 6.22** - Routing
- **Axios** - HTTP client
- **Framer Motion** - Animations

### Component Structure

**Main Components** (`frontend/src/components/`):

1. **Dashboard.js**
   - Overview metrics (total datasets, examples, domains)
   - Recent activity
   - System status (CPU, memory, disk)

2. **Generator.js**
   - Multi-step wizard for dataset creation
   - Domain/subdomain selection (card grid)
   - Parameter configuration (examples, format, language)
   - Real-time progress monitoring
   - Uses React Query for status polling

3. **QualityControl.js**
   - Dataset selection dropdown
   - Quality parameters (batch size, min score, auto-fix/remove)
   - Progress tracking with metrics
   - Creates improved dataset versions

4. **Datasets.js**
   - Filterable table (domain, format search)
   - Inline actions (download, delete, view)
   - Upload modal for custom datasets
   - Merge/sample/convert utilities

5. **DatasetDetail.js**
   - Tabbed interface (Overview, Examples, Metadata)
   - Inline example editing
   - Quality score display
   - Export options (JSONL, CSV)

6. **Tasks.js**
   - List of generation/quality jobs
   - Status badges (pending, running, completed, failed)
   - Cancel/delete actions
   - Error display for failed jobs

7. **Settings.js**
   - Provider selection (Ollama/OpenAI/Anthropic)
   - Model configuration
   - API key management
   - System settings (cache, concurrency)

### State Management

**Server State**: React Query
```javascript
const { data, isLoading, error } = useQuery({
  queryKey: ['datasets'],
  queryFn: () => getDatasets()
});

const mutation = useMutation({
  mutationFn: (params) => startGenerator(params),
  onSuccess: () => {
    queryClient.invalidateQueries(['datasets']);
    toast({ title: 'Generation started' });
  }
});
```

**Local State**: `useState` for forms, UI toggles, wizard steps

**No Global State Manager**: React Query cache serves as shared state

### API Service (api.js)

**Base Configuration**:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const api = axios.create({ baseURL: API_URL });
```

**Service Functions**:
- Domain operations: `getDomains()`, `getSubdomains(domain)`
- Provider operations: `getProviders()`, `setProviderConfig(config)`
- Generator: `startGenerator(params)`, `getGeneratorStatus(jobId)`
- Quality: `startQualityControl(params)`, `getQualityStatus(jobId)`
- Datasets: `getDatasets()`, `getDatasetDetails(id)`, `downloadDataset(id)`
- Tasks: `getAllTasks()`, `cancelTask(taskId)`, `deleteTask(taskId)`

---

## Development Workflows

### Local Development Setup

**Backend**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
cd frontend
npm install
npm start  # Runs on http://localhost:3000
```

**Ollama (for local LLM)**:
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:27b
ollama serve  # Runs on http://localhost:11434
```

### Docker Development

**Full Stack**:
```bash
cd docker
docker-compose up -d

# Access:
# Frontend: http://localhost:3002
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Backend Only**:
```bash
cd backend
docker-compose up -d
```

**Environment Variables** (create `.env` files):

**backend/.env**:
```bash
LLM_PROVIDER=ollama          # ollama, openai, anthropic
LLM_MODEL=gemma3:27b
OLLAMA_URL=http://host.docker.internal:11434  # Docker: use host.docker.internal
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DATA_DIR=/app/data
MAX_CONCURRENT_JOBS=2
LOG_LEVEL=info
```

**frontend/.env**:
```bash
REACT_APP_API_URL=http://localhost:8000/api
```

### Testing

**Backend** (manual testing via API docs):
```bash
# Open http://localhost:8000/docs
# Interactive Swagger UI for all endpoints
```

**Frontend** (no tests yet):
```bash
cd frontend
npm test  # No tests configured currently
```

---

## Code Conventions

### Backend (Python)

**File Naming**: `snake_case.py`

**Functions**: `snake_case()`
```python
def get_dataset_examples(dataset_id: int, limit: int = 100) -> List[dict]:
    """Retrieve paginated examples for a dataset."""
    pass
```

**Classes**: `PascalCase`
```python
class DatasetGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
```

**Constants**: `UPPER_CASE`
```python
DOMAINS = {...}
COMMON_PARAMS = {...}
DEFAULT_BATCH_SIZE = 10
```

**Type Hints**: Always use for function signatures
```python
from typing import List, Dict, Optional

def merge_datasets(dataset_ids: List[int]) -> Optional[Dict]:
    pass
```

**Async Functions**: Prefix with `async`, use `await`
```python
async def generate_examples_batch(dataset_id: int, num_examples: int):
    async with aiofiles.open(file_path, 'a') as f:
        await f.write(json.dumps(example) + '\n')
```

**Error Handling**: Use try-except with detailed messages
```python
try:
    result = await llm_provider.generate_text(prompt)
except Exception as e:
    logger.error(f"Generation failed: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

**Docstrings**: Module and function-level
```python
"""
Module for dataset generation using LLM providers.
Supports multiple domains, formats, and quality control.
"""

def generate_scenario_parameters(domain: str, subdomain: str) -> dict:
    """
    Generate weighted random scenario parameters for a given domain.

    Args:
        domain: Domain name (e.g., "Customer Support")
        subdomain: Subdomain name (e.g., "Technical Support")

    Returns:
        Dictionary of scenario parameters with weighted values
    """
    pass
```

### Frontend (JavaScript/React)

**Component Files**: `PascalCase.js`

**Utility Files**: `camelCase.js`

**Components**: Functional with hooks
```javascript
function Generator() {
  const [step, setStep] = useState(1);
  const { data: domains } = useQuery(['domains'], getDomains);

  return (
    <Box>...</Box>
  );
}
```

**Functions**: `camelCase()`
```javascript
const getDomains = async () => {
  const response = await api.get('/domains');
  return response.data;
};
```

**Constants**: `UPPER_CASE` or `camelCase` for configs
```javascript
const API_URL = process.env.REACT_APP_API_URL;
const defaultPageSize = 20;
```

**Hooks**: `use` prefix
```javascript
const useDatasets = (filters) => {
  return useQuery(['datasets', filters], () => getDatasets(filters));
};
```

**Props**: Destructure in function signature
```javascript
function DatasetCard({ dataset, onDelete, onView }) {
  return <Box>...</Box>;
}
```

**Styling**: Use Chakra UI props
```javascript
<Box p={4} borderWidth="1px" borderRadius="lg" bg="white">
  <Heading size="md">{dataset.name}</Heading>
</Box>
```

---

## Common Development Tasks

### Add a New Domain

1. **Update domains.py**:
```python
DOMAINS["New Domain"] = {
    "subdomains": ["Subdomain1", "Subdomain2"],
    "specific_params": {
        "param_name": {
            "values": ["value1", "value2"],
            "weights": [0.5, 0.5]
        }
    }
}
```

2. **Update generator.py**:
```python
# In generate_scenario_parameters()
if domain == "New Domain":
    scenario_params["specific_param"] = random.choices(
        specific_params["param_name"]["values"],
        weights=specific_params["param_name"]["weights"]
    )[0]
```

3. **Update prompts** (if needed):
```python
# In get_prompt_in_language()
if domain == "New Domain":
    domain_context = "Specific instructions for new domain..."
```

### Add a New API Endpoint

1. **Backend (main.py)**:
```python
@app.get("/api/custom-endpoint")
async def custom_endpoint(param: str = Query(...)):
    """Custom endpoint description."""
    result = perform_operation(param)
    return {"status": "success", "data": result}
```

2. **Frontend (api.js)**:
```javascript
export const customEndpoint = async (param) => {
  const response = await api.get('/custom-endpoint', { params: { param } });
  return response.data;
};
```

3. **Use in component**:
```javascript
const { data } = useQuery(['custom', param], () => customEndpoint(param));
```

### Add a New LLM Provider

1. **Create provider class (llm_providers.py)**:
```python
class NewProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_text(self, prompt: str, max_tokens: int = 2000) -> str:
        # Implementation
        pass

    def is_available(self) -> bool:
        # Check availability
        pass

    def get_models(self) -> List[str]:
        return ["model1", "model2"]
```

2. **Update factory (llm_providers.py)**:
```python
def create_provider(provider_type: str, config: dict) -> LLMProvider:
    if provider_type == "new_provider":
        return NewProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "default-model")
        )
```

3. **Update frontend Settings.js**:
```javascript
const providers = [
  { value: 'ollama', label: 'Ollama' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'new_provider', label: 'New Provider' }
];
```

### Modify Dataset Format

1. **Update format detection (utils.py)**:
```python
def detect_format_type(example: dict) -> str:
    if "new_format_field" in example:
        return "new_format"
    # existing logic
```

2. **Update generation (generator.py)**:
```python
if format_type == "new_format":
    example = {
        "new_format_field": generated_content,
        "metadata": scenario_params
    }
```

3. **Update quality validation (quality.py)**:
```python
def validate_example_structure(example: dict, format_type: str) -> dict:
    if format_type == "new_format":
        required_fields = ["new_format_field"]
        # validation logic
```

### Debug Generation Issues

1. **Check logs**:
```bash
# Backend logs
docker logs dataset-creator-backend -f

# Or if running locally
# Logs appear in console with uvicorn --reload
```

2. **Check database**:
```bash
sqlite3 backend/llm_dataset_creator.db
SELECT * FROM generation_jobs ORDER BY started_at DESC LIMIT 5;
SELECT * FROM datasets WHERE id = X;
SELECT * FROM examples WHERE dataset_id = X LIMIT 10;
```

3. **Check JSONL files**:
```bash
cat backend/data/datasets/domain_timestamp.jsonl | jq .
```

4. **Test LLM provider**:
```python
# In Python REPL
from llm_providers import create_provider

provider = create_provider("ollama", {"url": "http://localhost:11434", "model": "gemma3:27b"})
print(provider.is_available())
result = provider.generate_text("Say hello")
print(result)
```

---

## API Reference

### Core Endpoints

**Domains**:
- `GET /api/domains` - List all domains
- `GET /api/domains/{domain}` - Get domain details
- `GET /api/domains/{domain}/subdomains` - Get subdomains

**Providers**:
- `GET /api/providers` - List provider types
- `GET /api/providers/{provider}/models` - List models for provider
- `POST /api/providers/configure` - Set provider config
- `GET /api/providers/current` - Get current provider config

**Generator**:
- `POST /api/generator/start` - Start generation job
  ```json
  {
    "domain": "Customer Support",
    "subdomain": "Technical Support",
    "num_examples": 100,
    "format_type": "chat",
    "language": "en"
  }
  ```
- `GET /api/generator/status/{job_id}` - Get job status
- `POST /api/generator/cancel/{job_id}` - Cancel job

**Quality Control**:
- `POST /api/quality/start` - Start quality check job
  ```json
  {
    "dataset_id": 1,
    "batch_size": 10,
    "min_quality_score": 7.0,
    "auto_fix": true,
    "auto_remove": false
  }
  ```
- `GET /api/quality/status/{job_id}` - Get job status
- `POST /api/quality/cancel/{job_id}` - Cancel job

**Datasets**:
- `GET /api/datasets` - List datasets (filters: domain, format)
- `GET /api/datasets/{id}` - Get dataset details
- `GET /api/datasets/{id}/examples` - Get examples (pagination: limit, offset)
- `DELETE /api/datasets/{id}` - Delete dataset
- `GET /api/datasets/{id}/download` - Download JSONL file
- `POST /api/datasets/upload` - Upload JSONL file (multipart/form-data)

**Examples**:
- `GET /api/examples/{id}` - Get example details
- `PUT /api/examples/{id}` - Update example
  ```json
  {
    "content": {...},
    "quality_score": 8.5
  }
  ```
- `DELETE /api/examples/{id}` - Delete example

**Tasks**:
- `GET /api/tasks` - List all jobs (filters: status, type)
- `GET /api/tasks/{id}` - Get job details
- `POST /api/tasks/{id}/cancel` - Cancel task
- `DELETE /api/tasks/{id}` - Delete task record

**Utils**:
- `POST /api/utils/merge` - Merge datasets
  ```json
  {"dataset_ids": [1, 2, 3], "name": "Merged Dataset"}
  ```
- `POST /api/utils/sample` - Sample dataset
  ```json
  {"dataset_id": 1, "sample_size": 50}
  ```
- `POST /api/utils/convert` - Convert format
  ```json
  {"dataset_id": 1, "target_format": "instruction"}
  ```

**System**:
- `GET /api/system/status` - System metrics (CPU, memory, disk)
- `GET /api/settings` - Get system settings
- `PUT /api/settings` - Update settings

---

## Database Schema Reference

### datasets

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| name | TEXT NOT NULL | Dataset name |
| description | TEXT | Optional description |
| domain | TEXT NOT NULL | Domain name |
| subdomain | TEXT | Subdomain name |
| format | TEXT NOT NULL | "chat" or "instruction" |
| example_count | INTEGER | Number of examples |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |
| file_path | TEXT | JSONL file path |
| metadata | JSON | Additional metadata |

### generation_jobs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| dataset_id | INTEGER | Foreign key to datasets |
| status | TEXT | "pending", "running", "completed", "failed", "cancelled" |
| parameters | JSON | Generation parameters |
| started_at | TIMESTAMP | Job start time |
| completed_at | TIMESTAMP | Job completion time |
| examples_generated | INTEGER | Number generated |
| examples_requested | INTEGER | Number requested |
| errors | JSON | Error messages |

### quality_jobs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| dataset_id | INTEGER | Foreign key to datasets |
| status | TEXT | Job status |
| parameters | JSON | Quality parameters |
| started_at | TIMESTAMP | Job start time |
| completed_at | TIMESTAMP | Job completion time |
| examples_processed | INTEGER | Number processed |
| examples_kept | INTEGER | Number kept |
| examples_fixed | INTEGER | Number improved |
| examples_removed | INTEGER | Number removed |
| avg_quality_score | REAL | Average quality score |
| result_file_path | TEXT | Improved dataset path |
| errors | JSON | Error messages |

### examples

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| dataset_id | INTEGER | Foreign key to datasets |
| example_index | INTEGER | Index in dataset |
| content | JSON | Example data (messages or instruction/output) |
| quality_score | REAL | Quality score (0-10) |
| status | TEXT | "active", "removed" |
| metadata | JSON | Additional metadata |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |

---

## Configuration Reference

### Environment Variables

**Backend** (`.env` in `backend/`):
```bash
# LLM Provider
LLM_PROVIDER=ollama                    # ollama, openai, anthropic
LLM_MODEL=gemma3:27b                   # Model name

# Ollama Configuration
OLLAMA_URL=http://localhost:11434      # Ollama endpoint
# For Docker: http://host.docker.internal:11434

# OpenAI Configuration
OPENAI_API_KEY=sk-...                  # OpenAI API key
OPENAI_MODEL=gpt-4                     # Optional: override default model

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-...           # Anthropic API key
ANTHROPIC_MODEL=claude-3-7-sonnet-20250219  # Optional

# System Configuration
DATA_DIR=/app/data                     # Data storage directory
MAX_CONCURRENT_JOBS=2                  # Max parallel generation jobs
ENABLE_CACHING=true                    # Enable response caching
CACHE_TTL=24                           # Cache TTL (hours)
LOG_LEVEL=info                         # Logging level: debug, info, warning, error

# Database
DATABASE_PATH=llm_dataset_creator.db   # SQLite database file
```

**Frontend** (`.env` in `frontend/`):
```bash
REACT_APP_API_URL=http://localhost:8000/api  # Backend API URL
```

### Docker Configuration

**docker/docker-compose.yml** (full stack):
```yaml
version: '3.8'

services:
  backend:
    build: ../backend
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_URL=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"

  frontend:
    build: ../frontend
    ports:
      - "3002:80"
    depends_on:
      - backend
```

**Important Docker Notes**:
- Frontend uses Nginx on port 80 (mapped to host 3002)
- Backend needs `host.docker.internal` to access host's Ollama
- Data directory mounted as volume for persistence
- `extra_hosts` required on Linux for host access

---

## Important Notes for AI Assistants

### When Working on This Codebase

1. **Always Check Provider Availability**
   - Before suggesting LLM operations, verify provider is configured
   - Ollama requires `ollama serve` running locally
   - OpenAI/Anthropic require valid API keys

2. **Understand Async Patterns**
   - Backend uses async/await extensively
   - Long operations use background tasks
   - Status polling via frontend React Query

3. **Database & File Sync**
   - Examples stored in BOTH database and JSONL files
   - Updates to examples auto-sync to JSONL
   - Deleting dataset removes both DB records and files

4. **Format Types Matter**
   - Chat format: `{"messages": [...]}`
   - Instruction format: `{"instruction": "...", "output": "..."}`
   - Conversion is possible but may lose context

5. **Quality Control is Expensive**
   - Each quality check calls LLM (costs $$$ for paid providers)
   - Auto-fix doubles the cost (evaluation + improvement)
   - Use batch sizes wisely (default: 10)

6. **Domain Configuration is Complex**
   - Domains have weighted parameters for realistic variance
   - Changing domain structure requires generator updates
   - Test with small batches first

7. **Background Jobs Don't Block**
   - API returns immediately with job ID
   - Frontend polls status every 2 seconds
   - Jobs can be cancelled or fail silently

8. **Error Handling**
   - Backend returns detailed error messages in JSON
   - Frontend shows toast notifications
   - Check browser console and backend logs for details

9. **File Paths**
   - All dataset files in `backend/data/datasets/`
   - Naming: `{domain}_{timestamp}.jsonl`
   - Improved versions: `{original}_improved.jsonl`

10. **Docker vs Local Development**
    - Docker uses different host for Ollama (`host.docker.internal`)
    - Local dev uses `localhost:11434`
    - Volumes required for data persistence

### Code Modification Guidelines

**Before Making Changes**:
1. Read relevant module completely
2. Understand data flow (request → API → service → database → file)
3. Check if similar functionality exists elsewhere
4. Consider impact on JSONL files and database sync

**When Adding Features**:
1. Update backend API first (main.py + service module)
2. Test with curl or API docs (/docs)
3. Add frontend API service function (api.js)
4. Create/update React component
5. Use React Query for data fetching
6. Add error handling and loading states

**When Debugging**:
1. Check browser console for frontend errors
2. Check backend logs for API errors
3. Verify database state with SQL queries
4. Inspect JSONL files directly
5. Test LLM provider separately

**Best Practices**:
1. Always use type hints in Python
2. Always use async for I/O operations
3. Always validate input with Pydantic
4. Always handle errors gracefully
5. Always update both DB and files together
6. Always use React Query for server state
7. Always show loading/error states in UI
8. Always test with small batches first
9. Always check provider availability
10. Always commit with descriptive messages

### Common Pitfalls to Avoid

1. **Don't** call LLM providers synchronously (blocks server)
2. **Don't** forget to update JSONL when updating database
3. **Don't** assume provider is available (always check)
4. **Don't** use large batch sizes in quality control (expensive!)
5. **Don't** modify domain configs without testing generation
6. **Don't** forget CORS when adding new endpoints
7. **Don't** hardcode file paths (use `DATA_DIR` env var)
8. **Don't** skip error handling in background tasks
9. **Don't** forget to invalidate React Query cache on mutations
10. **Don't** commit API keys or sensitive data

---

## Useful Commands

### Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm start

# Database inspection
sqlite3 backend/llm_dataset_creator.db
.tables
.schema datasets
SELECT * FROM datasets;

# JSONL inspection
cat backend/data/datasets/file.jsonl | jq .
cat backend/data/datasets/file.jsonl | wc -l  # Count examples

# Ollama
ollama list  # List installed models
ollama pull gemma3:27b  # Pull model
ollama serve  # Start server
```

### Docker

```bash
# Full stack
cd docker
docker-compose up -d
docker-compose logs -f
docker-compose down

# Rebuild after changes
docker-compose up -d --build

# Backend only
cd backend
docker-compose up -d

# Clean up
docker-compose down -v  # Remove volumes too
```

### Git

```bash
# Current branch
git status
git branch

# Commit changes
git add .
git commit -m "feat: add new domain configuration"
git push -u origin <branch-name>

# View changes
git diff
git log --oneline -10
```

---

## Resources

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Frontend**: http://localhost:3000 (dev) or http://localhost:3002 (Docker)
- **Repository**: https://github.com/KazKozDev/dataset-creator
- **Issues**: https://github.com/KazKozDev/dataset-creator/issues
- **Ollama**: https://ollama.ai
- **FastAPI**: https://fastapi.tiangolo.com
- **React Query**: https://tanstack.com/query/latest
- **Chakra UI**: https://chakra-ui.com

---

**Last Updated**: 2025-11-14
**Version**: 1.1.1
