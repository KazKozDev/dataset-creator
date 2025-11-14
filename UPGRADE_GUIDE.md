# Dataset Creator - Upgrade Guide

## 🎉 New Features

Your dataset creator has been upgraded with powerful new capabilities:

### 1. **Data Collection Modules** 📥
- Web scraping from any website
- API integrations (GitHub, StackOverflow, RSS)
- File parsing (TXT, JSON, Markdown)

### 2. **Deduplication & Cleaning** 🧹
- Remove exact duplicates (hash-based)
- Remove fuzzy duplicates (similarity-based)
- Filter by content length
- Remove toxic content
- Fix encoding issues

### 3. **Dataset Versioning** 📚
- Track all changes to datasets
- Rollback to previous versions
- Compare versions
- Automatic snapshots

### 4. **Configuration System** ⚙️
- YAML/JSON based pipelines
- No-code configuration
- Environment variable support
- Multiple example templates

---

## 📦 Installation

### Update Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New dependencies added:
- `aiohttp` - Async HTTP client
- `beautifulsoup4` - HTML parsing
- `feedparser` - RSS feed parsing
- `pyyaml` - YAML configuration support
- `lxml` - XML/HTML processing

---

## 🚀 Quick Start

### Example 1: Generate Synthetic Data for IT Support Chatbot

```bash
# Using config file
python -c "from config_loader import load_pipeline_config; config = load_pipeline_config('configs/examples/basic_synthetic.yaml'); print(config.name)"
```

**Config file** (`configs/examples/basic_synthetic.yaml`):
```yaml
name: "it_support_chatbot_dataset"
description: "Generate synthetic training data for IT support chatbot"

llm_provider:
  provider: "ollama"
  model: "gemma3:27b"

data_sources:
  - type: "synthetic"
    enabled: true
    config:
      domain: "support"
      subdomain: "technical"
      count: 500
```

### Example 2: Scrape Documentation Website

**API Call**:
```bash
curl -X POST http://localhost:8000/api/v2/collect/web-scrape \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://docs.python.org/3/tutorial/"],
    "selector": "div.body",
    "convert_to_examples": true,
    "dataset_name": "python_docs",
    "provider": "openai",
    "model": "gpt-5.1"
  }'
```

**Python**:
```python
import asyncio
from data_collectors import WebScraperCollector
from llm_providers import create_provider

async def scrape_docs():
    # Create LLM provider
    llm = create_provider("openai", model="gpt-5.1", api_key="YOUR_KEY")

    # Create scraper
    scraper = WebScraperCollector(llm)

    # Scrape URLs
    data = await scraper.scrape_urls(
        urls=["https://docs.python.org/3/tutorial/"],
        selector="div.body"
    )

    # Convert to training examples
    examples = scraper.convert_to_training_examples(data)

    return examples

# Run
examples = asyncio.run(scrape_docs())
```

### Example 3: Collect GitHub Issues

**API Call**:
```bash
curl -X POST http://localhost:8000/api/v2/collect/github-issues \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "fastapi/fastapi",
    "state": "all",
    "max_issues": 200,
    "dataset_name": "fastapi_issues"
  }'
```

**Python**:
```python
import asyncio
from data_collectors import APICollector

async def collect_issues():
    collector = APICollector(api_key="YOUR_GITHUB_TOKEN")

    # Fetch issues
    issues = await collector.fetch_github_issues(
        repo="fastapi/fastapi",
        state="all",
        max_issues=200
    )

    # Convert to training examples
    examples = collector.convert_to_training_examples(
        issues,
        data_type='github_issues'
    )

    return examples

# Run
examples = asyncio.run(collect_issues())
```

### Example 4: Remove Duplicates

**API Call**:
```bash
curl -X POST http://localhost:8000/api/v2/process/deduplicate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "method": "fuzzy",
    "similarity_threshold": 0.9,
    "create_new_dataset": true
  }'
```

**Python**:
```python
from deduplication import DataDeduplicator
import utils

# Load dataset
examples = utils.load_jsonl("data/datasets/my_dataset.jsonl")

# Deduplicate
deduplicator = DataDeduplicator(similarity_threshold=0.9)
unique, stats = deduplicator.deduplicate_dataset(
    examples=examples,
    format_type='chat',
    method='fuzzy'
)

print(f"Removed {stats['duplicates_removed']} duplicates")
print(f"Kept {stats['unique_count']} unique examples")
```

### Example 5: Clean Dataset

**API Call**:
```bash
curl -X POST http://localhost:8000/api/v2/process/clean \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "min_length": 20,
    "max_length": 5000,
    "remove_toxic": true,
    "create_new_dataset": true
  }'
```

**Python**:
```python
from deduplication import DataCleaner
import utils

# Load dataset
examples = utils.load_jsonl("data/datasets/my_dataset.jsonl")

# Clean
cleaner = DataCleaner()
filtered, stats = cleaner.filter_dataset(
    examples=examples,
    format_type='chat',
    min_length=20,
    max_length=5000,
    remove_toxic=True
)

print(f"Filtered: {stats}")
```

### Example 6: Dataset Versioning

**Create Version**:
```bash
curl -X POST http://localhost:8000/api/v2/datasets/1/versions \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "description": "Before quality improvements",
    "metadata": {"operation": "quality_control"}
  }'
```

**List Versions**:
```bash
curl http://localhost:8000/api/v2/datasets/1/versions
```

**Rollback**:
```bash
curl -X POST http://localhost:8000/api/v2/datasets/1/rollback \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "version_id": "abc123def456"
  }'
```

**Python**:
```python
from versioning import version_manager

# Create version
version = version_manager.create_version(
    dataset_id=1,
    file_path="data/datasets/my_dataset.jsonl",
    description="Before improvements"
)

# List versions
versions = version_manager.list_versions(dataset_id=1)

# Rollback
version_manager.set_current_version(
    dataset_id=1,
    version_id="abc123def456"
)
```

---

## 📝 Configuration Examples

### Basic Synthetic Generation

```yaml
name: "basic_pipeline"
llm_provider:
  provider: "ollama"
  model: "gemma3:27b"

data_sources:
  - type: "synthetic"
    config:
      domain: "support"
      count: 100

deduplication:
  enabled: true
  method: "hash"

output:
  dataset_name: "my_dataset"
```

### Web Scraping

```yaml
name: "web_scraping_pipeline"
llm_provider:
  provider: "openai"
  model: "gpt-5.1"
  api_key: "${OPENAI_API_KEY}"

data_sources:
  - type: "web"
    config:
      start_url: "https://example.com/docs"
      max_pages: 50
      content_selector: "article"

deduplication:
  enabled: true
  method: "fuzzy"
  similarity_threshold: 0.85

cleaning:
  enabled: true
  min_length: 50
```

### Multi-Source Pipeline

```yaml
name: "multi_source_pipeline"

data_sources:
  # Synthetic data
  - type: "synthetic"
    config:
      domain: "support"
      count: 200

  # Web scraping
  - type: "web"
    config:
      urls: ["https://example.com/faq"]

  # GitHub issues
  - type: "api"
    config:
      api_type: "github"
      params:
        repo: "owner/repo"
        max_issues: 100

deduplication:
  enabled: true
  method: "fuzzy"

quality_control:
  enabled: true
  threshold: 7.0
  auto_fix: true
```

---

## 🔧 Module Reference

### Data Collectors

#### WebScraperCollector
- `scrape_urls()` - Scrape multiple URLs
- `crawl_website()` - Crawl entire website
- `convert_to_training_examples()` - Convert scraped data to examples

#### APICollector
- `fetch_github_issues()` - Fetch GitHub issues
- `fetch_github_discussions()` - Fetch GitHub discussions
- `fetch_stackoverflow_questions()` - Fetch StackOverflow questions
- `fetch_rss_feed()` - Fetch RSS/Atom feed

#### FileParserCollector
- `parse_text_file()` - Parse text files
- `parse_json_file()` - Parse JSON files
- `parse_markdown_file()` - Parse Markdown files
- `convert_documents_to_qa_pairs()` - Convert docs to Q&A

### Deduplication

#### DataDeduplicator
- `deduplicate_dataset()` - Remove duplicates
- Methods: `exact`, `hash`, `fuzzy`

#### DataCleaner
- `filter_dataset()` - Filter and clean dataset
- `clean_example()` - Clean single example

#### DataAnalyzer
- `analyze_dataset()` - Comprehensive analysis
- `calculate_length_stats()` - Length statistics

### Versioning

#### VersionManager
- `create_version()` - Create new version
- `list_versions()` - List all versions
- `get_current_version()` - Get current version
- `set_current_version()` - Rollback to version
- `compare_versions()` - Compare two versions

---

## 🌐 API Endpoints (v2)

All new endpoints are under `/api/v2/`

### Data Collection
- `POST /api/v2/collect/web-scrape` - Scrape websites
- `POST /api/v2/collect/web-crawl` - Crawl website
- `POST /api/v2/collect/github-issues` - Collect GitHub issues
- `POST /api/v2/collect/stackoverflow` - Collect StackOverflow questions

### Data Processing
- `POST /api/v2/process/deduplicate` - Remove duplicates
- `POST /api/v2/process/clean` - Clean dataset

### Versioning
- `POST /api/v2/datasets/{id}/versions` - Create version
- `GET /api/v2/datasets/{id}/versions` - List versions
- `POST /api/v2/datasets/{id}/rollback` - Rollback version

### Configuration
- `GET /api/v2/configs` - List configurations
- `POST /api/v2/configs/execute` - Execute pipeline
- `POST /api/v2/configs/examples` - Create example configs

---

## 💡 Use Cases

### 1. IT Support Chatbot Training Data

```yaml
# Use synthetic generation + GitHub issues
data_sources:
  - type: "synthetic"
    config:
      domain: "support"
      subdomain: "technical"
      count: 500

  - type: "api"
    config:
      api_type: "github"
      params:
        repo: "your-product/issues"
        state: "closed"
```

### 2. Documentation Q&A System

```yaml
# Scrape documentation + convert to Q&A
data_sources:
  - type: "web"
    config:
      start_url: "https://docs.yourproduct.com"
      max_pages: 200
      content_selector: "main"
```

### 3. Code Assistant Training

```yaml
# Combine StackOverflow + GitHub discussions
data_sources:
  - type: "api"
    config:
      api_type: "stackoverflow"
      params:
        tag: "python"
        max_questions: 500

  - type: "api"
    config:
      api_type: "github"
      params:
        repo: "python/cpython"
```

---

## 🔐 Environment Variables

Set these in `.env` file:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_URL=http://localhost:11434

# APIs
GITHUB_TOKEN=ghp_...

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Application
DATA_DIR=data/
MAX_CONCURRENT_JOBS=2
```

---

## 📊 Best Practices

### 1. Always Deduplicate
Remove duplicates to improve training quality:
```yaml
deduplication:
  enabled: true
  method: "fuzzy"  # Best for natural language
  similarity_threshold: 0.88
```

### 2. Enable Quality Control for Important Datasets
```yaml
quality_control:
  enabled: true
  threshold: 7.5
  auto_fix: true  # Fix low-quality examples
```

### 3. Version Before Major Changes
```python
# Create version before processing
version_manager.create_version(
    dataset_id=1,
    file_path=dataset['file_path'],
    description="Before deduplication"
)
```

### 4. Use Multi-Source for Better Coverage
Combine synthetic + real data for best results:
- Synthetic: 40-60%
- Web scraping: 20-30%
- API data: 20-30%

---

## 🐛 Troubleshooting

### Web Scraping Issues

**Problem**: Can't scrape website
- Check `robots.txt`
- Verify CSS selector
- Use rate limiting (add delays)

**Solution**:
```python
scraper = WebScraperCollector()
await scraper.crawl_website(
    start_url="...",
    max_pages=50,
    content_selector="article.content"  # Correct selector
)
```

### API Rate Limits

**Problem**: GitHub/StackOverflow rate limits
- Use API tokens
- Reduce `max_issues`/`max_questions`
- Add delays between requests

### Memory Issues

**Problem**: Out of memory with large datasets
- Process in batches
- Use `save_intermediate: true`
- Reduce `batch_size`

---

## 📚 Additional Resources

- [Configuration Schema](backend/config_loader.py)
- [API Documentation](http://localhost:8000/docs)
- [Example Configs](configs/examples/)

---

## 🎯 Next Steps

1. **Try Examples**: Start with example configs in `configs/examples/`
2. **Create Your Pipeline**: Customize config for your use case
3. **Process Data**: Run deduplication + cleaning
4. **Quality Control**: Enable LLM-based quality checks
5. **Version Control**: Track changes with versioning

Happy dataset building! 🚀
