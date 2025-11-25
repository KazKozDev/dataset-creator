# LLM Provider Configuration Guide

## Overview

The dataset creator now uses a flexible YAML-based configuration system for managing LLM providers. This allows you to easily switch between different providers, configure models, and manage API keys without modifying code.

## Configuration File

The main configuration file is located at `backend/config.yaml`. This file contains all settings for LLM providers, generation parameters, and system settings.

### Default Configuration

By default, the system uses **Ollama** with the **gpt-oss:20b** model, providing a privacy-focused, cost-free solution for running locally.

```yaml
default_provider: "ollama"
```

## Supported Providers

### 1. Ollama (Local - Default)

Ollama allows you to run LLMs locally without API costs.

**Configuration:**
```yaml
ollama:
  enabled: true
  base_url: "http://localhost:11434"
  default_model: "gpt-oss:20b"
  timeout: 300
  available_models:
    - "gpt-oss:20b"
    - "llama3:8b"
    - "llama3:70b"
    - "gemma3:27b"
```

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull gpt-oss:20b

# Or any other model
ollama pull llama3:8b
```

### 2. OpenAI (Cloud)

Use OpenAI's GPT models for high-quality generation.

**Configuration:**
```yaml
openai:
  enabled: false
  api_key: "${OPENAI_API_KEY}"  # Set via environment variable
  default_model: "gpt-4-turbo-preview"
  timeout: 120
```

**Setup:**
```bash
# Set your API key as an environment variable
export OPENAI_API_KEY="sk-..."

# Or create a .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### 3. Anthropic Claude (Cloud)

Use Anthropic's Claude models for advanced reasoning.

**Configuration:**
```yaml
anthropic:
  enabled: false
  api_key: "${ANTHROPIC_API_KEY}"  # Set via environment variable
  default_model: "claude-3-7-sonnet-20250219"
  timeout: 120
```

**Setup:**
```bash
# Set your API key as an environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Or create a .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

## Switching Providers

### Method 1: Edit config.yaml

Change the `default_provider` field:

```yaml
default_provider: "openai"  # or "anthropic", "ollama"
```

Enable the provider:

```yaml
openai:
  enabled: true
  # ... rest of config
```

### Method 2: Use the API

**List all providers:**
```bash
curl http://localhost:8000/api/providers
```

**Check provider status:**
```bash
curl http://localhost:8000/api/providers/ollama/status
```

**Enable a provider:**
```bash
curl -X POST http://localhost:8000/api/providers/openai/enable
```

**Update provider configuration:**
```bash
curl -X PUT http://localhost:8000/api/providers/ollama/config \
  -H "Content-Type: application/json" \
  -d '{
    "default_model": "llama3:8b",
    "timeout": 600
  }'
```

**Reload configuration:**
```bash
curl -X POST http://localhost:8000/api/config/reload
```

## Generation Settings

Configure default parameters for dataset generation:

```yaml
generation:
  default_temperature: 0.7      # Creativity level (0.0-1.0)
  default_max_tokens: 4000      # Maximum response length
  default_batch_size: 10        # Examples per batch
  max_concurrent_requests: 5    # Parallel requests
  retry_attempts: 3             # Retry on failure
  retry_delay: 2                # Seconds between retries
```

## Quality Control Settings

Configure quality assessment parameters:

```yaml
quality:
  default_threshold: 7.0        # Minimum quality score (0-10)
  default_batch_size: 10        # Examples per batch
  auto_fix_enabled: false       # Auto-fix low quality
  auto_remove_enabled: false    # Auto-remove bad examples
```

## System Settings

Configure directories and caching:

```yaml
system:
  data_directory: "./data"
  datasets_directory: "./data/datasets"
  cache_directory: "./data/cache"
  max_concurrent_jobs: 2
  enable_caching: true
  cache_ttl_hours: 24
  log_level: "info"
```

## Environment Variables

Sensitive information (API keys) can be stored as environment variables:

**Create a `.env` file:**
```bash
# backend/.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_URL=http://localhost:11434
```

**In config.yaml, reference them:**
```yaml
api_key: "${OPENAI_API_KEY}"
```

## API Endpoints

### Provider Management

- `GET /api/providers` - List all providers
- `GET /api/providers/models?provider=ollama` - List available models
- `GET /api/providers/{name}/status` - Check provider availability
- `POST /api/providers/{name}/enable` - Enable a provider
- `POST /api/providers/{name}/disable` - Disable a provider
- `PUT /api/providers/{name}/config` - Update provider configuration
- `POST /api/config/reload` - Reload configuration from file

### Generation

- `POST /api/generator/start` - Start generation job
  - Include `provider` and `model` in request body to override defaults

```json
{
  "domain": "support",
  "format": "chat",
  "count": 10,
  "provider": "ollama",
  "model": "llama3:8b"
}
```

## Best Practices

### For Development

1. **Use Ollama locally** - Free, private, no API costs
2. **Start with smaller models** - `llama3:8b` for testing
3. **Increase timeout** for large models - 300-600 seconds

### For Production

1. **Use cloud providers** - OpenAI or Anthropic for quality
2. **Set up API key rotation** - For security
3. **Enable caching** - Reduce API costs
4. **Monitor rate limits** - Configure concurrent requests

### For AI Engineers

1. **Experiment with models** - Try different providers
2. **Tune temperature** - Adjust creativity vs consistency
3. **Use batch processing** - Optimize throughput
4. **Track quality scores** - Monitor dataset quality

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve

# List available models
ollama list
```

### API Key Issues

```bash
# Verify environment variable is set
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Configuration Errors

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('backend/config.yaml'))"

# Reload configuration via API
curl -X POST http://localhost:8000/api/config/reload
```

## Advanced Usage

### Adding Custom Providers

You can add custom LLM providers by:

1. Extending the `LLMProvider` base class
2. Registering the provider with `ProviderRegistry`
3. Adding configuration to `config.yaml`

See `backend/llm_providers.py` for implementation details.

### Provider Registry

The system uses a registry pattern for managing providers:

```python
from llm_providers import ProviderRegistry, get_provider

# Get default provider
provider = get_provider()

# Get specific provider with overrides
provider = get_provider("ollama", model="llama3:70b")

# List registered providers
providers = ProviderRegistry.list_providers()
```

## Examples

### Generate Dataset with Ollama

```bash
curl -X POST http://localhost:8000/api/generator/start \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "support",
    "subdomain": "tech_support",
    "format": "chat",
    "count": 100,
    "language": "en",
    "temperature": 0.7,
    "provider": "ollama",
    "model": "gpt-oss:20b"
  }'
```

### Switch to OpenAI for High Quality

```yaml
# config.yaml
default_provider: "openai"

openai:
  enabled: true
  default_model: "gpt-4-turbo-preview"
```

### Use Multiple Models

Generate datasets with different models:

```bash
# Generate with Ollama
curl -X POST http://localhost:8000/api/generator/start \
  -d '{"domain": "medical", "count": 50, "provider": "ollama", "model": "llama3:70b"}'

# Generate with OpenAI
curl -X POST http://localhost:8000/api/generator/start \
  -d '{"domain": "medical", "count": 50, "provider": "openai", "model": "gpt-4"}'
```

## Migration from Old System

If you were using environment variables before:

**Old way:**
```bash
export LLM_PROVIDER="ollama"
export LLM_MODEL="gemma3:27b"
export OLLAMA_URL="http://localhost:11434"
```

**New way:**
Edit `backend/config.yaml`:
```yaml
default_provider: "ollama"

ollama:
  enabled: true
  base_url: "http://localhost:11434"
  default_model: "gpt-oss:20b"
```

The system will automatically use the new configuration system!
