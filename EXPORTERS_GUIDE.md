# Dataset Exporters Guide

This guide explains how to use the different export formats available in the LLM Dataset Creator.

## Available Export Formats

### 1. HuggingFace Format

**Best for:** Sharing datasets publicly, uploading to HuggingFace Hub

**Format:** Directory with JSONL files + metadata

**API Example:**
```bash
POST /api/datasets/export
{
  "dataset_id": 1,
  "format": "huggingface",
  "output_filename": "my_dataset",
  "additional_params": {
    "dataset_name": "my-awesome-dataset",
    "split": "train"
  }
}
```

**Output includes:**
- `train.jsonl` - Dataset in JSONL format
- `dataset_info.json` - Metadata and schema
- `README.md` - Dataset card
- `push_to_hub.py` - Script to upload to HuggingFace Hub

**Usage:**
```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='train.jsonl')
```

---

### 2. OpenAI Fine-tuning Format

**Best for:** Fine-tuning GPT-3.5/GPT-4 models

**Format:** JSONL with chat messages

**API Example:**
```bash
POST /api/datasets/export
{
  "dataset_id": 1,
  "format": "openai",
  "output_filename": "openai_dataset",
  "system_message": "You are a helpful assistant.",
  "train_split": 0.8
}
```

**Output includes:**
- `dataset_train.jsonl` - Training set (if split < 1.0)
- `dataset_validation.jsonl` - Validation set (if split < 1.0)
- `dataset.validation.txt` - Validation report with cost estimates

**Usage:**
```python
import openai

# Upload file
with open("openai_dataset.jsonl", "rb") as f:
    response = openai.File.create(file=f, purpose='fine-tune')

# Create fine-tuning job
openai.FineTuningJob.create(
    training_file=response['id'],
    model="gpt-3.5-turbo"
)
```

---

### 3. Alpaca Format

**Best for:** Instruction tuning with LLaMA, Alpaca, or similar models

**Format:** JSON array with instruction-input-output structure

**API Example:**
```bash
POST /api/datasets/export
{
  "dataset_id": 1,
  "format": "alpaca",
  "output_filename": "alpaca_dataset"
}
```

**Format Structure:**
```json
[
  {
    "instruction": "Task description",
    "input": "Additional context (optional)",
    "output": "Expected response"
  }
]
```

**Usage with FastChat:**
```bash
fastchat train \
  --model-path your-model \
  --data-path alpaca_dataset.json \
  --data-format alpaca
```

---

### 4. ShareGPT Format

**Best for:** Chat model training (Vicuna, WizardLM, etc.)

**Format:** JSON array with human-gpt conversations

**API Example:**
```bash
POST /api/datasets/export
{
  "dataset_id": 1,
  "format": "sharegpt",
  "output_filename": "sharegpt_dataset"
}
```

**Format Structure:**
```json
[
  {
    "conversations": [
      {"from": "human", "value": "User message"},
      {"from": "gpt", "value": "Assistant response"},
      {"from": "human", "value": "Follow-up"},
      {"from": "gpt", "value": "Response"}
    ]
  }
]
```

**Usage with Axolotl:**
```yaml
datasets:
  - path: sharegpt_dataset.json
    type: sharegpt
```

---

### 5. LangChain Format

**Best for:** RAG systems, vector databases, LangChain applications

**Format:** JSONL with documents/chat/qa_pairs

**API Example:**
```bash
POST /api/datasets/export
{
  "dataset_id": 1,
  "format": "langchain",
  "output_filename": "langchain_dataset",
  "additional_params": {
    "export_type": "documents"  # or "chat" or "qa_pairs"
  }
}
```

**Export Types:**

1. **Documents** (for vector stores):
```json
{
  "page_content": "Document text...",
  "metadata": {"source": "...", "domain": "..."}
}
```

2. **Chat** (for chat applications):
```json
{
  "messages": [
    {"role": "human", "content": "..."},
    {"role": "ai", "content": "..."}
  ],
  "metadata": {...}
}
```

3. **QA Pairs** (for retrieval):
```json
{
  "question": "...",
  "answer": "...",
  "metadata": {...}
}
```

**Usage:**
```python
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json

# Load documents
documents = []
with open("langchain_dataset.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        # Process based on export_type...

# Create vector store
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
```

---

### 6. CSV Format

**Best for:** Data analysis, Excel/Sheets, general purpose

**API Example:**
```bash
POST /api/datasets/{dataset_id}/export-csv
```

---

## Python Client Examples

### Export to Multiple Formats

```python
import requests

BASE_URL = "http://localhost:8000"
dataset_id = 1

# Export to HuggingFace
response = requests.post(f"{BASE_URL}/api/datasets/export", json={
    "dataset_id": dataset_id,
    "format": "huggingface",
    "output_filename": "hf_dataset"
})
print(response.json())

# Export to OpenAI with train/val split
response = requests.post(f"{BASE_URL}/api/datasets/export", json={
    "dataset_id": dataset_id,
    "format": "openai",
    "output_filename": "openai_dataset",
    "train_split": 0.8,
    "system_message": "You are a helpful AI assistant."
})
print(response.json())

# Export to Alpaca
response = requests.post(f"{BASE_URL}/api/datasets/export", json={
    "dataset_id": dataset_id,
    "format": "alpaca",
    "output_filename": "alpaca_dataset"
})
print(response.json())

# Export to LangChain as QA pairs
response = requests.post(f"{BASE_URL}/api/datasets/export", json={
    "dataset_id": dataset_id,
    "format": "langchain",
    "output_filename": "langchain_qa",
    "additional_params": {
        "export_type": "qa_pairs"
    }
})
print(response.json())
```

### List Available Formats

```python
response = requests.get(f"{BASE_URL}/api/export/formats")
formats = response.json()

for fmt in formats['formats']:
    print(f"\nFormat: {fmt['name']}")
    print(f"Description: {fmt['description']}")
    print(f"Use cases: {', '.join(fmt['use_cases'])}")
```

---

## Format Comparison

| Format | Best For | File Type | Multi-turn | Metadata |
|--------|----------|-----------|------------|----------|
| HuggingFace | Public sharing | JSONL | ✅ | ✅ |
| OpenAI | GPT fine-tuning | JSONL | ✅ | ❌ |
| Alpaca | Instruction tuning | JSON | ❌ | ❌ |
| ShareGPT | Chat models | JSON | ✅ | ❌ |
| LangChain | RAG/Vector DB | JSONL | ✅ | ✅ |
| CSV | Analysis | CSV | ❌ | ⚠️ |

---

## Tips & Best Practices

1. **Choose the right format:**
   - OpenAI → Fine-tuning GPT models
   - Alpaca → Instruction-tuned LLaMA models
   - ShareGPT → Multi-turn chat models
   - LangChain → RAG and retrieval systems
   - HuggingFace → Public datasets and sharing

2. **Train/Validation Splits:**
   - Use `train_split` parameter with OpenAI format
   - Typically use 0.8 (80% training, 20% validation)

3. **System Messages:**
   - OpenAI format supports system messages
   - Use for setting assistant behavior

4. **LangChain Export Types:**
   - `documents` → Vector stores (ChromaDB, Pinecone, etc.)
   - `chat` → Chat applications
   - `qa_pairs` → Q&A retrieval systems

5. **Check Export Status:**
   - Exports run in background
   - Check `data/exports/` directory for output files

---

## Troubleshooting

**Export not found:**
- Exports are saved to `data/exports/` directory
- Check background task logs for errors

**Invalid format error:**
- Verify dataset has proper conversation structure
- Use `/api/datasets/{id}/examples` to inspect data

**Large datasets:**
- Use background tasks for large exports
- Monitor server logs for progress

---

## API Reference

### Export Dataset
```
POST /api/datasets/export
```

**Request Body:**
```json
{
  "dataset_id": int,
  "format": "huggingface|openai|alpaca|sharegpt|langchain",
  "output_filename": "string",
  "system_message": "string (optional)",
  "train_split": float (0.0-1.0, optional),
  "additional_params": {}
}
```

### List Export Formats
```
GET /api/export/formats
```

**Response:**
```json
{
  "formats": [
    {
      "name": "string",
      "description": "string",
      "file_extension": "string",
      "features": ["string"],
      "use_cases": ["string"]
    }
  ]
}
```

---

For more examples and documentation, see the generated loader scripts in each export directory.
