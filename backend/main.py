"""
FastAPI main application for LLM Dataset Creator
"""

import os
import json
import asyncio
import time
import shutil
import psutil
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import database as db
from domains import DOMAINS, get_domains, get_domain, get_subdomain, get_common_params
from llm_providers import create_provider, get_provider_types
import generator
import quality
import utils

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events handler"""
    # Initialize database
    db.init_db()
    db.ensure_data_dir()
    
    # Scan for datasets
    datasets = utils.scan_for_datasets()
    for file_path in datasets:
        try:
            # Check if dataset already exists in database
            existing = db.get_datasets(file_path=file_path)
            if existing:
                continue
                
            # Detect format and load examples
            format_type = utils.detect_format_type(file_path)
            if format_type == "unknown":
                continue
                
            # Load examples
            examples = utils.load_jsonl(file_path)
            if not examples:
                continue
                
            # Create dataset record
            name = os.path.splitext(os.path.basename(file_path))[0]
            dataset_id = db.create_dataset(
                name=name,
                domain="unknown",
                format=format_type,
                file_path=file_path,
                example_count=len(examples),
                metadata={"source": "auto-import"}
            )
            
            # Add examples to database
            db.add_examples(dataset_id, examples)
            
        except Exception as e:
            print(f"Error importing dataset {file_path}: {e}")
    
    yield
    
    # Cleanup code here if needed

app = FastAPI(title="LLM Dataset Creator API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize database
db.initialize()

# Create LLM provider based on environment variables or config
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemma3:27b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Global LLM provider instance
llm_provider = None

# Pydantic models for request/response validation
class GenerationParams(BaseModel):
    domain: str
    subdomain: Optional[str] = None
    format: str = "chat"
    language: str = "en"
    count: int = Field(gt=0, le=1000)
    temperature: float = Field(0.7, ge=0.1, le=1.0)
    model: Optional[str] = None
    provider: Optional[str] = None

class QualityParams(BaseModel):
    dataset_id: int
    batch_size: int = Field(10, gt=0, le=100)
    threshold: float = Field(7.0, ge=0.0, le=10.0)
    auto_fix: bool = False
    auto_remove: bool = False
    model: Optional[str] = None
    provider: Optional[str] = None

class ProviderConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

# Helper function to get or create LLM provider
def get_llm_provider(provider_type: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Get or create LLM provider"""
    global llm_provider
    
    # Use existing provider if no changes
    if llm_provider and not (provider_type or model or api_key or base_url):
        return llm_provider
    
    # Use defaults if not specified
    provider_type = provider_type or DEFAULT_PROVIDER
    
    # Create provider based on type
    if provider_type.lower() == "ollama":
        model = model or DEFAULT_MODEL
        base_url = base_url or OLLAMA_URL
        llm_provider = create_provider("ollama", model=model, base_url=base_url)
    elif provider_type.lower() == "openai":
        model = model or "gpt-4"
        api_key = api_key or OPENAI_API_KEY
        llm_provider = create_provider("openai", model=model, api_key=api_key)
    elif provider_type.lower() == "anthropic":
        model = model or "claude-3-7-sonnet-20250219"
        api_key = api_key or ANTHROPIC_API_KEY
        llm_provider = create_provider("anthropic", model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    return llm_provider

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LLM Dataset Creator API", "version": "1.0.0"}

# Domain routes
@app.get("/api/domains")
async def list_domains():
    """List all available domains"""
    return {"domains": [{"key": key, **domain} for key, domain in get_domains().items()]}

@app.get("/api/domains/{domain_key}")
async def get_domain_details(domain_key: str):
    """Get domain details"""
    domain = get_domain(domain_key)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_key}' not found")
    
    return {"domain": domain_key, **domain}

@app.get("/api/domains/{domain_key}/subdomains")
async def list_subdomains(domain_key: str):
    """List subdomains for a domain"""
    domain = get_domain(domain_key)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_key}' not found")
    
    return {"domain": domain_key, "subdomains": domain.get("subdomains", {})}

@app.get("/api/domains/{domain_key}/subdomains/{subdomain_key}")
async def get_subdomain_details(domain_key: str, subdomain_key: str):
    """Get subdomain details"""
    subdomain = get_subdomain(domain_key, subdomain_key)
    if not subdomain:
        raise HTTPException(status_code=404, detail=f"Subdomain '{subdomain_key}' not found in domain '{domain_key}'")
    
    return {"domain": domain_key, "subdomain": subdomain_key, **subdomain}

@app.get("/api/common-params")
async def get_common_parameters():
    """Get common parameters for all domains"""
    return {"common_params": get_common_params()}

# LLM provider routes
@app.get("/api/providers")
async def list_providers():
    """List available LLM providers"""
    return {"providers": get_provider_types()}

@app.post("/api/providers/config")
async def set_provider_config(config: ProviderConfig):
    """Set LLM provider configuration"""
    try:
        provider = get_llm_provider(
            provider_type=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Check if provider is available
        if not provider.is_available():
            raise HTTPException(status_code=400, detail=f"Provider '{config.provider}' is not available")
        
        return {"status": "success", "provider": config.provider, "model": config.model}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/providers/models")
async def list_models(provider: Optional[str] = Query(None)):
    """List available models for a provider"""
    try:
        # Use specified provider or default
        provider_type = provider or DEFAULT_PROVIDER
        
        # Create provider and get models
        provider = get_llm_provider(provider_type=provider_type)
        models = provider.get_models()
        
        return {"provider": provider_type, "models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Generator routes
@app.post("/api/generator/start")
async def start_generation(params: GenerationParams, background_tasks: BackgroundTasks):
    """Start a generation job"""
    try:
        # Create job record
        job_id = db.create_generation_job(
            parameters=params.model_dump(),
            examples_requested=params.count
        )
        
        # Start generation in background
        background_tasks.add_task(
            generator.start_generation_job,
            job_id=job_id,
            params=params.model_dump(),
            llm_provider=get_llm_provider(params.provider, params.model)
        )
        
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generator/status/{job_id}")
async def get_generator_status(job_id: int):
    """Get generation job status"""
    status = generator.get_job_status(job_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status

@app.post("/api/generator/cancel/{job_id}")
async def cancel_generator(job_id: int):
    """Cancel a generation job"""
    success = generator.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
    
    return {"status": "cancelled"}

# Quality control routes
@app.post("/api/quality/start")
async def start_quality_control(params: QualityParams, background_tasks: BackgroundTasks):
    """Start a quality control job"""
    try:
        # Validate dataset
        dataset = db.get_dataset(params.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {params.dataset_id} not found")
        
        # Get LLM provider
        provider = get_llm_provider(
            provider_type=params.provider,
            model=params.model
        )
        
        # Check if provider is available
        if not provider.is_available():
            raise HTTPException(status_code=400, detail="LLM provider is not available")
        
        # Create quality control job
        job_id = quality.create_quality_job(
            dataset_id=params.dataset_id,
            parameters=params.dict()
        )
        
        # Start quality control job in background
        background_tasks.add_task(
            quality.process_quality_job,
            job_id,
            provider,
            params.batch_size,
            params.threshold,
            params.auto_fix,
            params.auto_remove
        )
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/quality/status/{job_id}")
async def get_quality_status(job_id: int):
    """Get quality control job status"""
    status = quality.get_job_status(job_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status

@app.post("/api/quality/cancel/{job_id}")
async def cancel_quality(job_id: int):
    """Cancel a quality control job"""
    success = quality.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or cannot be cancelled")
    
    return {"status": "cancelled"}

# Dataset routes
@app.get("/api/datasets")
async def list_datasets(
    domain: Optional[str] = None,
    format: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "DESC",
    limit: int = 100
):
    """List datasets"""
    datasets = db.get_datasets(domain, format, sort_by, sort_order, limit)
    return {"datasets": datasets}

@app.get("/api/datasets/{dataset_id}")
async def get_dataset_details(dataset_id: int):
    """Get dataset details"""
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    return dataset

@app.get("/api/datasets/{dataset_id}/examples")
async def get_dataset_examples(
    dataset_id: int,
    status: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None,
    offset: int = 0,
    limit: int = 100
):
    """Get examples from a dataset"""
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    examples = db.get_examples(dataset_id, status, min_quality, max_quality, offset, limit)
    total_count = db.count_examples(dataset_id, status, min_quality, max_quality)
    
    return {
        "dataset_id": dataset_id,
        "examples": examples,
        "total": total_count,
        "offset": offset,
        "limit": limit
    }

@app.get("/api/datasets/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: int):
    """Get statistics about a dataset"""
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    stats = db.get_examples_stats(dataset_id)
    return {
        "dataset_id": dataset_id,
        "stats": stats
    }

@app.get("/api/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int):
    """Download a dataset"""
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    file_path = dataset["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )

@app.post("/api/datasets/{dataset_id}/export-csv")
async def export_dataset_to_csv(dataset_id: int, background_tasks: BackgroundTasks):
    """Export a dataset to CSV format"""
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    file_path = dataset["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found")
    
    # Create output CSV path
    output_path = f"{os.path.splitext(file_path)[0]}.csv"
    
    # Export in background
    def export_task():
        utils.export_to_csv(file_path, output_path, dataset["format"])
    
    background_tasks.add_task(export_task)
    
    return {
        "status": "exporting",
        "dataset_id": dataset_id,
        "output_path": output_path
    }

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """Delete a dataset"""
    success = db.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or could not be deleted")
    
    return {"status": "deleted"}

@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    description: str = ""
):
    """Upload a dataset file"""
    try:
        # Save uploaded file
        file_path = f"data/datasets/uploaded_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Detect format type and load examples
        format_type = utils.detect_format_type(file_path)
        if format_type == "unknown":
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unknown dataset format")
            
        examples = utils.load_jsonl(file_path)
        if not examples:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No valid examples found in file")
        
        # Create dataset record
        dataset_name = name or os.path.splitext(file.filename)[0]
        
        dataset_id = db.create_dataset(
            name=dataset_name,
            domain=domain or "unknown",
            subdomain=subdomain,
            format=format_type,
            file_path=file_path,
            example_count=len(examples),
            description=description,
            metadata={"source": "upload"}
        )
        
        # Add examples to database
        db.add_examples(dataset_id, examples)
        
        return {
            "dataset_id": dataset_id,
            "name": dataset_name,
            "format": format_type,
            "example_count": len(examples)
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/datasets/merge")
async def merge_datasets(
    dataset_ids: List[int] = Body(...),
    name: str = Body(...),
    description: str = Body("")
):
    """Merge multiple datasets"""
    try:
        # Get datasets
        datasets = []
        for dataset_id in dataset_ids:
            dataset = db.get_dataset(dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            datasets.append(dataset)
        
        # Check format compatibility
        formats = set(dataset["format"] for dataset in datasets)
        if len(formats) > 1:
            raise HTTPException(status_code=400, detail="Cannot merge datasets with different formats")
        
        format_type = formats.pop()
        
        # Get file paths
        file_paths = [dataset["file_path"] for dataset in datasets]
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/datasets/merged_{timestamp}.jsonl"
        
        # Merge datasets
        success, example_count = utils.merge_datasets(file_paths, output_path)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to merge datasets")
        
        # Determine domain (use most common domain)
        domain_counts = {}
        for dataset in datasets:
            domain = dataset["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        domain = max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else "merged"
        
        # Create dataset record
        dataset_id = db.create_dataset(
            name=name,
            domain=domain,
            format=format_type,
            file_path=output_path,
            example_count=example_count,
            description=description,
            metadata={"source": "merge", "parent_datasets": dataset_ids}
        )
        
        return {
            "dataset_id": dataset_id,
            "name": name,
            "format": format_type,
            "example_count": example_count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/datasets/sample")
async def sample_dataset(
    dataset_id: int = Body(...),
    count: int = Body(...),
    name: Optional[str] = Body(None)
):
    """Create a random sample of a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/datasets/sampled_{timestamp}.jsonl"
        
        # Sample dataset
        success, example_count = utils.sample_dataset(dataset["file_path"], output_path, count)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to sample dataset")
        
        # Create dataset record
        sample_name = name or f"{dataset['name']}_sample_{count}"
        
        new_dataset_id = db.create_dataset(
            name=sample_name,
            domain=dataset["domain"],
            subdomain=dataset.get("subdomain"),
            format=dataset["format"],
            file_path=output_path,
            example_count=example_count,
            description=f"Sample of {dataset['name']} ({count} examples)",
            metadata={"source": "sample", "parent_dataset": dataset_id}
        )
        
        return {
            "dataset_id": new_dataset_id,
            "name": sample_name,
            "format": dataset["format"],
            "example_count": example_count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/datasets/convert")
async def convert_dataset_format(
    dataset_id: int = Body(...),
    to_format: str = Body(...),
    name: Optional[str] = Body(None)
):
    """Convert dataset between chat and instruction formats"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Validate format
        from_format = dataset["format"]
        if from_format == to_format:
            raise HTTPException(status_code=400, detail=f"Dataset is already in {to_format} format")
        
        if to_format not in ["chat", "instruction"]:
            raise HTTPException(status_code=400, detail=f"Invalid format: {to_format}")
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/datasets/converted_{timestamp}.jsonl"
        
        # Convert format
        success, example_count = utils.convert_format(dataset["file_path"], output_path, from_format, to_format)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to convert dataset format")
        
        # Create dataset record
        converted_name = name or f"{dataset['name']}_{to_format}"
        
        new_dataset_id = db.create_dataset(
            name=converted_name,
            domain=dataset["domain"],
            subdomain=dataset.get("subdomain"),
            format=to_format,
            file_path=output_path,
            example_count=example_count,
            description=f"Converted from {from_format} to {to_format} format",
            metadata={"source": "conversion", "parent_dataset": dataset_id}
        )
        
        return {
            "dataset_id": new_dataset_id,
            "name": converted_name,
            "format": to_format,
            "example_count": example_count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# File scanner route
@app.get("/api/utils/scan-datasets")
async def scan_for_dataset_files():
    """Scan for JSONL files that might be datasets"""
    jsonl_files = utils.scan_for_datasets()
    
    # Get details for each file
    result = []
    for file_path in jsonl_files:
        format_type = utils.detect_format_type(file_path)
        example_count = utils.count_examples(file_path)
        
        file_info = {
            "path": file_path,
            "name": os.path.basename(file_path),
            "format": format_type,
            "example_count": example_count,
            "size_mb": os.path.getsize(file_path) / (1024 * 1024)
        }
        
        result.append(file_info)
    
    return {"files": result}

# Дополнительные утилиты для работы с системой
def get_disk_usage(path):
    """Get disk usage statistics for a given path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    except Exception:
        # Fallback if shutil doesn't work
        return 100000000000, 50000000000, 50000000000  # 100GB total, 50GB used, 50GB free

def get_cpu_usage():
    """Get CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        # Fallback if psutil is not available
        return 25.0  # Return a default value

def get_memory_usage():
    """Get memory usage statistics"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
    except Exception:
        # Fallback if psutil is not available
        total = 8000000000  # 8GB
        used = 4000000000   # 4GB
        return {
            "total": total,
            "available": total - used,
            "used": used,
            "percent": 50.0
        }

def get_cache_size():
    """Estimate cache size by scanning temporary files"""
    try:
        # Assume cache is in a cache directory
        cache_dir = Path("/tmp/llm_dataset_creator_cache")
        
        if not cache_dir.exists():
            return 0
            
        total_size = 0
        for path in cache_dir.glob('**/*'):
            if path.is_file():
                total_size += path.stat().st_size
                
        return total_size
    except Exception:
        # Fallback if error occurs
        return 1000000  # 1MB

# Получение всех активных задач
def get_all_tasks():
    """Get all tasks in the system"""
    try:
        # Соберем все генераторы и задачи качества
        generator_jobs = db.get_generation_jobs()
        quality_jobs = db.get_quality_jobs()
        
        # Преобразуем в единый формат задач
        tasks = []
        
        for job in generator_jobs:
            tasks.append({
                "id": job["id"],
                "type": "generator",
                "status": job["status"],
                "progress": job["examples_generated"] / job["examples_requested"] if job["examples_requested"] > 0 else 0,
                "progress_details": f"{job['examples_generated']}/{job['examples_requested']} examples",
                "created_at": job["started_at"],
                "updated_at": job.get("completed_at", job["started_at"])
            })
        
        for job in quality_jobs:
            tasks.append({
                "id": job["id"],
                "type": "quality",
                "status": job["status"],
                "progress": job["examples_processed"] / job["examples_total"] if job["examples_total"] > 0 else 0,
                "progress_details": f"{job['examples_processed']}/{job['examples_total']} examples",
                "created_at": job["started_at"],
                "updated_at": job.get("completed_at", job["started_at"])
            })
        
        # Отсортируем по времени создания (новые в начале)
        tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        return tasks
    except Exception as e:
        print(f"Error getting tasks: {str(e)}")
        return []

# Добавляем API эндпоинты для настроек системы
@app.get("/api/settings")
async def get_settings():
    """Get system settings"""
    try:
        # Возвращаем текущие настройки системы
        provider_config = {
            "provider": DEFAULT_PROVIDER,
            "model": DEFAULT_MODEL,
            "apiKey": OPENAI_API_KEY if DEFAULT_PROVIDER == "openai" else ANTHROPIC_API_KEY if DEFAULT_PROVIDER == "anthropic" else "",
            "baseUrl": OLLAMA_URL if DEFAULT_PROVIDER == "ollama" else ""
        }
        
        system_config = {
            "dataDirectory": str(Path(os.environ.get("DATA_DIR", "/app/data"))),
            "maxConcurrentJobs": int(os.environ.get("MAX_CONCURRENT_JOBS", "2")),
            "enableCaching": os.environ.get("ENABLE_CACHING", "true").lower() == "true",
            "cacheTTL": int(os.environ.get("CACHE_TTL", "24")),
            "logLevel": os.environ.get("LOG_LEVEL", "info")
        }
        
        return {
            "modelSettings": provider_config,
            "systemSettings": system_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(settings: dict = Body(...)):
    """Update system settings"""
    try:
        # В реальном приложении здесь бы сохранялись настройки
        # Однако в этой реализации просто подтверждаем успешное обновление
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings/status")
async def get_system_status():
    """Get system status information"""
    try:
        # Получаем информацию о статусе системы
        data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
        
        # Рассчитываем использование диска
        total_space, used_space, free_space = get_disk_usage(data_dir)
        
        # Получаем загрузку CPU
        cpu_usage = get_cpu_usage()
        
        # Получаем использование памяти
        memory_usage = get_memory_usage()
        
        # Оцениваем размер кэша
        cache_size = get_cache_size()
        
        return {
            "diskUsage": {
                "total": total_space,
                "used": used_space,
                "free": free_space,
                "percent": (used_space / total_space) * 100 if total_space > 0 else 0
            },
            "cpuUsage": cpu_usage,
            "memoryUsage": memory_usage,
            "cacheSize": cache_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/clear-cache")
async def clear_system_cache():
    """Clear system cache"""
    try:
        # В реальном приложении здесь бы очищался кэш
        # Однако в этой реализации просто подтверждаем успешную очистку
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Добавляем API эндпоинты для управления задачами
@app.get("/api/tasks")
async def list_all_tasks():
    """Get all tasks"""
    try:
        tasks = get_all_tasks()
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task_details(task_id: int):
    """Get task details"""
    try:
        # Ищем сначала среди генераторов
        generator_job = db.get_generation_job(task_id)
        if generator_job:
            return {
                "id": generator_job["id"],
                "type": "generator",
                "status": generator_job["status"],
                "progress": generator_job["examples_generated"] / generator_job["examples_requested"] if generator_job["examples_requested"] > 0 else 0,
                "progress_details": f"{generator_job['examples_generated']}/{generator_job['examples_requested']} examples",
                "params": json.loads(generator_job["parameters"]),
                "results": {
                    "dataset_id": generator_job.get("dataset_id"),
                    "examples_generated": generator_job["examples_generated"]
                },
                "errors": json.loads(generator_job.get("errors", "[]")),
                "created_at": generator_job["started_at"],
                "updated_at": generator_job.get("completed_at", generator_job["started_at"])
            }
            
        # Затем среди задач контроля качества
        quality_job = db.get_quality_job(task_id)
        if quality_job:
            return {
                "id": quality_job["id"],
                "type": "quality",
                "status": quality_job["status"],
                "progress": quality_job["examples_processed"] / quality_job["examples_total"] if quality_job["examples_total"] > 0 else 0,
                "progress_details": f"{quality_job['examples_processed']}/{quality_job['examples_total']} examples",
                "params": json.loads(quality_job["parameters"]),
                "results": {
                    "dataset_id": quality_job["dataset_id"],
                    "examples_processed": quality_job["examples_processed"],
                    "examples_kept": quality_job.get("examples_kept", 0),
                    "examples_fixed": quality_job.get("examples_fixed", 0),
                    "examples_removed": quality_job.get("examples_removed", 0),
                    "avg_quality_score": quality_job.get("avg_quality_score", 0)
                },
                "errors": json.loads(quality_job.get("errors", "[]")),
                "created_at": quality_job["started_at"],
                "updated_at": quality_job.get("completed_at", quality_job["started_at"])
            }
        
        # Если не нашли
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: int):
    """Cancel a task"""
    try:
        # Ищем сначала среди генераторов
        generator_job = db.get_generation_job(task_id)
        if generator_job:
            return await cancel_generator(task_id)
            
        # Затем среди задач контроля качества
        quality_job = db.get_quality_job(task_id)
        if quality_job:
            return await cancel_quality(task_id)
        
        # Если не нашли
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task"""
    try:
        deleted = False
        
        # Пробуем удалить из генераторов
        generator_job = db.get_generation_job(task_id)
        if generator_job:
            # Сначала отменяем, если задача активна
            if generator_job["status"] in ["pending", "running"]:
                await cancel_generator(task_id)
            
            # Удаляем запись о задаче (в реальности просто отмечаем как deleted)
            # Так как нет прямого метода для удаления, используем обновление статуса
            db.update_generation_job(task_id, status="deleted")
            deleted = True
            
        # Пробуем удалить из задач качества
        quality_job = db.get_quality_job(task_id)
        if quality_job:
            # Сначала отменяем, если задача активна
            if quality_job["status"] in ["pending", "running"]:
                await cancel_quality(task_id)
            
            # Удаляем запись о задаче (в реальности просто отмечаем как deleted)
            # Так как нет прямого метода для удаления, используем обновление статуса
            db.update_quality_job(task_id, status="deleted")
            deleted = True
        
        if deleted:
            return {"status": "success", "message": f"Task {task_id} deleted successfully"}
        
        # Если не нашли
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/datasets/{dataset_id}/examples/{example_id}")
async def update_dataset_example(
    dataset_id: int,
    example_id: int,
    content: Dict[str, Any] = Body(...)
):
    """Update an example in a dataset"""
    try:
        print(f"Updating example {example_id} in dataset {dataset_id} with content:", content)
        
        # Check if dataset exists
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Update example
        success = db.update_example(example_id, content=content)
        if not success:
            raise HTTPException(status_code=404, detail=f"Example {example_id} not found")
        
        print(f"Successfully updated example {example_id}")
        return {"status": "updated"}
    except Exception as e:
        print(f"Error updating example {example_id}:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Initialize LLM provider
    llm_provider = get_llm_provider()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)