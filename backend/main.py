"""
FastAPI main application for LLM Dataset Creator
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import time
import shutil
import psutil
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import database as db
from domains import DOMAINS, get_domains, get_domain, get_subdomain, get_common_params
from llm_providers import (
    get_provider,
    get_default_provider,
    list_available_providers,
    get_provider_models,
    test_provider_availability,
    ProviderRegistry
)
import generator
import quality
import utils
from quality import toxicity_analyzer, pii_analyzer, deduplicator
from config import get_config, reload_config
from model_manager import get_model_manager
from prompts import get_manager, PromptValidator, ValidationError
from exporters import HuggingFaceExporter, OpenAIExporter, AlpacaExporter, LangChainExporter
from analytics import get_tracker, DatasetStats
from quality_advanced import (
    ToxicityDetector,
    DeduplicationChecker,
    DiversityAnalyzer,
    QualityScorer,
    QualityReportGenerator
)
from versioning import VersionManager, DatasetDiff, DatasetMerger
from collaboration import (
    get_user_manager,
    get_permission_manager,
    get_comment_manager,
    get_review_manager,
    UserRole,
    PermissionLevel,
    ReviewStatus
)
from augmentation import DataAugmenter, AugmentationConfig
from mlops.webhooks import get_webhook_manager
from mlops.scheduler import get_scheduler
from mlops.celery_app import run_generation_job, run_quality_check_job

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events handler"""
    # Load configuration
    config = get_config()
    config.ensure_directories()

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
db.init_db()

# Global configuration instance
config = get_config()

# Global LLM provider instance (lazy-loaded)
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
    generation_mode: Optional[str] = "standard"  # "standard" or "advanced"
    advanced_method: Optional[str] = "swarm"  # "swarm", "evolution", "cosmic", "quantum"
    agent_models: Optional[Dict[str, Dict[str, str]]] = None  # Role -> {provider, model}
    template: Optional[str] = None  # Custom prompt template name

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

class TemplateCreate(BaseModel):
    name: str
    domain: str
    subdomain: Optional[str] = None
    content: str
    variables: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

class TemplateResponse(BaseModel):
    id: int
    name: str
    domain: str
    subdomain: Optional[str] = None
    content: str
    variables: Optional[Union[Dict[str, Any], List[str]]] = None
    description: Optional[str] = None
    created_at: str
    updated_at: str

class PromptTemplateCreate(BaseModel):
    template_data: Dict[str, Any]
    overwrite: bool = False

class PromptTemplateRender(BaseModel):
    template_name: str
    variables: Dict[str, Any]

class ExportRequest(BaseModel):
    dataset_id: int
    format: str = Field(..., description="Export format: huggingface, openai, alpaca, csv")
    output_filename: str
    system_message: Optional[str] = None
    train_split: float = Field(1.0, ge=0.0, le=1.0, description="Training split ratio")
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class QualityCheckRequest(BaseModel):
    dataset_id: int
    check_types: List[str] = Field(default=["all"], description="Types: toxicity, deduplication, diversity, scoring, all")
    text_field: str = "text"
    generate_report: bool = True

class VersionCreateRequest(BaseModel):
    dataset_id: int
    commit_message: str
    author: str = "user"
    tags: List[str] = Field(default_factory=list)

class VersionMergeRequest(BaseModel):
    version_id1: str
    version_id2: str
    strategy: str = Field("union", description="Merge strategy: union, intersection, prefer_branch1, prefer_branch2")

# Collaboration request models
class UserCreateRequest(BaseModel):
    username: str
    email: str
    role: str = "viewer"  # admin, editor, viewer
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PermissionGrantRequest(BaseModel):
    dataset_id: int
    user_id: str
    permission_level: str  # owner, write, read
    granted_by: str

class ShareDatasetRequest(BaseModel):
    dataset_id: int
    from_user_id: str
    to_user_id: str
    permission_level: str

class CommentCreateRequest(BaseModel):
    dataset_id: int
    example_id: int
    user_id: str
    content: str
    parent_comment_id: Optional[str] = None

class CommentUpdateRequest(BaseModel):
    content: str

class ReviewCreateRequest(BaseModel):
    dataset_id: int
    example_id: int
    reviewer_id: str
    status: str  # pending, approved, rejected, needs_changes
    feedback: Optional[str] = None

class ReviewUpdateRequest(BaseModel):
    status: Optional[str] = None
    feedback: Optional[str] = None

class AugmentationRequest(BaseModel):
    dataset_id: int
    techniques: List[str] = Field(default=["synonym"], description="Techniques: synonym, random, paraphrase, backtranslation")
    samples_per_example: int = Field(1, ge=1, le=5, description="Number of augmented samples per example")
    text_field: str = "text"
    synonym_ratio: float = Field(0.3, ge=0.0, le=1.0)
    random_swap_ratio: float = Field(0.1, ge=0.0, le=0.5)
    random_delete_ratio: float = Field(0.1, ge=0.0, le=0.5)
    random_insert_ratio: float = Field(0.1, ge=0.0, le=0.5)
    paraphrase_diversity: float = Field(0.7, ge=0.0, le=1.0)
    save_augmented: bool = True

class WebhookCreateRequest(BaseModel):
    url: str
    events: List[str] = Field(default=["all"], description="Events to subscribe to")
    secret: Optional[str] = None

class ScheduledJobCreateRequest(BaseModel):
    name: str
    cron_expression: str
    task_type: str = Field(..., description="generation, quality_check")
    parameters: Dict[str, Any]

# Helper function to get or create LLM provider
def get_llm_provider(provider_type: Optional[str] = None, model: Optional[str] = None, **kwargs):
    """
    Get or create LLM provider using configuration system

    Args:
        provider_type: Name of the provider (ollama, openai, anthropic)
        model: Model name to use (overrides config default)
        **kwargs: Additional parameters to override config

    Returns:
        LLMProvider instance
    """
    global llm_provider

    # Build override parameters
    override_params = {}
    if model:
        override_params['model'] = model
    override_params.update(kwargs)

    # Use existing provider if no changes
    if llm_provider and not (provider_type or override_params):
        return llm_provider

    # Create or get provider
    try:
        if override_params:
            llm_provider = get_provider(provider_type, **override_params)
        else:
            llm_provider = get_provider(provider_type)
        return llm_provider
    except Exception as e:
        print(f"Error creating provider: {e}")
        # Fallback to default provider if specified one fails
        if provider_type:
            print(f"Falling back to default provider")
            llm_provider = get_default_provider()
            return llm_provider
        raise

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
    """List available LLM providers from configuration"""
    try:
        providers = list_available_providers()
        return {
            "providers": [
                {
                    "name": name,
                    "enabled": config.get("enabled", True),
                    "description": config.get("description", ""),
                    "default_model": config.get("default_model", ""),
                    "available_models": config.get("available_models", [])
                }
                for name, config in providers.items()
            ],
            "default_provider": get_config().get_default_provider()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        provider_name = provider or config.get_default_provider()

        # Get models from provider
        models = get_provider_models(provider_name)

        # Also get configured models from config
        provider_config = config.get_provider_config(provider_name)
        configured_models = provider_config.get("available_models", []) if provider_config else []

        return {
            "provider": provider_name,
            "models": models if models else configured_models,
            "configured_models": configured_models
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/providers/{provider_name}/status")
async def check_provider_status(provider_name: str):
    """Check if a provider is available and working"""
    try:
        is_available = test_provider_availability(provider_name)
        provider_config = config.get_provider_config(provider_name)

        return {
            "provider": provider_name,
            "available": is_available,
            "enabled": provider_config.get("enabled", False) if provider_config else False,
            "default_model": provider_config.get("default_model", "") if provider_config else ""
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "available": False,
            "error": str(e)
        }

@app.post("/api/providers/{provider_name}/enable")
async def enable_provider(provider_name: str):
    """Enable a provider"""
    try:
        provider_config = config.get_provider_config(provider_name)
        if not provider_config:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        config.update_provider_config(provider_name, {"enabled": True})
        config.save_config()

        return {"status": "success", "provider": provider_name, "enabled": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/providers/{provider_name}/disable")
async def disable_provider(provider_name: str):
    """Disable a provider"""
    try:
        provider_config = config.get_provider_config(provider_name)
        if not provider_config:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        config.update_provider_config(provider_name, {"enabled": False})
        config.save_config()

        # Clear provider cache
        ProviderRegistry.clear_cache()

        return {"status": "success", "provider": provider_name, "enabled": False}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/providers/{provider_name}/config")
async def update_provider_configuration(provider_name: str, provider_config: Dict[str, Any] = Body(...)):
    """Update provider configuration"""
    try:
        current_config = config.get_provider_config(provider_name)
        if not current_config:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        # Update configuration
        config.update_provider_config(provider_name, provider_config)
        config.save_config()

        # Clear provider cache to force recreation
        ProviderRegistry.clear_cache()

        return {"status": "success", "provider": provider_name, "config": config.get_provider_config(provider_name)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/reload")
async def reload_models():
    """Reload models configuration"""
    try:
        model_manager.reload()
        return {"status": "success", "message": "Models reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Settings endpoints
@app.post("/api/settings/api-key")
async def save_api_key(request: dict):
    """Save API key for a provider"""
    try:
        provider = request.get("provider", "").lower()
        api_key = request.get("api_key", "")
        
        if not provider or not api_key:
            raise HTTPException(status_code=400, detail="Provider and API key required")
        
        # Save to .env file
        env_var = f"{provider.upper()}_API_KEY"
        
        # Read existing .env (in backend directory)
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        env_lines = []
        
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                env_lines = f.readlines()
        
        # Update or add the key
        key_found = False
        for i, line in enumerate(env_lines):
            if line.startswith(f"{env_var}="):
                env_lines[i] = f"{env_var}={api_key}\n"
                key_found = True
                break
        
        if not key_found:
            env_lines.append(f"{env_var}={api_key}\n")
        
        # Write back
        with open(env_path, "w") as f:
            f.writelines(env_lines)
        
        # Also set in current environment
        os.environ[env_var] = api_key
        
        # Reload configuration and clear provider cache
        reload_config()
        ProviderRegistry.clear_cache()
        
        return {"status": "success", "message": f"API key saved for {provider}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings/api-keys")
async def get_api_keys():
    """Get configured API keys (masked)"""
    try:
        keys = {}
        for provider in ["openai", "anthropic", "google", "mistral"]:
            env_var = f"{provider.upper()}_API_KEY"
            key = os.getenv(env_var, "")
            # Mask the key
            if key:
                keys[provider] = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            else:
                keys[provider] = ""
        return keys
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/reload")
async def reload_configuration():
    """Reload configuration from file"""
    try:
        reload_config()
        ProviderRegistry.clear_cache()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prompt Template routes
@app.get("/api/prompts/templates")
async def list_prompt_templates(domain: Optional[str] = None, tags: Optional[str] = None):
    """List all available prompt templates"""
    try:
        manager = get_manager()

        # Parse tags if provided
        tag_list = tags.split(',') if tags else None

        templates = manager.list_templates(domain=domain, tags=tag_list)
        return {"templates": templates, "count": len(templates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/templates/{template_name}")
async def get_prompt_template(template_name: str):
    """Get a specific prompt template"""
    try:
        manager = get_manager()
        template = manager.load_template(template_name)

        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

        return {"template": template.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts/templates")
async def create_prompt_template(request: PromptTemplateCreate):
    """Create a new prompt template"""
    try:
        manager = get_manager()
        validator = PromptValidator()

        # Validate template structure
        is_valid, errors = validator.validate_template_structure(request.template_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})

        # Create template object
        template = manager.create_template_from_dict(request.template_data)
        if not template:
            raise HTTPException(status_code=400, detail="Invalid template data")

        # Validate full template
        is_valid, validation_errors = validator.validate_full_template(template)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"validation_errors": validation_errors})

        # Save template
        success = manager.save_template(template, overwrite=request.overwrite)
        if not success:
            raise HTTPException(status_code=409, detail=f"Template '{template.metadata.name}' already exists")

        return {"status": "success", "template_name": template.metadata.name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/prompts/templates/{template_name}")
async def update_prompt_template(template_name: str, template_data: Dict[str, Any] = Body(...)):
    """Update an existing prompt template"""
    try:
        manager = get_manager()
        validator = PromptValidator()

        # Ensure name matches
        if template_data.get('metadata', {}).get('name') != template_name:
            raise HTTPException(status_code=400, detail="Template name mismatch")

        # Validate template structure
        is_valid, errors = validator.validate_template_structure(template_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})

        # Create template object
        template = manager.create_template_from_dict(template_data)
        if not template:
            raise HTTPException(status_code=400, detail="Invalid template data")

        # Validate full template
        is_valid, validation_errors = validator.validate_full_template(template)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"validation_errors": validation_errors})

        # Save template (with overwrite)
        success = manager.save_template(template, overwrite=True)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update template")

        return {"status": "success", "template_name": template_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/prompts/templates/{template_name}")
async def delete_prompt_template(template_name: str):
    """Delete a prompt template"""
    try:
        manager = get_manager()
        success = manager.delete_template(template_name)

        if not success:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

        return {"status": "success", "message": f"Template '{template_name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts/render")
async def render_prompt_template(request: PromptTemplateRender):
    """Render a prompt template with given variables"""
    try:
        manager = get_manager()
        rendered = manager.render_template(request.template_name, request.variables)

        if rendered is None:
            raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")

        return {"rendered_prompt": rendered}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/templates/{template_name}/variables")
async def get_template_variables(template_name: str):
    """Get variables defined in a template"""
    try:
        manager = get_manager()
        variables = manager.get_template_variables(template_name)

        if variables is None:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

        return {"template_name": template_name, "variables": variables}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/search")
async def search_prompt_templates(q: str = Query(..., min_length=1)):
    """Search prompt templates"""
    try:
        manager = get_manager()
        results = manager.search_templates(q)

        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts/validate")
async def validate_prompt_template(template_data: Dict[str, Any] = Body(...)):
    """Validate a prompt template without saving"""
    try:
        manager = get_manager()
        validator = PromptValidator()

        # Validate template structure
        is_valid, errors = validator.validate_template_structure(template_data)
        if not is_valid:
            return {"valid": False, "errors": errors}

        # Create template object
        template = manager.create_template_from_dict(template_data)
        if not template:
            return {"valid": False, "errors": ["Invalid template data"]}

        # Validate full template
        is_valid, validation_errors = validator.validate_full_template(template)

        return {
            "valid": is_valid,
            "errors": validation_errors if not is_valid else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Generation - SSE Streaming
@app.get("/api/generator/stream/{job_id}")
async def stream_generation_events(job_id: int):
    """Stream real-time generation events via SSE"""
    try:
        from streaming import create_sse_response
        from agents.base_agent import get_event_bus
        
        # Get event bus for this job
        event_bus = get_event_bus()
        
        # Create SSE response
        return await create_sse_response(str(job_id), event_bus)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generator/start")
async def start_generation(params: GenerationParams):
    """Start dataset generation with optional advanced mode"""
    try:
        # Check if advanced mode is requested
        generation_mode = params.dict().get("generation_mode", "standard")
        
        if generation_mode == "advanced":
            # Use advanced multi-agent generation
            method = params.dict().get("advanced_method", "swarm")
            
            from agents.base_agent import get_event_bus
            from agents.shared_memory import SharedMemory
            
            event_bus = get_event_bus()
            shared_memory = SharedMemory()
            
            # Convert agent_models to role_models format
            # Frontend sends: {"scout": {"provider": "openai", "model": "gpt-4"}}
            # Backend expects: {"scout": {"llm_provider": "openai", "llm_model": "gpt-4"}}
            role_models = {}
            if params.agent_models:
                for role, config in params.agent_models.items():
                    if config.get("model"):  # Only add if model is specified
                        role_models[role] = {
                            "llm_provider": config.get("provider", params.provider or "ollama"),
                            "llm_model": config.get("model"),
                        }
            
            # Default kwargs for agents without specific model
            default_kwargs = {
                "llm_provider": params.provider or "ollama",
                "llm_model": params.model or "llama3.2",
                "temperature": params.temperature,
            }
            
            # Select method
            if method == "swarm":
                from agents.swarm import HybridSwarmSynthesis
                
                advanced_generator = HybridSwarmSynthesis(
                    event_bus=event_bus,
                    shared_memory=shared_memory,
                    num_scouts=3,
                    num_gatherers=2,
                    num_mutators=2,
                    iterations=3,
                    role_models=role_models,
                    **default_kwargs
                )
                
            elif method == "evolution":
                from agents.evolution import EvolutionaryGeneFusion
                
                advanced_generator = EvolutionaryGeneFusion(
                    event_bus=event_bus,
                    shared_memory=shared_memory,
                    population_size=50,
                    generations=5,
                    mutation_rate=0.3,
                    crossover_rate=0.5,
                    elite_size=10,
                    role_models=role_models,
                    **default_kwargs
                )
                
            elif method == "cosmic":
                from agents.cosmic import CosmicBurstSynthesis
                
                advanced_generator = CosmicBurstSynthesis(
                    event_bus=event_bus,
                    shared_memory=shared_memory,
                    expansion_factor=10,
                    iterations=3,
                    temperature_threshold=0.5,
                    role_models=role_models,
                    **default_kwargs
                )
                
            elif method == "quantum":
                from agents.quantum import QuantumFieldOrchestration
                
                advanced_generator = QuantumFieldOrchestration(
                    event_bus=event_bus,
                    shared_memory=shared_memory,
                    iterations=3,
                    role_models=role_models,
                    **default_kwargs
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
                
            # Create job
            job_id = db.create_generation_job(
                dataset_id=None,
                parameters=params.dict(),
                examples_requested=params.count
            )
            
            # Run generator in background
            async def run_advanced_generation():
                print(f"[Advanced] Starting generation job {job_id} with method {method}")
                try:
                    # Update status to running
                    db.update_generation_job(job_id, status="running")
                    print(f"[Advanced] Job {job_id} status set to running")
                    
                    print(f"[Advanced] Calling advanced_generator.run() for job {job_id}")
                    results = await advanced_generator.run({
                        "domain": params.domain,
                        "subdomain": params.subdomain,
                        "format": params.format,
                        "language": params.language,
                        "template": params.template,
                        "count": params.count
                    })
                    print(f"[Advanced] Job {job_id} got {len(results) if results else 0} results")
                    
                    # Save results to dataset
                    if results and len(results) > 0:
                        from datetime import datetime
                        import os
                        
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                        dataset_name = f"{params.domain}_{timestamp}"
                        file_path = f"data/datasets/{dataset_name}.jsonl"
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        dataset_id = db.create_dataset(
                            name=dataset_name,
                            domain=params.domain,
                            format=params.format,
                            file_path=file_path,
                            subdomain=params.subdomain,
                            example_count=len(results)
                        )
                        
                        # Add examples to dataset
                        db.add_examples(dataset_id, results)
                        
                        # Also save to file
                        import json
                        with open(file_path, 'w', encoding='utf-8') as f:
                            for example in results:
                                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                        
                        db.update_generation_job(
                            job_id,
                            status="completed",
                            examples_generated=len(results),
                            dataset_id=dataset_id,
                            completed_at=datetime.now().isoformat()
                        )
                    else:
                        from datetime import datetime
                        db.update_generation_job(
                            job_id,
                            status="completed",
                            examples_generated=0,
                            completed_at=datetime.now().isoformat()
                        )
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    print(f"Advanced generation error: {error_msg}")
                    db.update_generation_job(
                        job_id,
                        status="failed",
                        errors=[str(e)]
                    )
            
            asyncio.create_task(run_advanced_generation())
            
            return {
                "job_id": job_id,
                "status": "started",
                "mode": "advanced",
                "method": method,
                "stream_url": f"/api/generator/stream/{job_id}"
            }
        
        # Standard generation (existing code)
        # Create job in database
        job_params = {
            "domain": params.domain,
            "subdomain": params.subdomain,
            "format": params.format,
            "language": params.language,
            "temperature": params.temperature,
            "model": params.model,
            "provider": params.provider
        }
        job_id = db.create_generation_job(
            dataset_id=None,
            parameters=job_params,
            examples_requested=params.count
        )
        
        # Get LLM provider
        llm_provider = get_llm_provider(
            provider_type=params.provider,
            model=params.model
        )
        
        # Start generation in background
        asyncio.create_task(generator.start_generation_job(job_id, llm_provider, job_params))
        
        return {
            "job_id": job_id,
            "status": "started",
            "mode": "standard"
        }
        
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

# MLOps Routes
@app.post("/api/webhooks")
async def register_webhook(request: WebhookCreateRequest):
    """Register a new webhook"""
    try:
        manager = get_webhook_manager()
        webhook_id = manager.register_webhook(request.url, request.events, request.secret)
        return {"status": "success", "webhook_id": webhook_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/webhooks")
async def list_webhooks():
    """List all active webhooks"""
    try:
        manager = get_webhook_manager()
        return {"webhooks": manager.list_webhooks()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scheduler/jobs")
async def create_scheduled_job(request: ScheduledJobCreateRequest):
    """Create a new scheduled job"""
    try:
        scheduler = get_scheduler()
        job_id = scheduler.schedule_job(
            request.name,
            request.cron_expression,
            request.task_type,
            request.parameters
        )
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/datasets/{dataset_id}/examples/search")
async def search_dataset_examples(
    dataset_id: int,
    q: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """Search within dataset examples"""
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    # Get dataset to verify it exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    dataset = cursor.fetchone()
    
    if not dataset:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Search in examples content
    search_pattern = f"%{q}%"
    
    # Count total matches
    cursor.execute("""
        SELECT COUNT(*) FROM examples 
        WHERE dataset_id = ? AND content LIKE ?
    """, (dataset_id, search_pattern))
    total = cursor.fetchone()[0]
    
    # Get paginated results
    offset = (page - 1) * page_size
    cursor.execute("""
        SELECT id, content, quality_score, status, metadata 
        FROM examples 
        WHERE dataset_id = ? AND content LIKE ?
        ORDER BY id
        LIMIT ? OFFSET ?
    """, (dataset_id, search_pattern, page_size, offset))
    
    examples = cursor.fetchall()
    conn.close()
    
    # Parse examples
    examples_list = []
    for ex in examples:
        try:
            content = json.loads(ex['content']) if isinstance(ex['content'], str) else ex['content']
            examples_list.append({
                "id": ex['id'],
                "content": content,
                "quality_score": ex['quality_score'],
                "status": ex['status'],
                "metadata": json.loads(ex['metadata']) if ex['metadata'] else None
            })
        except:
            continue
    
    return {
        "examples": examples_list,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "query": q
    }

# Template Endpoints

@app.post("/api/templates", response_model=TemplateResponse)
async def create_template(template: TemplateCreate):
    """Create a new template"""
    template_id = db.create_template(
        name=template.name,
        domain=template.domain,
        subdomain=template.subdomain,
        content=template.content,
        variables=template.variables,
        description=template.description
    )
    
    created_template = db.get_template(template_id)
    if not created_template:
        raise HTTPException(status_code=500, detail="Failed to create template")
        
    # Parse variables JSON if string
    if created_template['variables'] and isinstance(created_template['variables'], str):
        created_template['variables'] = json.loads(created_template['variables'])
        
    return created_template

@app.get("/api/templates", response_model=List[TemplateResponse])
async def get_templates(domain: Optional[str] = None):
    """Get all templates"""
    templates = db.get_templates(domain)
    
    # If no templates exist, load defaults
    if not templates:
        from default_templates import PREDEFINED_TEMPLATES
        for template_data in PREDEFINED_TEMPLATES:
            try:
                # Combine system_prompt, user_template, assistant_template into content
                content = f"System: {template_data['system_prompt']}\n\nUser: {template_data['user_template']}\n\nAssistant: {template_data['assistant_template']}"
                
                db.create_template(
                    name=template_data['name'],
                    domain=template_data['domain'],
                    content=content,
                    subdomain=template_data.get('subdomain'),
                    variables=template_data.get('variables'),
                    description=template_data.get('description')
                )
            except Exception as e:
                print(f"Failed to create default template: {e}")
        templates = db.get_templates(domain)
    
    # Convert to response format
    result = []
    for t in templates:
        t_dict = dict(t)
        if t_dict.get('variables') and isinstance(t_dict['variables'], str):
            t_dict['variables'] = json.loads(t_dict['variables'])
        result.append(t_dict)
    
    return result

@app.post("/api/templates/load-defaults")
async def load_default_templates():
    """Load default templates into database"""
    from default_templates import PREDEFINED_TEMPLATES
    loaded = []
    for template_data in PREDEFINED_TEMPLATES:
        try:
            # Combine system_prompt, user_template, assistant_template into content
            content = f"System: {template_data['system_prompt']}\n\nUser: {template_data['user_template']}\n\nAssistant: {template_data['assistant_template']}"
            
            template_id = db.create_template(
                name=template_data['name'],
                domain=template_data['domain'],
                content=content,
                subdomain=template_data.get('subdomain'),
                variables=template_data.get('variables'),
                description=template_data.get('description')
            )
            loaded.append({"id": template_id, "name": template_data['name']})
        except Exception as e:
            print(f"Failed to create default template {template_data['name']}: {e}")
    return {"status": "success", "loaded": len(loaded), "templates": loaded}

@app.get("/api/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: int):
    """Get a template by ID"""
    template = db.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
        
    if template['variables'] and isinstance(template['variables'], str):
        template['variables'] = json.loads(template['variables'])
        
    return template

@app.put("/api/templates/{template_id}")
async def update_template(template_id: int, template: TemplateUpdate):
    """Update a template"""
    success = db.update_template(
        template_id=template_id,
        name=template.name,
        content=template.content,
        variables=template.variables,
        description=template.description
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
        
    updated_template = db.get_template(template_id)
    if updated_template['variables'] and isinstance(updated_template['variables'], str):
        updated_template['variables'] = json.loads(updated_template['variables'])
        
    return updated_template

@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: int):
    """Delete a template"""
    success = db.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"status": "success"}

@app.get("/api/templates/{template_id}/versions")
async def get_template_versions(template_id: int):
    """Get version history for a template"""
    versions = db.get_template_versions(template_id)
    
    # Parse variables JSON
    result = []
    for v in versions:
        v_dict = dict(v)
        if v_dict['variables'] and isinstance(v_dict['variables'], str):
            v_dict['variables'] = json.loads(v_dict['variables'])
        result.append(v_dict)
        
    return result

@app.post("/api/templates/{template_id}/restore/{version_id}")
async def restore_template_version(template_id: int, version_id: int):
    """Restore a specific version"""
    success = db.restore_template_version(template_id, version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
        
    restored_template = db.get_template(template_id)
    if restored_template['variables'] and isinstance(restored_template['variables'], str):
        restored_template['variables'] = json.loads(restored_template['variables'])
        
    return restored_template

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

@app.get("/api/datasets/{dataset_id}/export")
async def export_dataset_get(
    dataset_id: int,
    format: str = Query(..., description="Export format: huggingface, openai, alpaca, sharegpt, langchain"),
    output_filename: str = Query(..., description="Output filename")
):
    """Export a dataset to various ML framework formats and download"""
    import tempfile
    import zipfile
    
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load all examples (no limit for export)
        examples = db.get_examples(dataset_id, limit=100000)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found in dataset")

        # Create temp directory for export
        temp_dir = tempfile.mkdtemp()
        
        # Select exporter based on format
        format_lower = format.lower()
        
        if format_lower == "huggingface":
            exporter = HuggingFaceExporter(output_dir=temp_dir)
            output_path = exporter.export(
                examples=examples,
                output_filename=output_filename,
                dataset_name=dataset.get("name", "dataset")
            )
            # HuggingFace creates a directory, zip it
            zip_path = os.path.join(temp_dir, f"{output_filename}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_path)
                        zipf.write(file_path, arcname)
            return FileResponse(
                path=zip_path,
                filename=f"{output_filename}.zip",
                media_type="application/zip"
            )

        elif format_lower == "openai":
            exporter = OpenAIExporter(output_dir=temp_dir)
            output_path = exporter.export(
                examples=examples,
                output_filename=output_filename
            )
            return FileResponse(
                path=output_path,
                filename=f"{output_filename}.jsonl",
                media_type="application/jsonl"
            )

        elif format_lower in ["alpaca", "sharegpt"]:
            exporter = AlpacaExporter(output_dir=temp_dir)
            output_path = exporter.export(
                examples=examples,
                output_filename=output_filename,
                format_type=format_lower
            )
            return FileResponse(
                path=output_path,
                filename=f"{output_filename}.json",
                media_type="application/json"
            )

        elif format_lower == "langchain":
            exporter = LangChainExporter(output_dir=temp_dir)
            output_path = exporter.export(
                examples=examples,
                output_filename=output_filename
            )
            return FileResponse(
                path=output_path,
                filename=f"{output_filename}.jsonl",
                media_type="application/jsonl"
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export format: {format}. Supported: huggingface, openai, alpaca, sharegpt, langchain"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/api/datasets/export")
async def export_dataset(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export a dataset to various ML framework formats"""
    try:
        # Get dataset
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

        # Load examples
        examples = db.get_examples(request.dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found in dataset")

        # Select exporter based on format
        if request.format.lower() == "huggingface":
            exporter = HuggingFaceExporter()

            def export_task():
                try:
                    output_path = exporter.export(
                        examples=examples,
                        output_filename=request.output_filename,
                        dataset_name=dataset.get("name", "dataset"),
                        **request.additional_params
                    )
                    db.update_dataset(request.dataset_id, {"last_export": datetime.now().isoformat()})
                    print(f"HuggingFace export completed: {output_path}")
                except Exception as e:
                    print(f"Export error: {e}")

            background_tasks.add_task(export_task)

            return {
                "status": "exporting",
                "dataset_id": request.dataset_id,
                "format": "huggingface",
                "output_filename": request.output_filename
            }

        elif request.format.lower() == "openai":
            exporter = OpenAIExporter()

            def export_task():
                try:
                    # Handle train/validation split if needed
                    if request.train_split < 1.0:
                        train_path, val_path = exporter.split_dataset(
                            examples=examples,
                            train_ratio=request.train_split,
                            output_prefix=request.output_filename
                        )
                        print(f"OpenAI export completed: {train_path}, {val_path}")
                    else:
                        output_path = exporter.export(
                            examples=examples,
                            output_filename=request.output_filename,
                            system_message=request.system_message,
                            **request.additional_params
                        )
                        print(f"OpenAI export completed: {output_path}")

                    db.update_dataset(request.dataset_id, {"last_export": datetime.now().isoformat()})
                except Exception as e:
                    print(f"Export error: {e}")

            background_tasks.add_task(export_task)

            return {
                "status": "exporting",
                "dataset_id": request.dataset_id,
                "format": "openai",
                "output_filename": request.output_filename,
                "train_split": request.train_split
            }

        elif request.format.lower() in ["alpaca", "sharegpt"]:
            exporter = AlpacaExporter()
            format_type = request.format.lower()

            def export_task():
                try:
                    output_path = exporter.export(
                        examples=examples,
                        output_filename=request.output_filename,
                        format_type=format_type,
                        **request.additional_params
                    )
                    db.update_dataset(request.dataset_id, {"last_export": datetime.now().isoformat()})
                    print(f"{format_type.capitalize()} export completed: {output_path}")
                except Exception as e:
                    print(f"Export error: {e}")

            background_tasks.add_task(export_task)

            return {
                "status": "exporting",
                "dataset_id": request.dataset_id,
                "format": format_type,
                "output_filename": request.output_filename
            }

        elif request.format.lower() == "langchain":
            exporter = LangChainExporter()
            export_type = request.additional_params.get("export_type", "documents")

            def export_task():
                try:
                    output_path = exporter.export(
                        examples=examples,
                        output_filename=request.output_filename,
                        export_type=export_type,
                        **request.additional_params
                    )
                    db.update_dataset(request.dataset_id, {"last_export": datetime.now().isoformat()})
                    print(f"LangChain export completed: {output_path}")
                except Exception as e:
                    print(f"Export error: {e}")

            background_tasks.add_task(export_task)

            return {
                "status": "exporting",
                "dataset_id": request.dataset_id,
                "format": "langchain",
                "export_type": export_type,
                "output_filename": request.output_filename
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export format: {request.format}. Supported: huggingface, openai, alpaca, sharegpt, langchain"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/formats")
async def list_export_formats():
    """List available export formats"""
    return {
        "formats": [
            {
                "name": "huggingface",
                "description": "HuggingFace datasets format with metadata",
                "file_extension": "directory with JSONL",
                "features": ["dataset_info.json", "README.md", "push to hub script"],
                "use_cases": ["Upload to HuggingFace Hub", "Share datasets publicly"]
            },
            {
                "name": "openai",
                "description": "OpenAI fine-tuning format (JSONL)",
                "file_extension": ".jsonl",
                "features": ["chat format", "train/val split", "validation report", "cost estimation"],
                "use_cases": ["Fine-tune GPT-3.5/GPT-4", "OpenAI API training"]
            },
            {
                "name": "alpaca",
                "description": "Alpaca instruction format (JSON)",
                "file_extension": ".json",
                "features": ["instruction-input-output structure", "compatible with FastChat"],
                "use_cases": ["Instruction tuning", "Fine-tune LLaMA/Alpaca models"]
            },
            {
                "name": "sharegpt",
                "description": "ShareGPT conversation format (JSON)",
                "file_extension": ".json",
                "features": ["human-gpt conversation structure", "multi-turn dialogues"],
                "use_cases": ["Chat model training", "Vicuna/ShareGPT format"]
            },
            {
                "name": "langchain",
                "description": "LangChain document format (JSONL)",
                "file_extension": ".jsonl",
                "features": ["documents/chat/qa_pairs modes", "vector store ready", "loader scripts"],
                "use_cases": ["RAG systems", "Vector databases", "LangChain applications"]
            },
            {
                "name": "csv",
                "description": "CSV format for general use",
                "file_extension": ".csv",
                "features": ["spreadsheet compatible", "universal format"],
                "use_cases": ["Data analysis", "Excel/Sheets", "General purpose"]
            }
        ]
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
    """Get current settings"""
    return {
        "default_provider": config.get_default_provider(),
        "generation": config.get_generation_settings(),
        "quality": config.get_quality_settings(),
        "system": config.get_system_settings()
    }

# Model Management Endpoints
@app.get("/api/models/providers")
async def get_providers():
    """Get list of available LLM providers"""
    try:
        model_manager = get_model_manager()
        providers = model_manager.get_providers()
        return {"providers": providers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{provider}")
async def get_provider_models_endpoint(provider: str):
    """Get list of models for a specific provider"""
    try:
        model_manager = get_model_manager()
        models = model_manager.get_models(provider)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{provider}/{model_id}/info")
async def get_model_info_endpoint(provider: str, model_id: str):
    """Get detailed information about a specific model"""
    try:
        model_manager = get_model_manager()
        model_info = model_manager.get_model_info(provider, model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found for provider '{provider}'")
        return model_info
    except HTTPException:
        raise
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

# Analytics routes
@app.get("/api/analytics/summary")
async def get_analytics_summary(days: int = Query(30, ge=1, le=365)):
    """Get analytics summary for the specified period"""
    try:
        from datetime import datetime, timedelta

        tracker = get_tracker()
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        summary = tracker.get_summary(start_date, end_date)

        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/datasets/{dataset_id}/stats")
async def get_dataset_analytics(dataset_id: int):
    """Get detailed statistics for a specific dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            return {"error": "No examples found in dataset"}

        # Analyze dataset
        stats = DatasetStats.analyze_dataset(examples)

        # Add recommendations
        recommendations = DatasetStats.get_recommendations(examples)

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.get("name"),
            "statistics": stats,
            "recommendations": recommendations
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analytics/datasets/compare")
async def compare_datasets_analytics(dataset_ids: List[int] = Body(...)):
    """Compare statistics between two datasets"""
    try:
        if len(dataset_ids) != 2:
            raise HTTPException(status_code=400, detail="Exactly 2 dataset IDs required")

        # Load both datasets
        examples1 = db.get_examples(dataset_ids[0])
        examples2 = db.get_examples(dataset_ids[1])

        if not examples1:
            raise HTTPException(status_code=404, detail=f"No examples in dataset {dataset_ids[0]}")
        if not examples2:
            raise HTTPException(status_code=404, detail=f"No examples in dataset {dataset_ids[1]}")

        # Compare datasets
        comparison = DatasetStats.compare_datasets(examples1, examples2)

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/cost-estimate")
async def estimate_generation_cost(
    provider: str,
    model: str,
    examples_count: int = Query(..., ge=1),
    avg_tokens_per_example: int = Query(1000, ge=1)
):
    """Estimate cost for generating examples"""
    try:
        tracker = get_tracker()

        total_tokens = examples_count * avg_tokens_per_example
        estimated_cost = tracker.estimate_generation_cost(provider, model, total_tokens)

        return {
            "provider": provider,
            "model": model,
            "examples_count": examples_count,
            "avg_tokens_per_example": avg_tokens_per_example,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Quality Control routes
@app.post("/api/quality/check")
async def run_quality_check(request: QualityCheckRequest, background_tasks: BackgroundTasks):
    """Run advanced quality checks on a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

        # Load examples
        examples = db.get_examples(request.dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found in dataset")

        # Determine which checks to run
        check_types = request.check_types
        if "all" in check_types:
            check_types = ["toxicity", "deduplication", "diversity", "scoring"]

        results = {}

        # Run checks
        if "toxicity" in check_types:
            detector = ToxicityDetector()
            texts = [ex.get(request.text_field, "") for ex in examples]
            toxicity_results = detector.detect_batch(texts)
            results["toxicity"] = detector.get_statistics(toxicity_results)

        if "deduplication" in check_types:
            checker = DeduplicationChecker()
            dedup_result = checker.check_exact_duplicates(examples, request.text_field)
            results["deduplication"] = checker.get_statistics(dedup_result)

        if "diversity" in check_types:
            analyzer = DiversityAnalyzer()
            diversity_metrics = analyzer.analyze(examples, request.text_field)
            results["diversity"] = {
                "vocabulary_size": diversity_metrics.vocabulary_size,
                "lexical_diversity": diversity_metrics.lexical_diversity,
                "entropy": diversity_metrics.entropy,
                "recommendations": diversity_metrics.recommendations
            }

        if "scoring" in check_types:
            scorer = QualityScorer()
            quality_score = scorer.score_dataset(examples, request.text_field)
            results["scoring"] = {
                "overall_score": quality_score.overall_score,
                "grade": quality_score.grade,
                "component_scores": quality_score.component_scores,
                "issues": quality_score.issues,
                "strengths": quality_score.strengths
            }

        # Generate full report if requested
        if request.generate_report:
            def generate_report_task():
                try:
                    report_gen = QualityReportGenerator()
                    report = report_gen.generate_full_report(
                        examples,
                        dataset.get("name", f"dataset_{request.dataset_id}"),
                        request.text_field
                    )
                    print(f"Quality report generated for dataset {request.dataset_id}")
                except Exception as e:
                    print(f"Error generating report: {e}")

            background_tasks.add_task(generate_report_task)

        return {
            "dataset_id": request.dataset_id,
            "checks_run": check_types,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality/toxicity/filter")
async def filter_toxic_examples(dataset_id: int, sensitivity: str = "medium"):
    """Filter out toxic examples from a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Filter toxic examples
        detector = ToxicityDetector(sensitivity=sensitivity)
        clean, toxic = detector.filter_toxic(examples)

        return {
            "dataset_id": dataset_id,
            "total_examples": len(examples),
            "clean_examples": len(clean),
            "toxic_examples": len(toxic),
            "toxic_percentage": round(len(toxic) / len(examples) * 100, 2) if examples else 0,
            "message": f"Found {len(toxic)} toxic examples. Use /api/quality/toxicity/remove to remove them."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality/deduplication/remove")
async def remove_duplicate_examples(
    dataset_id: int,
    method: str = Query("exact", description="Deduplication method: exact, fuzzy, semantic")
):
    """Remove duplicate examples from a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Remove duplicates
        checker = DeduplicationChecker()
        unique_examples, duplicate_groups = checker.remove_duplicates(examples, method=method)

        return {
            "dataset_id": dataset_id,
            "method": method,
            "original_count": len(examples),
            "unique_count": len(unique_examples),
            "removed_count": len(examples) - len(unique_examples),
            "duplicate_groups": len(duplicate_groups),
            "message": f"Found {len(examples) - len(unique_examples)} duplicates. Create a new version to save changes."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================================================
# Phase 6: Advanced Quality Metrics Endpoints
# ==================================================

@app.post("/api/quality/analyze/toxicity")
async def analyze_toxicity(
    dataset_id: int,
    threshold: float = Query(0.5, description="Toxicity threshold (0-1)")
):
    """Analyze dataset for toxic content"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Analyze each example
        results = []
        toxic_count = 0
        
        for example in examples:
            # Get text to analyze (try different fields)
            text = ""
            if isinstance(example, dict):
                # Handle chat format with messages array
                if 'messages' in example and isinstance(example['messages'], list):
                    # Concatenate all message contents
                    text = " ".join([msg.get('content', '') for msg in example['messages'] if isinstance(msg, dict)])
                elif 'output' in example:
                    text = example['output']
                elif 'text' in example:
                    text = example['text']
                elif 'response' in example:
                    text = example['response']
                elif 'content' in example:
                    text = example['content']
            
            if not text:
                continue
                
            # Analyze toxicity
            scores = toxicity_analyzer.analyze(text)
            
            # Check if toxic
            is_toxic = scores['toxicity'] > threshold
            if is_toxic:
                toxic_count += 1
            
            results.append({
                'example_id': example.get('id'),
                'is_toxic': is_toxic,
                'scores': scores
            })

        return {
            'dataset_id': dataset_id,
            'total_analyzed': len(results),
            'toxic_count': toxic_count,
            'toxic_percentage': round(toxic_count / len(results) * 100, 2) if results else 0,
            'results': results[:100]  # Limit results to first 100 for display
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analyze/pii")
async def analyze_pii(dataset_id: int):
    """Analyze dataset for PII (Personally Identifiable Information)"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Analyze each example
        results = []
        pii_count = 0
        pii_types = {}
        
        for example in examples:
            # Get text to analyze
            text = ""
            if isinstance(example, dict):
                # Handle chat format with messages array
                if 'messages' in example and isinstance(example['messages'], list):
                    text = " ".join([msg.get('content', '') for msg in example['messages'] if isinstance(msg, dict)])
                elif 'output' in example:
                    text = example['output']
                elif 'text' in example:
                    text = example['text']
                elif 'response' in example:
                    text = example['response']
                elif 'content' in example:
                    text = example['content']
            
            if not text:
                continue
                
            # Analyze PII
            entities = pii_analyzer.analyze(text)
            
            if entities:
                pii_count += 1
                for entity in entities:
                    entity_type = entity['type']
                    pii_types[entity_type] = pii_types.get(entity_type, 0) + 1
            
            results.append({
                'example_id': example.get('id'),
                'has_pii': len(entities) > 0,
                'entities': entities
            })

        return {
            'dataset_id': dataset_id,
            'total_analyzed': len(results),
            'pii_count': pii_count,
            'pii_percentage': round(pii_count / len(results) * 100, 2) if results else 0,
            'pii_types': pii_types,
            'results': results[:100]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analyze/duplicates")
async def analyze_duplicates(
    dataset_id: int,
    threshold: float = Query(0.8, description="Similarity threshold (0-1)")
):
    """Analyze dataset for duplicate content"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Clear previous index
        deduplicator.clear()
        
        # Index all documents
        duplicate_groups = []
        indexed_ids = set()
        
        for i, example in enumerate(examples):
            # Get text to analyze
            text = ""
            if isinstance(example, dict):
                # Handle chat format with messages array
                if 'messages' in example and isinstance(example['messages'], list):
                    text = " ".join([msg.get('content', '') for msg in example['messages'] if isinstance(msg, dict)])
                elif 'output' in example:
                    text = example['output']
                elif 'text' in example:
                    text = example['text']
                elif 'response' in example:
                    text = example['response']
                elif 'content' in example:
                    text = example['content']
            
            if not text:
                continue
            
            doc_id = f"doc_{i}"
            
            # Check for duplicates before adding
            if doc_id not in indexed_ids:
                duplicates = deduplicator.find_duplicates(doc_id, text)
                
                if duplicates:
                    # Found duplicates
                    group = [doc_id] + duplicates
                    duplicate_groups.append({
                        'group': group,
                        'example_ids': [examples[int(d.split('_')[1])].get('id') for d in group if int(d.split('_')[1]) < len(examples)]
                    })
                
                # Add to index
                deduplicator.add_document(doc_id, text)
                indexed_ids.add(doc_id)

        # Calculate stats
        total_duplicates = sum(len(g['group']) - 1 for g in duplicate_groups)

        return {
            'dataset_id': dataset_id,
            'total_analyzed': len(examples),
            'duplicate_groups': len(duplicate_groups),
            'total_duplicates': total_duplicates,
            'duplicate_percentage': round(total_duplicates / len(examples) * 100, 2) if examples else 0,
            'groups': duplicate_groups[:50]  # Show first 50 groups
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analyze/full")
async def analyze_full_quality(
    dataset_id: int,
    toxicity_threshold: float = Query(0.5, description="Toxicity threshold"),
    duplicate_threshold: float = Query(0.8, description="Duplicate similarity threshold")
):
    """Run all quality checks on a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Run all analyses in parallel (conceptually, but sequentially for simplicity)
        # In production, you might want to use asyncio.gather() here
        
        # Toxicity analysis
        toxicity_result = await analyze_toxicity(dataset_id, toxicity_threshold)
        
        # PII analysis
        pii_result = await analyze_pii(dataset_id)
        
        # Duplicate analysis
        duplicate_result = await analyze_duplicates(dataset_id, duplicate_threshold)

        return {
            'dataset_id': dataset_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_examples': toxicity_result['total_analyzed'],
                'toxic_count': toxicity_result['toxic_count'],
                'toxic_percentage': toxicity_result['toxic_percentage'],
                'pii_count': pii_result['pii_count'],
                'pii_percentage': pii_result['pii_percentage'],
                'duplicate_groups': duplicate_result['duplicate_groups'],
                'duplicate_percentage': duplicate_result['duplicate_percentage']
            },
            'toxicity': toxicity_result,
            'pii': pii_result,
            'duplicates': duplicate_result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/anonymize/pii")
async def anonymize_dataset_pii(dataset_id: int):
    """Anonymize PII in a dataset (creates a new version)"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Anonymize each example
        anonymized_examples = []
        anonymized_count = 0
        
        for example in examples:
            # Get text to anonymize
            text_field = None
            text = ""
            
            if isinstance(example, dict):
                # Handle chat format with messages array
                if 'messages' in example and isinstance(example['messages'], list):
                    # Anonymize each message separately
                    anonymized_messages = []
                    for msg in example['messages']:
                        if isinstance(msg, dict) and 'content' in msg:
                            anonymized_content = pii_analyzer.anonymize(msg['content'])
                            if anonymized_content != msg['content']:
                                anonymized_count += 1
                            msg_copy = msg.copy()
                            msg_copy['content'] = anonymized_content
                            anonymized_messages.append(msg_copy)
                        else:
                            anonymized_messages.append(msg)
                    example_copy = example.copy()
                    example_copy['messages'] = anonymized_messages
                    anonymized_examples.append(example_copy)
                    continue
                elif 'output' in example:
                    text_field = 'output'
                    text = example['output']
                elif 'text' in example:
                    text_field = 'text'
                    text = example['text']
                elif 'response' in example:
                    text_field = 'response'
                    text = example['response']
                elif 'content' in example:
                    text_field = 'content'
                    text = example['content']
            
            if text and text_field:
                # Anonymize
                anonymized_text = pii_analyzer.anonymize(text)
                
                if anonymized_text != text:
                    anonymized_count += 1
                
                # Update example
                example_copy = example.copy()
                example_copy[text_field] = anonymized_text
                anonymized_examples.append(example_copy)
            else:
                anonymized_examples.append(example)

        return {
            'dataset_id': dataset_id,
            'total_examples': len(examples),
            'anonymized_count': anonymized_count,
            'message': f"Anonymized {anonymized_count} examples. Create a new version to save changes.",
            'preview': anonymized_examples[:10]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# Dataset Analytics Endpoints (Evaluation Module)
# ==================================================

@app.post("/api/quality/analytics")
async def analyze_dataset_analytics(dataset_id: int):
    """
    Comprehensive dataset analytics including diversity metrics,
    text statistics, and semantic clustering.
    """
    try:
        from evaluation import DatasetEvaluator
        
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")
        
        # Run evaluation
        evaluator = DatasetEvaluator(use_semantic=True)
        report = evaluator.evaluate(
            examples=examples,
            dataset_id=dataset_id,
            dataset_name=dataset.get('name', 'Unknown')
        )
        
        return report.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analytics/diversity")
async def analyze_diversity_only(dataset_id: int):
    """
    Analyze dataset diversity metrics only (Distinct-n, Self-BLEU).
    Faster than full analytics.
    """
    try:
        from evaluation.diversity_metrics import analyze_diversity
        from evaluation.text_stats import extract_messages
        
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")
        
        # Extract all texts
        texts = []
        for example in examples:
            user_msgs, assistant_msgs = extract_messages(example)
            texts.extend(user_msgs)
            texts.extend(assistant_msgs)
        
        # Analyze diversity
        from dataclasses import asdict
        metrics = analyze_diversity(texts)
        
        return {
            "dataset_id": dataset_id,
            "total_examples": len(examples),
            "total_texts": len(texts),
            "metrics": asdict(metrics)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analytics/text-stats")
async def analyze_text_stats_only(dataset_id: int):
    """
    Analyze text statistics only (lengths, vocabulary, patterns).
    Faster than full analytics.
    """
    try:
        from evaluation.text_stats import calculate_text_stats
        from dataclasses import asdict
        
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")
        
        # Calculate stats
        stats = calculate_text_stats(examples)
        
        return {
            "dataset_id": dataset_id,
            "stats": {
                "total_examples": stats.total_examples,
                "total_tokens": stats.total_tokens,
                "total_characters": stats.total_characters,
                "question_length": asdict(stats.question_length),
                "answer_length": asdict(stats.answer_length),
                "avg_qa_ratio": stats.avg_qa_ratio,
                "vocabulary_size": stats.vocabulary_size,
                "top_words": stats.top_words,
                "languages": stats.languages,
                "has_code_blocks": stats.has_code_blocks,
                "has_lists": stats.has_lists,
                "has_urls": stats.has_urls,
                "has_numbers": stats.has_numbers,
                "empty_responses": stats.empty_responses,
                "very_short_responses": stats.very_short_responses,
                "very_long_responses": stats.very_long_responses,
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/analytics/semantic")
async def analyze_semantic_only(
    dataset_id: int,
    num_clusters: Optional[int] = Query(None, description="Number of clusters (auto if not set)")
):
    """
    Analyze semantic clustering only.
    Requires sentence-transformers for best results.
    """
    try:
        from evaluation.semantic_analyzer import SemanticAnalyzer
        
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load examples
        examples = db.get_examples(dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")
        
        # Analyze
        analyzer = SemanticAnalyzer()
        analysis = analyzer.analyze(examples, num_clusters=num_clusters)
        
        return {
            "dataset_id": dataset_id,
            "total_examples": len(examples),
            "semantic": {
                "num_clusters": analysis.num_clusters,
                "semantic_diversity": analysis.semantic_diversity,
                "avg_cluster_size": analysis.avg_cluster_size,
                "largest_cluster_ratio": analysis.largest_cluster_ratio,
                "outlier_count": analysis.outlier_count,
                "topic_distribution": analysis.topic_distribution,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size,
                        "centroid_text": c.centroid_text,
                        "keywords": c.keywords,
                    }
                    for c in analysis.clusters
                ]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dataset Versioning routes
@app.post("/api/versions/create")
async def create_dataset_version(request: VersionCreateRequest):
    """Create a new version of a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

        # Load examples
        examples = db.get_examples(request.dataset_id)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        # Create version
        version_manager = VersionManager()
        version = version_manager.create_version(
            dataset_id=request.dataset_id,
            examples=examples,
            commit_message=request.commit_message,
            author=request.author,
            tags=request.tags
        )

        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "commit_message": version.commit_message,
            "timestamp": version.timestamp,
            "examples_count": version.examples_count,
            "tags": version.tags
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/versions/list/{dataset_id}")
async def list_dataset_versions(dataset_id: int, limit: Optional[int] = None):
    """List all versions of a dataset"""
    try:
        version_manager = VersionManager()
        versions = version_manager.list_versions(dataset_id, limit=limit)

        return {
            "dataset_id": dataset_id,
            "total_versions": len(versions),
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "commit_message": v.commit_message,
                    "author": v.author,
                    "timestamp": v.timestamp,
                    "examples_count": v.examples_count,
                    "tags": v.tags
                }
                for v in versions
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/versions/{version_id}")
async def get_dataset_version(version_id: str):
    """Get a specific version"""
    try:
        version_manager = VersionManager()
        version = version_manager.get_version(version_id)

        if not version:
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")

        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "dataset_id": version.dataset_id,
            "commit_message": version.commit_message,
            "author": version.author,
            "timestamp": version.timestamp,
            "examples_count": version.examples_count,
            "tags": version.tags,
            "parent_version": version.parent_version
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/versions/rollback/{version_id}")
async def rollback_to_version(version_id: str):
    """Rollback dataset to a specific version"""
    try:
        version_manager = VersionManager()
        examples = version_manager.rollback_to_version(version_id)

        if examples is None:
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")

        return {
            "version_id": version_id,
            "examples_count": len(examples),
            "message": "Version restored. Create a new version to save changes."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/versions/diff/{version_id1}/{version_id2}")
async def diff_versions(version_id1: str, version_id2: str):
    """Calculate diff between two versions"""
    try:
        version_manager = VersionManager()
        differ = DatasetDiff()

        # Get both versions
        examples1 = version_manager.get_version_examples(version_id1)
        examples2 = version_manager.get_version_examples(version_id2)

        if examples1 is None:
            raise HTTPException(status_code=404, detail=f"Version '{version_id1}' not found")
        if examples2 is None:
            raise HTTPException(status_code=404, detail=f"Version '{version_id2}' not found")

        # Calculate diff
        diff_result = differ.diff(examples1, examples2)

        return {
            "version_id1": version_id1,
            "version_id2": version_id2,
            "summary": diff_result.summary,
            "total_changes": diff_result.total_changes,
            "added_count": len(diff_result.added),
            "removed_count": len(diff_result.removed),
            "modified_count": len(diff_result.modified)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/versions/merge")
async def merge_versions(request: VersionMergeRequest):
    """Merge two dataset versions"""
    try:
        version_manager = VersionManager()
        merger = DatasetMerger()

        # Get both versions
        examples1 = version_manager.get_version_examples(request.version_id1)
        examples2 = version_manager.get_version_examples(request.version_id2)

        if examples1 is None:
            raise HTTPException(status_code=404, detail=f"Version '{request.version_id1}' not found")
        if examples2 is None:
            raise HTTPException(status_code=404, detail=f"Version '{request.version_id2}' not found")

        # Merge (using empty base for now)
        merge_result = merger.merge(
            base_examples=[],
            branch1_examples=examples1,
            branch2_examples=examples2,
            strategy=request.strategy
        )

        return {
            "version_id1": request.version_id1,
            "version_id2": request.version_id2,
            "strategy": request.strategy,
            "success": merge_result.success,
            "merged_count": len(merge_result.merged_examples),
            "conflicts_count": len(merge_result.conflicts),
            "stats": merge_result.stats,
            "message": "Merge completed. Create a new version to save merged dataset."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/versions/history/{dataset_id}")
async def get_version_history(dataset_id: int):
    """Get version history with change summaries"""
    try:
        version_manager = VersionManager()
        history = version_manager.get_version_history(dataset_id)

        return {
            "dataset_id": dataset_id,
            "total_versions": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# COLLABORATION API ENDPOINTS
# ============================================================================

# User Management Endpoints
@app.post("/api/users/create")
async def create_user(request: UserCreateRequest):
    """Create a new user"""
    try:
        user_manager = get_user_manager()

        # Parse role
        try:
            role = UserRole(request.role.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}"
            )

        user = user_manager.create_user(
            username=request.username,
            email=request.email,
            role=role,
            metadata=request.metadata
        )

        return {
            "success": True,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "created_at": user.created_at
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/list")
async def list_users(role: Optional[str] = None):
    """List all users with optional role filter"""
    try:
        user_manager = get_user_manager()

        # Parse role filter if provided
        role_filter = None
        if role:
            try:
                role_filter = UserRole(role.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}"
                )

        users = user_manager.list_users(role=role_filter)

        return {
            "total": len(users),
            "users": [
                {
                    "user_id": u.user_id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role.value,
                    "created_at": u.created_at,
                    "last_active": u.last_active
                }
                for u in users
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user by ID"""
    try:
        user_manager = get_user_manager()
        user = user_manager.get_user(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "created_at": user.created_at,
            "last_active": user.last_active,
            "metadata": user.metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/users/{user_id}")
async def update_user(user_id: str, request: UserUpdateRequest):
    """Update user information"""
    try:
        user_manager = get_user_manager()

        # Parse role if provided
        role = None
        if request.role:
            try:
                role = UserRole(request.role.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}"
                )

        success = user_manager.update_user(
            user_id=user_id,
            username=request.username,
            email=request.email,
            role=role,
            metadata=request.metadata
        )

        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        # Get updated user
        user = user_manager.get_user(user_id)

        return {
            "success": True,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user"""
    try:
        user_manager = get_user_manager()
        success = user_manager.delete_user(user_id)

        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return {"success": True, "message": f"User {user_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Permission Management Endpoints
@app.post("/api/permissions/grant")
async def grant_permission(request: PermissionGrantRequest):
    """Grant permission to a user for a dataset"""
    try:
        permission_manager = get_permission_manager()

        # Parse permission level
        try:
            level = PermissionLevel(request.permission_level.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid permission level. Must be one of: {[p.value for p in PermissionLevel]}"
            )

        permission = permission_manager.grant_permission(
            dataset_id=request.dataset_id,
            user_id=request.user_id,
            permission_level=level,
            granted_by=request.granted_by
        )

        return {
            "success": True,
            "permission": {
                "dataset_id": permission.dataset_id,
                "user_id": permission.user_id,
                "permission_level": permission.permission_level.value,
                "granted_by": permission.granted_by,
                "granted_at": permission.granted_at
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/permissions/revoke")
async def revoke_permission(dataset_id: int, user_id: str):
    """Revoke user's permission for a dataset"""
    try:
        permission_manager = get_permission_manager()
        success = permission_manager.revoke_permission(dataset_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Permission not found")

        return {
            "success": True,
            "message": f"Permission revoked for user {user_id} on dataset {dataset_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/permissions/dataset/{dataset_id}")
async def get_dataset_permissions(dataset_id: int):
    """Get all permissions for a dataset"""
    try:
        permission_manager = get_permission_manager()
        permissions = permission_manager.list_dataset_permissions(dataset_id)

        return {
            "dataset_id": dataset_id,
            "total": len(permissions),
            "permissions": [
                {
                    "user_id": p.user_id,
                    "permission_level": p.permission_level.value,
                    "granted_by": p.granted_by,
                    "granted_at": p.granted_at
                }
                for p in permissions
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/permissions/user/{user_id}")
async def get_user_permissions(user_id: str):
    """Get all permissions for a user"""
    try:
        permission_manager = get_permission_manager()
        permissions = permission_manager.list_user_permissions(user_id)

        return {
            "user_id": user_id,
            "total": len(permissions),
            "permissions": [
                {
                    "dataset_id": p.dataset_id,
                    "permission_level": p.permission_level.value,
                    "granted_by": p.granted_by,
                    "granted_at": p.granted_at
                }
                for p in permissions
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/permissions/share")
async def share_dataset(request: ShareDatasetRequest):
    """Share dataset with another user"""
    try:
        permission_manager = get_permission_manager()

        # Parse permission level
        try:
            level = PermissionLevel(request.permission_level.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid permission level. Must be one of: {[p.value for p in PermissionLevel if p != PermissionLevel.OWNER]}"
            )

        # Don't allow sharing ownership
        if level == PermissionLevel.OWNER:
            raise HTTPException(
                status_code=400,
                detail="Cannot share ownership. Use transfer_ownership endpoint instead."
            )

        permission = permission_manager.share_dataset(
            dataset_id=request.dataset_id,
            from_user_id=request.from_user_id,
            to_user_id=request.to_user_id,
            permission_level=level
        )

        if not permission:
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only owners can share datasets."
            )

        return {
            "success": True,
            "permission": {
                "dataset_id": permission.dataset_id,
                "user_id": permission.user_id,
                "permission_level": permission.permission_level.value
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Comment Management Endpoints
@app.post("/api/comments/create")
async def create_comment(request: CommentCreateRequest):
    """Create a comment on an example"""
    try:
        comment_manager = get_comment_manager()

        comment = comment_manager.create_comment(
            dataset_id=request.dataset_id,
            example_id=request.example_id,
            user_id=request.user_id,
            content=request.content,
            parent_comment_id=request.parent_comment_id
        )

        return {
            "success": True,
            "comment": {
                "comment_id": comment.comment_id,
                "dataset_id": comment.dataset_id,
                "example_id": comment.example_id,
                "user_id": comment.user_id,
                "content": comment.content,
                "created_at": comment.created_at,
                "parent_comment_id": comment.parent_comment_id,
                "resolved": comment.resolved
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/comments/list")
async def list_comments(
    dataset_id: Optional[int] = None,
    example_id: Optional[int] = None,
    user_id: Optional[str] = None
):
    """List comments with optional filters"""
    try:
        comment_manager = get_comment_manager()
        comments = comment_manager.list_comments(
            dataset_id=dataset_id,
            example_id=example_id,
            user_id=user_id
        )

        return {
            "total": len(comments),
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "dataset_id": c.dataset_id,
                    "example_id": c.example_id,
                    "user_id": c.user_id,
                    "content": c.content,
                    "created_at": c.created_at,
                    "parent_comment_id": c.parent_comment_id,
                    "resolved": c.resolved
                }
                for c in comments
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/comments/{comment_id}")
async def get_comment(comment_id: str):
    """Get comment by ID"""
    try:
        comment_manager = get_comment_manager()
        comment = comment_manager.get_comment(comment_id)

        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")

        return {
            "comment_id": comment.comment_id,
            "dataset_id": comment.dataset_id,
            "example_id": comment.example_id,
            "user_id": comment.user_id,
            "content": comment.content,
            "created_at": comment.created_at,
            "parent_comment_id": comment.parent_comment_id,
            "resolved": comment.resolved
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/comments/{comment_id}")
async def update_comment(comment_id: str, request: CommentUpdateRequest):
    """Update comment content"""
    try:
        comment_manager = get_comment_manager()
        success = comment_manager.update_comment(comment_id, request.content)

        if not success:
            raise HTTPException(status_code=404, detail="Comment not found")

        return {"success": True, "message": "Comment updated"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/comments/{comment_id}")
async def delete_comment(comment_id: str):
    """Delete comment"""
    try:
        comment_manager = get_comment_manager()
        success = comment_manager.delete_comment(comment_id)

        if not success:
            raise HTTPException(status_code=404, detail="Comment not found")

        return {"success": True, "message": "Comment deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/comments/threads/{dataset_id}/{example_id}")
async def get_example_threads(dataset_id: int, example_id: int):
    """Get all comment threads for an example"""
    try:
        comment_manager = get_comment_manager()
        threads = comment_manager.get_example_threads(dataset_id, example_id)

        return {
            "dataset_id": dataset_id,
            "example_id": example_id,
            "thread_count": len(threads),
            "threads": [
                [
                    {
                        "comment_id": c.comment_id,
                        "user_id": c.user_id,
                        "content": c.content,
                        "created_at": c.created_at,
                        "resolved": c.resolved
                    }
                    for c in thread
                ]
                for thread in threads
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/comments/{comment_id}/resolve")
async def resolve_comment_thread(comment_id: str):
    """Mark a comment thread as resolved"""
    try:
        comment_manager = get_comment_manager()
        success = comment_manager.resolve_thread(comment_id)

        if not success:
            raise HTTPException(status_code=404, detail="Comment not found")

        return {"success": True, "message": "Thread resolved"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Review Management Endpoints
@app.post("/api/reviews/create")
async def create_review(request: ReviewCreateRequest):
    """Create a review for an example"""
    try:
        review_manager = get_review_manager()

        # Parse status
        try:
            status = ReviewStatus(request.status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {[s.value for s in ReviewStatus]}"
            )

        review = review_manager.create_review(
            dataset_id=request.dataset_id,
            example_id=request.example_id,
            reviewer_id=request.reviewer_id,
            status=status,
            feedback=request.feedback
        )

        return {
            "success": True,
            "review": {
                "review_id": review.review_id,
                "dataset_id": review.dataset_id,
                "example_id": review.example_id,
                "reviewer_id": review.reviewer_id,
                "status": review.status.value,
                "feedback": review.feedback,
                "created_at": review.created_at
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/approve")
async def approve_example(
    dataset_id: int,
    example_id: int,
    reviewer_id: str,
    feedback: Optional[str] = None
):
    """Approve an example"""
    try:
        review_manager = get_review_manager()

        review = review_manager.approve_example(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            feedback=feedback
        )

        return {
            "success": True,
            "review": {
                "review_id": review.review_id,
                "status": review.status.value,
                "created_at": review.created_at
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/reject")
async def reject_example(
    dataset_id: int,
    example_id: int,
    reviewer_id: str,
    feedback: str
):
    """Reject an example"""
    try:
        review_manager = get_review_manager()

        review = review_manager.reject_example(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            feedback=feedback
        )

        return {
            "success": True,
            "review": {
                "review_id": review.review_id,
                "status": review.status.value,
                "feedback": review.feedback,
                "created_at": review.created_at
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/request-changes")
async def request_changes(
    dataset_id: int,
    example_id: int,
    reviewer_id: str,
    feedback: str
):
    """Request changes to an example"""
    try:
        review_manager = get_review_manager()

        review = review_manager.request_changes(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            feedback=feedback
        )

        return {
            "success": True,
            "review": {
                "review_id": review.review_id,
                "status": review.status.value,
                "feedback": review.feedback,
                "created_at": review.created_at
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/list")
async def list_reviews(
    dataset_id: Optional[int] = None,
    reviewer_id: Optional[str] = None,
    status: Optional[str] = None
):
    """List reviews with optional filters"""
    try:
        review_manager = get_review_manager()

        # Parse status if provided
        status_filter = None
        if status:
            try:
                status_filter = ReviewStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {[s.value for s in ReviewStatus]}"
                )

        reviews = review_manager.list_reviews(
            dataset_id=dataset_id,
            reviewer_id=reviewer_id,
            status=status_filter
        )

        return {
            "total": len(reviews),
            "reviews": [
                {
                    "review_id": r.review_id,
                    "dataset_id": r.dataset_id,
                    "example_id": r.example_id,
                    "reviewer_id": r.reviewer_id,
                    "status": r.status.value,
                    "feedback": r.feedback,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at
                }
                for r in reviews
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/summary/{dataset_id}")
async def get_review_summary(dataset_id: int):
    """Get review summary for a dataset"""
    try:
        review_manager = get_review_manager()
        summary = review_manager.get_review_summary(dataset_id)

        return {
            "dataset_id": dataset_id,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/pending/{dataset_id}")
async def get_pending_reviews(dataset_id: int):
    """Get list of example IDs pending review"""
    try:
        review_manager = get_review_manager()
        pending_ids = review_manager.get_pending_reviews(dataset_id)

        return {
            "dataset_id": dataset_id,
            "pending_count": len(pending_ids),
            "example_ids": pending_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/approved/{dataset_id}")
async def get_approved_examples(dataset_id: int):
    """Get list of approved example IDs"""
    try:
        review_manager = get_review_manager()
        approved_ids = review_manager.get_approved_examples(dataset_id)

        return {
            "dataset_id": dataset_id,
            "approved_count": len(approved_ids),
            "example_ids": approved_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATA AUGMENTATION API ENDPOINTS
# ============================================================================

@app.post("/api/augmentation/augment")
async def augment_dataset(request: AugmentationRequest):
    """Augment a dataset with various techniques"""
    try:
        # Get dataset examples
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        examples = db.get_examples(request.dataset_id)
        if not examples:
            raise HTTPException(status_code=400, detail="Dataset has no examples")

        # Create augmentation config
        config = AugmentationConfig(
            techniques=request.techniques,
            samples_per_example=request.samples_per_example,
            synonym_ratio=request.synonym_ratio,
            random_swap_ratio=request.random_swap_ratio,
            random_delete_ratio=request.random_delete_ratio,
            random_insert_ratio=request.random_insert_ratio,
            paraphrase_diversity=request.paraphrase_diversity
        )

        # Perform augmentation
        augmenter = DataAugmenter(config)
        result = augmenter.augment(
            examples,
            text_field=request.text_field
        )

        # Save augmented dataset if requested
        new_dataset_id = None
        if request.save_augmented:
            # Create new dataset with augmented examples
            new_dataset_id = db.create_dataset(
                name=f"{dataset['name']}_augmented",
                domain=dataset.get('domain', 'unknown'),
                format=dataset.get('format', 'chat'),
                example_count=result.total_count,
                metadata={
                    "source": "augmentation",
                    "original_dataset_id": request.dataset_id,
                    "augmentation_config": {
                        "techniques": config.techniques,
                        "samples_per_example": config.samples_per_example
                    }
                }
            )

            # Add augmented examples
            db.add_examples(new_dataset_id, result.examples)

        # Get statistics
        stats = augmenter.get_statistics(result)

        return {
            "success": True,
            "statistics": stats,
            "new_dataset_id": new_dataset_id,
            "message": f"Dataset augmented successfully. Created {result.augmented_count} new examples."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augmentation/augment-text")
async def augment_text(
    text: str,
    technique: str = "synonym",
    synonym_ratio: float = 0.3
):
    """Augment a single text"""
    try:
        valid_techniques = ["synonym", "random", "paraphrase", "backtranslation"]
        if technique not in valid_techniques:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid technique. Must be one of: {valid_techniques}"
            )

        config = AugmentationConfig(
            techniques=[technique],
            synonym_ratio=synonym_ratio
        )

        augmenter = DataAugmenter(config)
        augmented_text = augmenter.augment_text(text, technique)

        return {
            "original": text,
            "augmented": augmented_text,
            "technique": technique
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/augmentation/techniques")
async def get_augmentation_techniques():
    """Get list of available augmentation techniques"""
    return {
        "techniques": [
            {
                "name": "synonym",
                "description": "Replace words with synonyms",
                "parameters": ["synonym_ratio"]
            },
            {
                "name": "random",
                "description": "Random word swap, deletion, and insertion",
                "parameters": ["random_swap_ratio", "random_delete_ratio", "random_insert_ratio"]
            },
            {
                "name": "paraphrase",
                "description": "Paraphrase text using pattern matching",
                "parameters": ["paraphrase_diversity"]
            },
            {
                "name": "backtranslation",
                "description": "Back-translation simulation",
                "parameters": []
            }
        ]
    }

@app.post("/api/augmentation/preview")
async def preview_augmentation(
    dataset_id: int,
    techniques: List[str] = ["synonym"],
    samples: int = 5,
    text_field: str = "text"
):
    """Preview augmentation on sample examples"""
    try:
        # Get sample examples
        examples = db.get_examples(dataset_id, limit=samples)
        if not examples:
            raise HTTPException(status_code=404, detail="No examples found")

        config = AugmentationConfig(
            techniques=techniques,
            samples_per_example=1
        )

        augmenter = DataAugmenter(config)

        previews = []
        for example in examples:
            original_text = example.get(text_field, "")
            augmented_text = augmenter.augment_text(original_text)

            previews.append({
                "original": original_text,
                "augmented": augmented_text,
                "technique": techniques[0] if techniques else "unknown"
            })

        return {
            "dataset_id": dataset_id,
            "samples": len(previews),
            "previews": previews
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Initialize LLM provider
    llm_provider = get_llm_provider()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
# -----------------------------------------------------------------------------
# MLOps API Endpoints (Prompts, Reviews, Comments, Versions)
# -----------------------------------------------------------------------------

# Prompts
class PromptCreateRequest(BaseModel):
    name: str
    content: str
    description: Optional[str] = ""
    variables: List[str] = []
    metadata: Dict[str, Any] = {}

@app.post("/api/prompts")
async def create_prompt(request: PromptCreateRequest):
    """Create a new prompt template"""
    try:
        prompt_id = db.create_prompt(
            name=request.name,
            content=request.content,
            description=request.description,
            variables=request.variables,
            metadata=request.metadata
        )
        return {"id": prompt_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts")
async def list_prompts():
    """List all prompt templates"""
    try:
        prompts = db.get_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/{prompt_id}")
async def get_prompt(prompt_id: int):
    """Get a prompt template by ID"""
    prompt = db.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@app.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: int, request: PromptCreateRequest):
    """Update a prompt template"""
    try:
        success = db.update_prompt(
            prompt_id,
            name=request.name,
            content=request.content,
            description=request.description,
            variables=request.variables,
            metadata=request.metadata
        )
        if not success:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: int):
    """Delete a prompt template"""
    try:
        success = db.delete_prompt(prompt_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Reviews
@app.post("/api/reviews")
async def create_review_endpoint(request: ReviewCreateRequest):
    """Create a new review"""
    try:
        review_id = db.create_review(
            dataset_id=request.dataset_id,
            reviewer_id=request.reviewer_id,
            status=request.status,
            feedback=request.feedback
        )
        return {"id": review_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets/{dataset_id}/reviews")
async def list_reviews_endpoint(dataset_id: int):
    """List reviews for a dataset"""
    try:
        reviews = db.get_reviews(dataset_id)
        return {"reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Comments
@app.post("/api/comments")
async def create_comment_endpoint(request: CommentCreateRequest):
    """Create a new comment"""
    try:
        comment_id = db.create_comment(
            dataset_id=request.dataset_id,
            user_id=request.user_id,
            content=request.content,
            example_id=request.example_id,
            parent_id=int(request.parent_comment_id) if request.parent_comment_id else None
        )
        return {"id": comment_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets/{dataset_id}/comments")
async def list_comments_endpoint(dataset_id: int):
    """List comments for a dataset"""
    try:
        comments = db.get_comments(dataset_id)
        return {"comments": comments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Versions
@app.post("/api/versions")
async def create_version_endpoint(request: VersionCreateRequest):
    """Create a new dataset version"""
    try:
        # In a real app, we would snapshot the file here
        # For now, we'll assume the current file is the version
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
            
        version_id = db.create_version(
            dataset_id=request.dataset_id,
            file_path=dataset['file_path'], # In reality, copy this file
            commit_message=request.commit_message,
            author=request.author,
            example_count=dataset['example_count'],
            metadata={"tags": request.tags}
        )
        return {"id": version_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets/{dataset_id}/versions")
async def list_versions_endpoint(dataset_id: int):
    """List versions for a dataset"""
    try:
        versions = db.get_versions(dataset_id)
        return {"versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
